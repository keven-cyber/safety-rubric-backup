"""CPU tests, including the real trainer dump method without requiring Ray/CUDA."""
import ast
import copy
import importlib.util
import json
import math
import os
import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace, ModuleType

import pytest
import numpy as np
import torch
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evidence = load("training_evidence_v4", "verl/utils/training_evidence.py")
dataset = load("prepare_safety_rubric_v4", "scripts/prepare_safety_rubric_v4.py")
runs = load("run_evidence_v4", "scripts/run_evidence.py")
analysis = load("analyze_training_evidence_v4", "scripts/analyze_training_evidence.py")


def record(index, reward, advantage, passes, uid="q"):
    return {"sample_index": index, "uid": uid, "rollout_count": 2,
            "optimization_reward": reward, "advantage_mean": advantage,
            "data_source": "ordinary_harm", "extra_info": {"critical_safety_rule_count": 1},
            "reward_result": {"rubric_rule_passes": {"outcome_safety_rule": passes},
                              "rubric_rules": {"outcome_safety_rule": ["No harmful assistance"]},
                              "judge_parse_valid": True}}


def test_negative_reward_can_have_positive_advantage_and_groups_are_kept(tmp_path):
    rows = [record(0, -1, -0.7, [0]), record(1, -0.6, 0.7, [0]),
            record(2, 0.5, 0, [1], "next")]
    report = evidence.write_step(tmp_path / "rollouts", 4, rows, max_samples=1)
    assert report["saved_responses"] == 2
    assert report["responses"] == 3
    assert report["critical_failed_positive_advantage"] == 1
    assert report["all_critical_failed_groups"] == 1
    assert rows[1]["critical_rule_failed"] is True
    report = analysis.analyze(tmp_path)
    assert report["totals"]["responses"] == 3
    assert report["case_counts"]["critical_failed_positive_advantage"] == 1


def test_invalid_or_missing_critical_judgment_is_not_a_safety_label():
    rows = [record(0, 0, 1, [])]
    evidence.annotate_groups(rows)
    assert rows[0]["critical_rule_failed"] is None
    rows[0]["reward_result"]["rubric_rule_passes"]["outcome_safety_rule"] = [0]
    rows[0]["reward_result"]["judge_parse_valid"] = False
    evidence.annotate_groups(rows)
    assert rows[0]["critical_rule_failed"] is None


def test_redaction_preserves_hyperparameters():
    result = evidence.redact_config({"api_key": "secret", "inner": {"token": "secret", "max_tokens": 20}})
    assert result == {"api_key": "[REDACTED]", "inner": {"token": "[REDACTED]", "max_tokens": 20}}


def sample(profile, split, missing_reasoning=False):
    rules = {k: [k + " specific criterion"] for k in dataset.FIELDS}
    if missing_reasoning:
        rules["reasoning_process_rule"] = []
    request = f"{split} {profile} distinct request"
    info = {**rules, "sample_id": request, "sample_family": request, "prompt": request,
            "reward_profile": profile, "rubric_version": "old", "critical_safety_rule_count": 1,
            "canonical_category": "general"}
    return {**rules, "extra_info": info, "prompt": [{"role": "user", "content": request}], "data_source": profile}


def make_source(tmp_path):
    root = tmp_path / "source"
    for split in ("train", "val"):
        (root / split).mkdir(parents=True)
        for profile, name in dataset.FILES.items():
            row = sample(profile, split, missing_reasoning=split == "val")
            # Force a stable list<string> schema even for empty rule arrays.
            schema = pa.Table.from_pylist([sample(profile, split)]).schema
            pq.write_table(pa.Table.from_pylist([row], schema=schema), root / split / name)
    return root


def test_dataset_preserves_queries_critical_rules_and_safe_training(tmp_path):
    source = make_source(tmp_path)
    output = tmp_path / "v4"
    manifest = dataset.build(source, output)
    assert manifest["validation_missing_reasoning_repaired"] == 4
    assert manifest["revised_harmful_training_rubrics"] == 2
    for profile, name in dataset.FILES.items():
        assert dataset.digest(source / "val" / name) == dataset.digest(output / "val_legacy" / name)
        if profile in ("benign", "sensitive_benign"):
            assert dataset.digest(source / "train" / name) == dataset.digest(output / "train" / name)
        old = pq.read_table(source / "train" / name).to_pylist()[0]
        new = pq.read_table(output / "train" / name).to_pylist()[0]
        assert new["prompt"] == old["prompt"]
        if "harm" in profile:
            assert new["extra_info"]["critical_safety_rule_count"] == 4
            assert set(old["outcome_safety_rule"]) <= set(new["outcome_safety_rule"])
        val = pq.read_table(output / "val" / name).to_pylist()[0]
        assert val["reasoning_process_rule"]
    with pytest.raises(FileExistsError):
        dataset.build(source, output)


def test_dataset_rejects_invalid_critical_count(tmp_path):
    row = sample("ordinary_harm", "train")
    row["extra_info"]["critical_safety_rule_count"] = 0
    with pytest.raises(ValueError):
        dataset.revise(row)


def test_runs_are_unique_and_snapshot_matches_actual_input(tmp_path, monkeypatch):
    monkeypatch.setenv("RUBRIC_JUDGE_API_KEY", "do-not-save")
    first = runs.init(tmp_path / "runs", ROOT)
    second = runs.init(tmp_path / "runs", ROOT)
    assert first != second
    assert "do-not-save" not in (first / "run_manifest.json").read_text()
    source = make_source(tmp_path)
    runs.snapshot(first, ROOT, source, source / "train", source / "val")
    assert len(json.loads((first / "input_manifest.json").read_text())["datasets"]) == 8


def test_primus_run_id_resumes_same_commit_and_rejects_commit_drift(tmp_path, monkeypatch):
    current = runs.git(ROOT, "rev-parse", "HEAD")
    first = runs.init(tmp_path / "runs", ROOT, "primus-test_A", resume_existing=True)
    second = runs.init(tmp_path / "runs", ROOT, "primus-test_A", resume_existing=True)
    assert first == second
    assert json.loads((first / "status.json").read_text())["phase"] == "resuming"

    manifest = json.loads((first / "run_manifest.json").read_text())
    manifest["git_commit"] = "different-commit"
    (first / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="different git commit"):
        runs.init(tmp_path / "runs", ROOT, "primus-test_A", resume_existing=True)

    manifest["git_commit"] = current
    (first / "run_manifest.json").write_text(json.dumps(manifest))


def test_oss_environment_uses_mounted_root_and_fails_on_missing_mount(tmp_path):
    env = os.environ.copy()
    for key in ("RUBRIC_EVIDENCE_INITIALIZED", "RUBRIC_ARTIFACT_ROOT", "RUBRIC_RUN_DIR"):
        env.pop(key, None)
    env.update({"PROJECT_ROOT": str(ROOT), "PRIMUS_OSS_WORKSPACE_DIR": str(tmp_path),
                "RUBRIC_REQUIRE_OSS": "1", "PATH": str(Path(sys.executable).parent) + os.pathsep + env["PATH"]})
    result = subprocess.run(["bash", "-e", "-c", 'source "$PROJECT_ROOT/scripts/rubric_evidence_env.sh"'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    directories = list((tmp_path / "rubricalign_runs").iterdir())
    assert len(directories) == 1
    assert (directories[0] / "run_manifest.json").is_file()
    env["PRIMUS_OSS_WORKSPACE_DIR"] = str(tmp_path / "not-mounted")
    result = subprocess.run(["bash", "-e", "-c", 'source "$PROJECT_ROOT/scripts/rubric_evidence_env.sh"'],
                            env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert not (tmp_path / "not-mounted").exists()


def test_primus_environment_reuses_stable_run_directory(tmp_path):
    env = os.environ.copy()
    for key in ("RUBRIC_EVIDENCE_INITIALIZED", "RUBRIC_ARTIFACT_ROOT", "RUBRIC_RUN_DIR"):
        env.pop(key, None)
    env.update({
        "PROJECT_ROOT": str(ROOT),
        "PRIMUS_OSS_WORKSPACE_DIR": str(tmp_path),
        "PRIMUS_JOB_UUID": "primus-test-job",
        "RUBRIC_TARGETED_EXPERIMENT_ID": "A",
        "RUBRIC_REQUIRE_OSS": "1",
        "PATH": str(Path(sys.executable).parent) + os.pathsep + env["PATH"],
    })
    command = 'source "$PROJECT_ROOT/scripts/rubric_evidence_env.sh"; printf "%s" "$RUBRIC_RUN_DIR"'
    first = subprocess.run(
        ["bash", "-e", "-c", command], env=env, capture_output=True, text=True, check=True
    )
    second = subprocess.run(
        ["bash", "-e", "-c", command], env=env, capture_output=True, text=True, check=True
    )
    assert first.stdout.splitlines()[-1] == second.stdout.splitlines()[-1]
    commit = runs.git(ROOT, "rev-parse", "--short=12", "HEAD")
    assert first.stdout.splitlines()[-1].endswith(f"primus-test-job_A_g{commit}")


def test_primus_environment_isolates_branch_updates_by_commit(tmp_path):
    """A direct restart on new code must not reuse the previous evidence run."""
    script = ROOT / "scripts" / "rubric_evidence_env.sh"
    text = script.read_text(encoding="utf-8")
    assert 'rev-parse --short=12 HEAD' in text
    assert 'stable_run_id="${stable_run_id}_g${evidence_git_commit}"' in text


def test_real_trainer_dump_uses_actual_advantage(tmp_path):
    tree = ast.parse((ROOT / "verl/trainer/ppo/ray_trainer.py").read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RayPPOTrainer")
    klass.body = [n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name in
                  {"_rollout_trace_json_value", "_dump_rollout_traces"}]
    namespace = {"os": os, "np": np, "torch": torch, "DataProto": object, "write_step": evidence.write_step}
    exec(compile(ast.Module(body=[klass], type_ignores=[]), "trainer_dump", "exec"), namespace)
    trainer = namespace["RayPPOTrainer"]()
    trainer.config = SimpleNamespace(trainer={"rollout_trace": {"enabled": True, "dir": str(tmp_path), "max_samples_per_step": 1}})
    trainer.global_steps = 9
    trainer.tokenizer = SimpleNamespace(decode=lambda ids, **kw: " ".join(map(str, ids.tolist())))
    class Batch:
        def __len__(self):
            return 2
    batch = Batch()
    batch.batch = {"prompts": torch.tensor([[0, 11], [0, 11]]),
                   "responses": torch.tensor([[21, 22], [23, 24]]),
                   "attention_mask": torch.tensor([[0, 1, 1, 1], [0, 1, 1, 1]]),
                   "token_level_scores": torch.tensor([[0., -1.], [0., -0.6]]),
                   "token_level_rewards": torch.tensor([[0., -1.], [0., -0.6]]),
                   "advantages": torch.tensor([[-0.7, -0.7], [0.7, 0.7]])}
    batch.non_tensor_batch = {"uid": np.array(["q", "q"], dtype=object),
                              "rollout_n": np.array([2, 2]),
                              "extra_info": np.array([{"critical_safety_rule_count": 1}] * 2, dtype=object)}
    results = [record(i, -1, 0, [0])["reward_result"] for i in range(2)]
    trainer._dump_rollout_traces(batch, 1, [], results)
    saved = [json.loads(x) for x in (tmp_path / "step_00000009.jsonl").read_text().splitlines()]
    assert len(saved) == 2
    assert saved[1]["advantage_mean"] == pytest.approx(0.7)
    assert saved[1]["critical_failed_positive_advantage"] is True
    trainer.config.trainer["rollout_trace"]["enabled"] = False
    trainer.global_steps = 10
    # No batch traversal, tensor conversion or files when tracing is disabled.
    trainer._dump_rollout_traces(None, 1, [], None)
    assert not (tmp_path / "step_00000010.jsonl").exists()


@pytest.fixture
def judge(monkeypatch):
    monkeypatch.setitem(sys.modules, "verl.utils.training_evidence", evidence)
    return load("rubric_judge_v4_test", "verl/utils/reward_score/rubric_judge.py")


def test_parser_distinguishes_invalid_json_from_rule_failure(judge):
    rules = judge.RubricRules(["safety"], [], [], [])
    assert judge._normalize_judge_output({}, rules)["_parse_valid"] is False
    assert judge._normalize_judge_output({"outcome_safety_rule": [0]}, rules)["_parse_valid"] is True
    assert judge._normalize_judge_output({"outcome_safety_rule": [0.5]}, rules)["_parse_valid"] is False


@pytest.mark.parametrize("category", list(dataset.FIELDS))
def test_single_rule_ignores_unsolicited_category_scores(judge, category):
    # Actual Primus failure shape: every category gets a score despite one rule.
    obj = {"outcome_safety_rule": [1], "outcome_useful_rule": [1],
           "reasoning_process_rule": [1], "refusal_rule": [0]}
    rules = judge._single_rule_set(category, "criterion")
    result = judge._normalize_judge_output(obj, rules)
    assert result["_parse_valid"] is True
    for key in dataset.FIELDS:
        assert result[key] == (obj[key] if key == category else [])


@pytest.mark.parametrize("value", [None, [], [1, 1], [0.5], [2], {"score": 1}, "bad"])
def test_requested_rule_remains_strict(judge, value):
    rules = judge._single_rule_set("outcome_useful_rule", "criterion")
    assert judge._normalize_judge_output({"outcome_useful_rule": value}, rules)["_parse_valid"] is False


def test_non_object_and_multirule_validation(judge):
    rules = judge.RubricRules(["a", "b"], ["useful"], [], [])
    assert not judge._normalize_judge_output([], rules)["_parse_valid"]
    assert not judge._normalize_judge_output({"outcome_safety_rule": [1, 0]}, rules)["_parse_valid"]
    assert judge._normalize_judge_output({"outcome_safety_rule": [1, 0], "outcome_useful_rule": [1],
                                         "refusal_rule": [0]}, rules)["_parse_valid"]


def test_guided_output_allows_both_scores_and_exact_lengths(judge, monkeypatch):
    monkeypatch.setenv("RUBRIC_JUDGE_GUIDED_OUTPUT", "1")
    for category in dataset.FIELDS:
        rules = judge._single_rule_set(category, "criterion")
        choices = [json.loads(s) for s in judge._judge_guided_body(rules)["guided_choice"]]
        assert [obj[category] for obj in choices] == [[0], [1]]
        assert all(judge._normalize_judge_output(obj, rules)["_parse_valid"] for obj in choices)
        assert all(obj[key] == [] for obj in choices for key in dataset.FIELDS if key != category)
    rules = judge.RubricRules(["a", "b"], [], [], [])
    import re
    pattern = judge._judge_guided_body(rules)['guided_regex']
    valid = dict(outcome_safety_rule=[0, 1], outcome_useful_rule=[], reasoning_process_rule=[], refusal_rule=[])
    assert re.fullmatch(pattern, json.dumps(valid))
    valid['refusal_rule'] = [0]
    assert not re.fullmatch(pattern, json.dumps(valid))
    monkeypatch.setenv("RUBRIC_JUDGE_GUIDED_OUTPUT", "0")
    assert judge._judge_guided_body(rules) == {}


def test_api_single_rule_merging_and_guidance(judge, monkeypatch):
    monkeypatch.setenv("RUBRIC_JUDGE_GRANULARITY", "single_rule")
    monkeypatch.setenv("RUBRIC_JUDGE_STRICT_PARSE", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_GUIDED_OUTPUT", "1")
    monkeypatch.setattr(judge, "_append_api_output", lambda *args: None)
    calls = []
    def create(**kw):
        calls.append(kw)
        choices = [json.loads(s) for s in kw["extra_body"]["guided_choice"]]
        active = next(key for key, values in choices[0].items() if values)
        # Match the actual guided contract, including empty unused categories.
        obj = {key: [] for key in dataset.FIELDS}
        obj[active] = [0 if active == "outcome_safety_rule" else 1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(obj)))])
    fake = ModuleType("openai")
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", fake)
    api = judge.ApiJudge("http://localhost:8000/v1", "test-secret", "test-judge")
    result = api.judge_batch(["request"], ["answer"], [judge.RubricRules(["a", "b"], ["u"], [], [])])[0]
    assert result["_parse_valid"]
    assert result["outcome_safety_rule"] == [0, 0]
    assert result["outcome_useful_rule"] == [1]
    assert result["refusal_rule"] == []
    assert len(calls) == 3


def test_judge_retries_malformed_output_and_records_both_attempts(judge, tmp_path, monkeypatch):
    monkeypatch.setenv("RUBRIC_INTERMEDIATE_RECORD_ENABLE", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RUBRIC_JUDGE_STRICT_PARSE", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_API_MAX_RETRIES", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_API_RETRY_BACKOFF", "0")
    monkeypatch.setattr(judge, "_append_api_error_output", lambda *args: None)
    monkeypatch.setattr(judge, "_append_api_output", lambda *args: None)
    responses = iter(["not json", '{"outcome_safety_rule": [0]}'])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:
        SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(responses)))]))))
    fake = ModuleType("openai")
    fake.OpenAI = lambda **kw: client
    monkeypatch.setitem(sys.modules, "openai", fake)
    api = judge.ApiJudge("http://localhost:8000/v1", "test-secret", "test-judge")
    result = api.judge("request", "answer", judge.RubricRules(["safe"], [], [], []), {"sample_id": "s", "uid": "u"})
    assert result["outcome_safety_rule"] == [0]
    assert result["_parse_valid"]
    events = [json.loads(line) for path in tmp_path.glob("*.jsonl") for line in path.read_text().splitlines()]
    assert len(events) == 2
    assert events[0]["error_type"] == "ValueError"
    assert events[1]["error_type"] is None
    assert "test-secret" not in json.dumps(events)


@pytest.mark.parametrize("failure", ["malformed", "truncated", "timeout", "exhausted"])
def test_judge_recovery_never_fabricates_reward(judge, monkeypatch, capsys, failure):
    monkeypatch.setenv("RUBRIC_JUDGE_STRICT_PARSE", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_GUIDED_OUTPUT", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_API_MAX_RETRIES", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_API_RETRY_BACKOFF", "0")
    monkeypatch.setenv("RUBRIC_JUDGE_MAX_NEW_TOKENS", "512")
    monkeypatch.setenv("RUBRIC_JUDGE_LOAD_BALANCE", "sticky")
    monkeypatch.setenv("RUBRIC_INTERMEDIATE_RECORD_ENABLE", "0")
    monkeypatch.setattr(judge, "_append_api_error_output", lambda *args: None)
    monkeypatch.setattr(judge, "_append_api_output", lambda *args: None)
    requests, endpoints = [], []

    def make_client(**options):
        assert options["max_retries"] == 0
        assert options["timeout"] > 0
        def create(**request):
            endpoints.append(options["base_url"])
            requests.append(request)
            first = len(requests) == 1
            if first and failure == "timeout":
                raise TimeoutError("service timeout")
            text = "not JSON" if (first or failure == "exhausted") else json.dumps(dict(
                outcome_safety_rule=[0], outcome_useful_rule=[], reasoning_process_rule=[], refusal_rule=[]))
            reason = "length" if first and failure == "truncated" else "stop"
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content=text), finish_reason=reason)])
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    fake = ModuleType("openai")
    fake.OpenAI = make_client
    monkeypatch.setitem(sys.modules, "openai", fake)
    api = judge.ApiJudge("http://replica-a/v1,http://replica-b/v1", "secret", "judge")
    rules = judge.RubricRules(["safe"], [], [], [])
    if failure == "exhausted":
        with pytest.raises(RuntimeError, match="no reward was fabricated"):
            api.judge("request", "answer", rules)
    else:
        result = api.judge("request", "answer", rules)
        assert result["_parse_valid"] and result["outcome_safety_rule"] == [0]
    assert len(requests) == 2 and endpoints[0] != endpoints[1]
    assert all("guided_choice" in r["extra_body"] for r in requests)
    if failure != "timeout":
        assert "Required array lengths" in requests[1]["messages"][0]["content"]
    assert requests[1]["max_tokens"] == (1024 if failure == "truncated" else 512)
    log = capsys.readouterr().out
    assert "[RUBRIC_JUDGE][RETRY]" in log and "secret" not in log


def test_all_experiments_enable_guidance_with_strict_parsing():
    experiments = load("rubric_experiments_guidance_test", "scripts/rubric_experiments.py")
    for arm in experiments.ARMS:
        env = experiments.configuration(arm)["environment"]
        assert env["RUBRIC_JUDGE_STRICT_PARSE"] == env["RUBRIC_JUDGE_GUIDED_OUTPUT"] == "1"


def test_judge_drop_marks_invalid_without_hiding_configuration_errors(judge, monkeypatch):
    monkeypatch.setenv("RUBRIC_JUDGE_STRICT_PARSE", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_DROP_FAILED", "1")
    monkeypatch.setenv("RUBRIC_JUDGE_API_MAX_RETRIES", "0")
    monkeypatch.setenv("RUBRIC_INTERMEDIATE_RECORD_ENABLE", "0")
    monkeypatch.setattr(judge, "_append_api_error_output", lambda *a: None)
    monkeypatch.setattr(judge, "_append_api_output", lambda *a: None)
    def create(**kw):
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="bad JSON"))])
    fake = ModuleType("openai")
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, "openai", fake)
    api = judge.ApiJudge("http://replica/v1", "secret", "judge")
    rules = judge.RubricRules(["safe"], [], [], [])
    monkeypatch.setenv("RUBRIC_JUDGE_GRANULARITY", "single_rule")
    result = api.judge_batch(["request"], ["answer"], [rules])[0]
    assert result["_judge_failed"] and not result["_parse_valid"]
    monkeypatch.setenv("RUBRIC_JUDGE_LOAD_BALANCE", "invalid-config")
    with pytest.raises(RuntimeError, match="no reward was fabricated"):
        api.judge("request", "answer", rules)


def test_failed_rollout_excludes_entire_interleaved_group():
    filt = load("judge_filter_test", "verl/utils/judge_filter.py")
    uids = ["a", "b", "c", "a", "b", "c"]
    keep, dropped = filt.valid_group_indices(uids, [False, True, False, False, False, False], 2)
    assert keep == [0, 2, 3, 5] and dropped == 1
    with pytest.raises(RuntimeError, match="empty update"):
        filt.valid_group_indices(["a", "a"], [False, True])
    with pytest.raises(RuntimeError, match="actor ranks"):
        filt.valid_group_indices(uids, [False, True, False, False, False, False], 3)
    with pytest.raises(ValueError, match="align"):
        filt.valid_group_indices(uids, [False])


def test_reward_extras_remain_aligned_when_profiles_expose_different_metrics():
    filt = load("judge_filter_dense_extras_test", "verl/utils/judge_filter.py")
    extras = {}
    rows = [
        {"judge_failed": False, "safety": 1.0},
        {"judge_failed": True, "boundary": 0.0},
        {"judge_failed": False, "safety": 0.5},
        {"judge_failed": False, "boundary": 1.0},
    ]
    for index, row in enumerate(rows):
        filt.append_dense_reward_extras(extras, row, index)

    assert all(len(values) == len(rows) for values in extras.values())
    assert extras["judge_failed"] == [False, True, False, False]
    assert math.isnan(extras["safety"][1]) and math.isnan(extras["boundary"][0])

    selected, skipped = filt.select_aligned_reward_extras(extras, [0, 2], len(rows))
    assert not skipped
    assert selected["judge_failed"] == [False, False]
    assert selected["safety"] == [1.0, 0.5]


def test_group_filter_quarantines_legacy_sparse_optional_metrics():
    filt = load("judge_filter_sparse_extras_test", "verl/utils/judge_filter.py")
    extras = {
        "judge_failed": [False, True, False, False],
        "aligned": [10, 20, 30, 40],
        "legacy_sparse": [1],
    }
    selected, skipped = filt.select_aligned_reward_extras(extras, [0, 2, 3], 4)
    assert selected == {
        "judge_failed": [False, False, False],
        "aligned": [10, 30, 40],
    }
    assert skipped == {"legacy_sparse": 1}


@pytest.mark.parametrize("setting", [None, "0"])
def test_disabled_judge_evidence_never_locks_or_writes(judge, tmp_path, monkeypatch, setting):
    monkeypatch.delenv("RUBRIC_INTERMEDIATE_RECORD_ENABLE", raising=False)
    if setting is not None:
        monkeypatch.setenv("RUBRIC_INTERMEDIATE_RECORD_ENABLE", setting)
    monkeypatch.setenv("RUBRIC_JUDGE_TRACE_DIR", str(tmp_path))
    def forbidden(*args, **kwargs):
        pytest.fail("disabled evidence must not serialize or write")
    monkeypatch.setattr(judge, "_normalize_extra_info", forbidden)
    monkeypatch.setattr(judge, "append_event", forbidden)
    judge._judge_evidence("request", "answer", None, None, "raw", "model", 1, 0.1)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("initialized", [False, True])
@pytest.mark.parametrize("enabled", ["0", "1"])
def test_intermediate_switch_overrides_stale_logging_flags(tmp_path, initialized, enabled):
    env = os.environ.copy()
    env.update({"PROJECT_ROOT": str(ROOT), "PRIMUS_OSS_WORKSPACE_DIR": str(tmp_path),
                "RUBRIC_EVIDENCE_INITIALIZED": "1" if initialized else "0",
                "RUBRIC_INTERMEDIATE_RECORD_ENABLE": enabled,
                "RUBRIC_RUN_DIR": str(tmp_path / "existing"),
                "ROLLOUT_TRACE_DIR": str(tmp_path / "existing" / "rollouts"),
                "RUBRIC_JUDGE_TRACE_DIR": str(tmp_path / "stale-judge"),
                "ROLLOUT_TRACE_ENABLE": "1", "RUBRIC_JUDGE_API_LOG_ENABLE": "1",
                "PATH": str(Path(sys.executable).parent) + os.pathsep + env["PATH"]})
    env.pop("RUBRIC_ARTIFACT_ROOT", None)
    check = ('test "$ROLLOUT_TRACE_ENABLE" = 0 && test "$RUBRIC_JUDGE_API_LOG_ENABLE" = 0 '
             '&& test -z "${RUBRIC_JUDGE_TRACE_DIR:-}"') if enabled == "0" else (
             'test "$ROLLOUT_TRACE_ENABLE" = 1 && test -d "$RUBRIC_JUDGE_TRACE_DIR"')
    result = subprocess.run(["bash", "-eu", "-c",
                             'source "$PROJECT_ROOT/scripts/rubric_evidence_env.sh"; ' + check],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
