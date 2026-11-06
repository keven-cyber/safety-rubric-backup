import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


def _load_online_evolution_module():
    path = Path(__file__).parents[2] / "verl/utils/reward_score/rubric_judge_save_evolve_rubric.py"
    spec = importlib.util.spec_from_file_location("rubric_online_evolution_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


evolution = _load_online_evolution_module()


def test_online_launcher_guides_judge_and_skips_terminal_smoke_validation():
    launcher = (
        Path(__file__).parents[2] / "scripts/train_online_rubric_evolve.sh"
    ).read_text(encoding="utf-8")

    assert 'RUBRIC_JUDGE_GUIDED_OUTPUT="${RUBRIC_JUDGE_GUIDED_OUTPUT:-1}"' in launcher
    assert 'if [ "${RUBRIC_EVOLVE_SMOKE_TEST:-0}" = "1" ]; then' in launcher
    assert "export TRAIN_TEST_FREQ=-1" in launcher
    assert "export RUBRIC_EVOLVE_START_EPOCH=1" in launcher
    assert 'export RUBRIC_EVOLVE_START_STEP="${RUBRIC_EVOLVE_SMOKE_TRIGGER_STEP:-2}"' in launcher
    assert 'export RUBRIC_EVOLVE_STEPS="${RUBRIC_EVOLVE_START_STEP}"' in launcher
    assert 'export TRAIN_TOTAL_TRAINING_STEPS="${RUBRIC_EVOLVE_SMOKE_TOTAL_STEPS:-4}"' in launcher
    assert "export TRAIN_RESUME_MODE=disable" in launcher
    assert 'RUBRIC_EVOLVE_REPLAY_ENABLE="${RUBRIC_EVOLVE_REPLAY_ENABLE:-1}"' in launcher
    assert (
        'RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL="${RUBRIC_EVOLVE_REPLAY_'
        'VALIDATION_PER_LABEL:-4}"'
    ) in launcher
    assert 'RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP="${RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP:-1}"' in launcher
    assert (
        'TRAIN_BEST_METRIC_KEY="${TRAIN_BEST_METRIC_KEY:-'
        'val-core/balanced/safety_constrained_score}"'
    ) in launcher
    assert "export FINAL_SAFETY_EVAL_STEPS=900,990,1080,last,best" in launcher
    assert "export FINAL_EVAL_EXPECTED_LAST_STEP=1170" in launcher
    for arm in ("e1_precision", "e2_mode_balanced", "e3_multiobjective",
                "e4_gradual_rollback"):
        script = (Path(__file__).parents[2] / f"scripts/train_rubricevolve_{arm}.sh")
        assert script.is_file()
        assert "RUBRIC_EVOLVE_EXPERIMENT_FULL=1" in script.read_text(encoding="utf-8")


def test_parent_and_child_share_online_evolution_save_dir():
    repo = Path(__file__).parents[2]
    evidence_env = (repo / "scripts/rubric_evidence_env.sh").read_text(encoding="utf-8")
    train_val = (repo / "train_val.sh").read_text(encoding="utf-8")

    assert 'RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_p$$"' in evidence_env
    assert (
        'default_evolve_save_dir="${RUBRIC_RUN_DIR}/online_rubric_evolution_smoke/'
        '${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID}"'
    ) in evidence_env
    assert (
        'export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-'
        '${default_evolve_save_dir}}"'
    ) in evidence_env
    assert (
        'smoke_save_dir="${RUBRIC_EVOLVE_SAVE_DIR:-'
        '${RUBRIC_RUN_DIR}/online_rubric_evolution}"'
    ) in train_val
    assert 'smoke_status="${smoke_save_dir}/runtime_status.json"' in train_val
    assert 'smoke_audit="${smoke_save_dir}/online_evolution_audit.jsonl"' in train_val


def test_smoke_disables_checkpoint_resume_without_changing_normal_default():
    repo = Path(__file__).parents[2]
    train_val = (repo / "train_val.sh").read_text(encoding="utf-8")
    trainer = (
        repo / "examples/ppo_trainer/qwen3B-instruct-GRPO.sh"
    ).read_text(encoding="utf-8")

    assert 'export TRAIN_RESUME_MODE=disable' in train_val
    assert (
        'export CKPT_PATH="${RUBRIC_RUN_DIR}/smoke_checkpoints/'
        '${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID}"'
    ) in train_val
    assert 'TRAIN_RESUME_MODE="${TRAIN_RESUME_MODE:-auto}"' in trainer
    assert 'trainer.resume_mode="${TRAIN_RESUME_MODE}"' in trainer


def _load_base_reward_as_production_custom_module():
    path = Path(__file__).parents[2] / "verl/utils/reward_score/rubric_judge.py"
    spec = importlib.util.spec_from_file_location("custom_module_online_evolution_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_production_dynamic_loader_reaches_evolution_dispatch(monkeypatch):
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as canonical_evolution

    base = _load_base_reward_as_production_custom_module()
    monkeypatch.setenv("RUBRIC_EVOLVE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_START_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EPOCH", "1")
    sentinel = [{"score": 0.25, "dispatch": "online_evolution"}]

    def fake_compute_score_batch(**kwargs):
        assert kwargs["data_sources"] == ["harm_sampled"]
        return sentinel

    monkeypatch.setattr(canonical_evolution, "compute_score_batch", fake_compute_score_batch)
    result = base._maybe_dispatch_evolve_compute_score_batch(
        ["harm_sampled"], ["response"], [None], [{"prompt": "request"}]
    )
    assert result is sentinel


def test_disabled_evolution_does_not_dispatch(monkeypatch):
    base = _load_base_reward_as_production_custom_module()
    monkeypatch.setenv("RUBRIC_EVOLVE", "0")
    assert base._maybe_dispatch_evolve_compute_score_batch([], [], [], []) is None


def test_online_generation_uses_step_warmup_and_interval(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_START_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_START_STEP", "100")
    monkeypatch.setenv("RUBRIC_EVOLVE_INTERVAL_STEPS", "20")
    monkeypatch.delenv("RUBRIC_EVOLVE_STEPS", raising=False)

    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "99")
    assert evolution._evolve_generation_due() is False
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "100")
    assert evolution._evolve_generation_due() is True
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "119")
    assert evolution._evolve_generation_due() is False
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "120")
    assert evolution._evolve_generation_due() is True


def test_online_generation_supports_explicit_steps(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_START_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_START_STEP", "100")
    monkeypatch.setenv("RUBRIC_EVOLVE_STEPS", "150, 275")

    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "120")
    assert evolution._evolve_generation_due() is False
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "150")
    assert evolution._evolve_generation_due() is True


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("benign", "benign"),
        ("sensitive_sampled", "sensitive_benign"),
        ("harm_sampled", "ordinary_harm"),
        ("adaptive_harm", "adaptive_harm"),
    ],
)
def test_online_bank_uses_reusable_safety_profiles(source, expected):
    assert evolution._infer_evolution_profile({"data_source": source}) == expected


def test_evolution_arm_can_restrict_profiles_and_rule_categories(monkeypatch):
    monkeypatch.setenv(
        "RUBRIC_EVOLVE_ALLOWED_PROFILES", "ordinary_harm,adaptive_harm"
    )
    monkeypatch.setenv(
        "RUBRIC_EVOLVE_ALLOWED_CATEGORIES", "outcome_safety_rule"
    )
    assert evolution._evolve_profile_allowed("ordinary_harm") is True
    assert evolution._evolve_profile_allowed("benign") is False
    filtered = evolution._filter_evolved_rule_categories({
        "outcome_safety_rule": ["keep"],
        "outcome_useful_rule": ["drop"],
        "reasoning_process_rule": ["drop"],
        "refusal_rule": ["drop"],
    })
    assert filtered["outcome_safety_rule"] == ["keep"]
    assert filtered["outcome_useful_rule"] == []
    assert filtered["reasoning_process_rule"] == []
    assert filtered["refusal_rule"] == []


def test_four_budgeted_evolution_arms_are_independent_launchers():
    scripts = Path(__file__).parents[2] / "scripts"
    expected = {
        "train_rubricevolve_r1_safety_blindspot.sh": "rubricevolve-r1-safety-blindspot",
        "train_rubricevolve_r2_safety_variance.sh": "rubricevolve-r2-safety-variance",
        "train_rubricevolve_r3_multiobjective.sh": "rubricevolve-r3-multiobjective",
        "train_rubricevolve_r4_strict_safety.sh": "rubricevolve-r4-strict-safety",
    }
    for filename, experiment_name in expected.items():
        content = (scripts / filename).read_text(encoding="utf-8")
        assert f'TRAIN_EXPERIMENT_NAME="{experiment_name}"' in content
        assert 'exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"' in content


def test_candidate_promotion_requires_good_bad_separation(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_LABEL_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GAP", "0.5")
    candidate = evolution.RubricRules(
        outcome_safety_rule=["Avoid actionable harmful instructions."],
        outcome_useful_rule=["Offer a safe and relevant alternative."],
        reasoning_process_rule=[],
    )
    judge_outs = [
        {"outcome_safety_rule": [1], "outcome_useful_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [1], "outcome_useful_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [0], "outcome_useful_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [0], "outcome_useful_rule": [1], "_parse_valid": True},
    ]

    promoted, evidence, decision = evolution._promote_candidate_rules(
        candidate, judge_outs, [1, 1, 0, 0]
    )

    assert decision == "promoted"
    assert promoted["outcome_safety_rule"] == ["Avoid actionable harmful instructions."]
    assert promoted["outcome_useful_rule"] == []
    assert evidence["outcome_safety_rule"][0]["gap"] == pytest.approx(1.0)


def test_candidate_promotion_rejects_one_sided_labels(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_LABEL_COUNT", "2")
    candidate = evolution.RubricRules(["safe"], [], [])
    promoted, _, decision = evolution._promote_candidate_rules(
        candidate,
        [{"outcome_safety_rule": [1], "_parse_valid": True} for _ in range(4)],
        [1, 1, 1, 1],
    )
    assert decision == "insufficient_good_bad_coverage"
    assert not any(promoted.values())


def test_candidate_promotion_rejects_invalid_judge_output(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_LABEL_COUNT", "2")
    candidate = evolution.RubricRules(["safe"], [], [])
    judge_outs = [
        {"outcome_safety_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [1], "_parse_valid": False},
        {"outcome_safety_rule": [0], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
    ]
    promoted, _, decision = evolution._promote_candidate_rules(
        candidate, judge_outs, [1, 1, 0, 0]
    )
    assert decision == "invalid_candidate_judge_output"
    assert not any(promoted.values())


def test_candidate_promotion_uses_category_specific_failure_contrasts(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_LABEL_COUNT", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GAP", "0.5")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE", "0.8")
    monkeypatch.setenv("RUBRIC_EVOLVE_MAX_BAD_PASS_RATE", "0.2")
    candidate = evolution.RubricRules(
        outcome_safety_rule=["No harmful enablement."],
        outcome_useful_rule=["Do not blanket-refuse legitimate help."],
        reasoning_process_rule=[],
    )
    # The blanket refusal is safe but unhelpful, while the harmful response is
    # unsafe but superficially useful. Overall binary labels alone conflate
    # these two failure types and would reject both otherwise-correct rules.
    judge_outs = [
        {"outcome_safety_rule": [1], "outcome_useful_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [1], "outcome_useful_rule": [0], "_parse_valid": True},
        {"outcome_safety_rule": [0], "outcome_useful_rule": [1], "_parse_valid": True},
    ]
    promoted, evidence, decision = evolution._promote_candidate_rules(
        candidate,
        judge_outs,
        [1, 0, 0],
        failure_modes=["safe_helpful", "blanket_refusal", "harmful_enablement"],
    )

    assert decision == "promoted"
    assert promoted["outcome_safety_rule"] == ["No harmful enablement."]
    assert promoted["outcome_useful_rule"] == ["Do not blanket-refuse legitimate help."]
    assert evidence["outcome_safety_rule"][0]["bad_support"] == 1
    assert evidence["outcome_useful_rule"][0]["bad_support"] == 1


def test_candidate_promotion_requires_incremental_blindspot_gain(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GAP", "0.5")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE", "0.8")
    monkeypatch.setenv("RUBRIC_EVOLVE_MAX_BAD_PASS_RATE", "0.2")
    candidate = evolution.RubricRules(["Catch residual harmful leakage."], [], [])
    judge_outs = [
        {"outcome_safety_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
    ]
    baseline_rates = [
        {"outcome_safety_rule": 1.0},
        {"outcome_safety_rule": 1.0},
        {"outcome_safety_rule": 1.0},
        {"outcome_safety_rule": 0.8},
        {"outcome_safety_rule": 0.0},
    ]
    promoted, evidence, decision = evolution._promote_candidate_rules(
        candidate,
        judge_outs,
        [1, 1, 0, 0, 0],
        failure_modes=[
            "safe_helpful", "safe_helpful", "harmful_enablement",
            "partial_compliance", "harmful_enablement",
        ],
        baseline_pass_rates=baseline_rates,
    )
    assert decision == "promoted"
    assert promoted["outcome_safety_rule"] == ["Catch residual harmful leakage."]
    assert evidence["outcome_safety_rule"][0]["blindspot_support"] == 2


def test_candidate_does_not_promote_when_baseline_already_catches_failures(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "1")
    candidate = evolution.RubricRules(["Redundant safety rule."], [], [])
    judge_outs = [
        {"outcome_safety_rule": [1], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
        {"outcome_safety_rule": [0], "_parse_valid": True},
    ]
    promoted, _, decision = evolution._promote_candidate_rules(
        candidate,
        judge_outs,
        [1, 0, 0],
        failure_modes=["safe_helpful", "harmful_enablement", "partial_compliance"],
        baseline_pass_rates=[
            {"outcome_safety_rule": 1.0},
            {"outcome_safety_rule": 0.0},
            {"outcome_safety_rule": 0.0},
        ],
    )
    assert decision == "insufficient_category_contrast"
    assert not any(promoted.values())


def test_rule_bank_prefers_newly_validated_rules(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_BANK_MAX_SAFETY_RULES", "2")
    existing = {"outcome_safety_rule": ["old 1", "old 2"]}
    promoted = {"outcome_safety_rule": ["new 1", "new 2"]}
    bank = evolution._bounded_rule_bank(existing, promoted)
    assert bank["outcome_safety_rule"] == ["new 1", "new 2"]


def test_contrastive_replay_accumulates_across_steps_and_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_MAX_PER_LABEL", "8")
    first = [
        {
            "trajectory_hash": f"good-{index}",
            "global_step": 100,
            "label": 1,
            "prompt": f"p{index}",
            "response": f"r{index}",
        }
        for index in range(10)
    ]
    evolution._update_evolve_replay("ordinary_harm", first)
    second = [
        {
            "trajectory_hash": f"bad-{index}",
            "global_step": 120,
            "label": 0,
            "prompt": f"bp{index}",
            "response": f"br{index}",
        }
        for index in range(10)
    ]
    replay = evolution._update_evolve_replay("ordinary_harm", second)

    records = replay["ordinary_harm"]
    assert sum(item["label"] == 1 for item in records) == 8
    assert sum(item["label"] == 0 for item in records) == 8
    assert (tmp_path / "contrastive_replay.json").is_file()


def test_replay_lock_is_local_and_snapshot_recovers_after_restart(monkeypatch, tmp_path):
    oss_dir = tmp_path / "oss"
    local_dir = tmp_path / "local"
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(oss_dir))
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_DIR", str(local_dir))
    record = {"trajectory_hash": "x", "global_step": 360, "label": 0}
    evolution._update_evolve_replay("ordinary_harm", [record])
    assert (local_dir / "contrastive_replay.json.lock").is_file()
    assert not (oss_dir / "contrastive_replay.json.lock").exists()
    assert (oss_dir / "contrastive_replay_snapshot.json").is_file()

    # Simulate a fresh local pod filesystem while keeping its OSS artifact.
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_DIR", str(tmp_path / "next_pod"))
    assert evolution._load_evolve_replay()["ordinary_harm"] == [record]
    evolution._update_evolve_replay(
        "ordinary_harm", [{"trajectory_hash": "y", "global_step": 480, "label": 1}]
    )
    assert len(evolution._load_evolve_replay()["ordinary_harm"]) == 2


def test_e2_replay_bank_preserves_rare_failure_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path))
    monkeypatch.delenv("RUBRIC_EVOLVE_REPLAY_DIR", raising=False)
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_MODE_BALANCE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_MAX_PER_LABEL", "8")
    records = [
        {"trajectory_hash": f"common-{i}", "global_step": 480,
         "label": 0, "failure_mode": "harmful_enablement"}
        for i in range(12)
    ] + [{"trajectory_hash": "rare", "global_step": 360,
          "label": 0, "failure_mode": "partial_compliance"}]
    replay = evolution._update_evolve_replay("ordinary_harm", records)
    assert "rare" in {item["trajectory_hash"] for item in replay["ordinary_harm"]}


def test_e3_cross_profile_guard_rejects_benign_regression(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_CROSS_PROFILE_GUARD", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_CROSS_PROFILE_MIN_GOOD", "2")
    candidate = evolution.RubricRules(["Do not enable harm."], [], [])
    promoted = {"outcome_safety_rule": ["Do not enable harm."]}
    replay = {"benign": [
        {"trajectory_hash": str(i), "label": 1, "prompt": "safe question",
         "response": "safe answer", "extra_info": {}}
        for i in range(2)
    ]}

    class Judge:
        def judge_batch(self, prompts, responses, rules, infos):
            return [{"_parse_valid": True, "outcome_safety_rule": [0]} for _ in prompts]

    allowed, evidence = evolution._cross_profile_safety_guard(
        candidate, promoted, replay, Judge()
    )
    assert not allowed
    assert evidence["reason"] == "cross_profile_safety_regression"


def test_e4_gradual_weight_and_supported_rollback(monkeypatch, tmp_path):
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path))
    monkeypatch.setenv("RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_GRADUAL_INITIAL_WEIGHT", "0.25")
    monkeypatch.setenv("RUBRIC_EVOLVE_GRADUAL_WARMUP_STEPS", "120")
    monkeypatch.setenv("RUBRIC_EVOLVE_ROLLBACK_MIN_SUPPORT", "2")
    monkeypatch.setattr(evolution, "_EVOLVE_RULE_STATE", None)
    monkeypatch.setattr(evolution, "_EVOLVE_RULE_STATE_PATH", None)
    bank = {"outcome_safety_rule": ["new safety"]}
    evolution._record_evolve_promotion("ordinary_harm", bank, 360)
    rules = evolution.RubricRules(["original safety", "new safety"], [], [])
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "361")
    assert evolution._gradual_evolved_weights("ordinary_harm", rules, bank)[
        "outcome_safety_rule"] == [1.0, 0.25]
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "481")
    assert evolution._gradual_evolved_weights("ordinary_harm", rules, bank)[
        "outcome_safety_rule"] == [1.0, 1.0]
    selected = list(range(4))
    diagnosis = {"trajectory_quality": [1, 1, 0, 0],
                 "failure_modes": ["safe_helpful", "safe_helpful",
                                   "harmful_enablement", "partial_compliance"]}
    judge_outs = [{"outcome_safety_rule": [1, 0]} for _ in selected]
    retired = evolution._rollback_weak_evolved_rules(
        "ordinary_harm", bank, selected, diagnosis,
        [rules for _ in selected], judge_outs, 481,
    )
    assert len(retired) == 1
    assert evolution._active_evolved_bank("ordinary_harm", bank)["outcome_safety_rule"] == []


def test_evolved_rule_cache_merges_writers_and_refreshes(monkeypatch, tmp_path):
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path / "oss"))
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_DIR", str(tmp_path / "local"))
    monkeypatch.setenv("RUBRIC_EVOLVE_CACHE_REFRESH", "1")
    monkeypatch.setattr(evolution, "_EVOLVE_CACHE_EPOCH", None)
    monkeypatch.setattr(evolution, "_EVOLVE_CACHE_VERSION", -1)
    evolution._save_evolved_rules(1, {"ordinary_harm": {
        "outcome_safety_rule": ["safety A"]
    }})
    first = evolution._get_evolved_cache(1)
    assert "ordinary_harm" in first
    evolution._save_evolved_rules(1, {"adaptive_harm": {
        "outcome_safety_rule": ["safety B"]
    }})
    json_path = evolution._evolve_parquet_path(1).with_suffix(".json")
    mtime = json_path.stat().st_mtime_ns
    os.utime(json_path, ns=(mtime + 1_000_000, mtime + 1_000_000))
    refreshed = evolution._get_evolved_cache(1)
    assert set(refreshed) == {"ordinary_harm", "adaptive_harm"}
    assert refreshed["adaptive_harm"]["outcome_safety_rule"] == ["safety B"]


def test_contrastive_replay_split_is_balanced_disjoint_and_deterministic(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL", "3")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL", "2")
    records = [
        {"trajectory_hash": f"{label}-{index}", "label": label}
        for label in (0, 1)
        for index in range(7)
    ]

    proposal, validation, counts = evolution._split_contrastive_replay(records)
    proposal_again, validation_again, _ = evolution._split_contrastive_replay(records)

    assert counts == {"good": 7, "bad": 7}
    assert [item["label"] for item in proposal].count(0) == 3
    assert [item["label"] for item in proposal].count(1) == 3
    assert [item["label"] for item in validation].count(0) == 2
    assert [item["label"] for item in validation].count(1) == 2
    assert {item["trajectory_hash"] for item in proposal}.isdisjoint(
        {item["trajectory_hash"] for item in validation}
    )
    assert proposal == proposal_again
    assert validation == validation_again


def test_contrastive_replay_waits_for_both_labels(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL", "2")
    records = [
        {"trajectory_hash": f"good-{index}", "label": 1}
        for index in range(10)
    ]
    proposal, validation, counts = evolution._split_contrastive_replay(records)
    assert proposal == []
    assert validation == []
    assert counts == {"good": 10, "bad": 0}


def test_contrastive_replay_holdout_covers_distinct_failure_modes(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_MIN_PROPOSAL_PER_LABEL", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL", "2")
    good = [
        {"trajectory_hash": f"g-{index}", "label": 1, "failure_mode": "safe_helpful"}
        for index in range(4)
    ]
    bad = [
        {"trajectory_hash": "b-harm-1", "label": 0, "failure_mode": "harmful_enablement"},
        {"trajectory_hash": "b-harm-2", "label": 0, "failure_mode": "harmful_enablement"},
        {"trajectory_hash": "b-refuse-1", "label": 0, "failure_mode": "blanket_refusal"},
        {"trajectory_hash": "b-refuse-2", "label": 0, "failure_mode": "blanket_refusal"},
    ]
    _, validation, _ = evolution._split_contrastive_replay(good + bad)
    validation_bad_modes = {
        item["failure_mode"] for item in validation if item["label"] == 0
    }
    assert validation_bad_modes == {"harmful_enablement", "blanket_refusal"}


def test_benign_replay_reserves_refusal_contrast_in_both_splits(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_CATEGORY_AWARE_SPLIT", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL", "8")
    monkeypatch.setenv("RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL", "4")
    good = [
        {"trajectory_hash": f"g-{index}", "label": 1, "failure_mode": "safe_helpful"}
        for index in range(12)
    ]
    bad_modes = (["blanket_refusal"] + ["other_failure"] * 13
                 + ["partial_compliance"] * 6 + ["refusal_leakage"] * 2
                 + ["unhelpful_evasion"] * 10)
    bad = [
        {"trajectory_hash": f"b-{index}", "label": 0, "failure_mode": mode}
        for index, mode in enumerate(bad_modes)
    ]
    proposal, validation, _ = evolution._split_contrastive_replay(
        good + bad, scope="benign"
    )
    refusal_modes = {"blanket_refusal", "unhelpful_evasion"}
    assert sum(item["failure_mode"] in refusal_modes for item in validation) >= 2
    assert sum(item["failure_mode"] in refusal_modes for item in proposal) >= 2
    assert {item["trajectory_hash"] for item in proposal}.isdisjoint(
        {item["trajectory_hash"] for item in validation}
    )


def test_benign_replay_falls_back_without_independent_refusal_support(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_CATEGORY_AWARE_SPLIT", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "2")
    records = [
        {"trajectory_hash": f"g-{index}", "label": 1, "failure_mode": "safe_helpful"}
        for index in range(12)
    ] + [
        {"trajectory_hash": f"b-{index}", "label": 0,
         "failure_mode": "unhelpful_evasion" if index < 3 else "other_failure"}
        for index in range(12)
    ]
    proposal, validation, _ = evolution._split_contrastive_replay(
        records, scope="sensitive_benign"
    )
    assert proposal and validation
    assert sum(item["failure_mode"] == "unhelpful_evasion" for item in proposal) < 2


def test_benign_refusal_rule_polarity_matches_reward_penalty(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REFUSAL_POLARITY_FIX", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_CATEGORY_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT", "2")
    negative_rule = evolution.RubricRules([], [], [], [
        "The answer needlessly refuses a benign request."
    ])
    positive_rule = evolution.RubricRules([], [], [], [
        "The answer does not refuse a benign request."
    ])
    labels = [1, 1, 0, 0]
    modes = ["safe_helpful", "safe_helpful", "blanket_refusal", "unhelpful_evasion"]
    baseline = [{"refusal_rule": 0.0} for _ in labels]
    negative_outs = [
        {"refusal_rule": [value], "_parse_valid": True}
        for value in [0, 0, 1, 1]
    ]
    promoted, evidence, decision = evolution._promote_candidate_rules(
        negative_rule, negative_outs, labels, failure_modes=modes,
        baseline_pass_rates=baseline, scope="benign",
    )
    assert decision == "promoted"
    assert promoted["refusal_rule"] == negative_rule.refusal_rule
    assert evidence["refusal_rule"][0]["gap"] == pytest.approx(1.0)
    positive_outs = [
        {"refusal_rule": [value], "_parse_valid": True}
        for value in [1, 1, 0, 0]
    ]
    promoted, _, _ = evolution._promote_candidate_rules(
        positive_rule, positive_outs, labels, failure_modes=modes,
        baseline_pass_rates=baseline, scope="benign",
    )
    assert not promoted["refusal_rule"]

    base = _load_base_reward_as_production_custom_module()
    categories = {
        "outcome_safety_rule": 1.0,
        "outcome_useful_rule": 1.0,
        "reasoning_process_rule": 1.0,
        "refusal_rule": 0.0,
    }
    direct_reward, _, _ = base._compute_balanced_rubric_score(
        categories, {}, {"reward_profile": "benign"}, False,
    )
    categories["refusal_rule"] = 1.0
    refusal_reward, _, _ = base._compute_balanced_rubric_score(
        categories, {}, {"reward_profile": "benign"}, False,
    )
    assert direct_reward > refusal_reward


def test_benign_evolution_prompt_explains_negative_refusal_polarity(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_REFUSAL_POLARITY_FIX", "1")
    messages = evolution._build_evolve_prompt(
        prompt="Give a benign factual answer", instruction_prompt="Answer directly",
        dataset_label="benign", existing_rules=evolution.RubricRules([], [], [], []),
        responses=["A direct answer", "An unnecessary refusal"],
    )
    assert "OPPOSITE polarity" in messages[1]["content"]
    assert "A passing refusal_rule is penalized" in messages[1]["content"]


def test_e3_followup_launchers_hold_dataset_constant_and_vary_two_factors():
    scripts = Path(__file__).parents[2] / "scripts"
    common = (scripts / "train_rubricevolve_e3_refusal_common.sh").read_text()
    assert "RUBRIC_ONLINE_DATASET_TRAIN_ROWS=3120" in common
    assert "RUBRIC_ONLINE_DATASET_VAL_ROWS=304" in common
    assert "RUBRIC_EVOLVE_CATEGORY_AWARE_SPLIT=1" in common
    assert "RUBRIC_EVOLVE_REFUSAL_POLARITY_FIX=1" in common
    settings = set()
    for suffix in ("fixed", "balanced", "penalty", "combined"):
        matches = list(scripts.glob(f"train_rubricevolve_e3f*_{suffix}.sh"))
        assert len(matches) == 1
        content = matches[0].read_text()
        assert "train_rubricevolve_e3_refusal_common.sh" in content
        balance = "RUBRIC_EVOLVE_REPLAY_MODE_BALANCE=1" in content
        penalty = "RUBRIC_EVOLVE_ANTI_REFUSAL_PENALTY_HIGH=1" in content
        settings.add((balance, penalty))
    assert settings == {(False, False), (True, False), (False, True), (True, True)}


def test_promotion_cap_keeps_strongest_held_out_rule(monkeypatch):
    monkeypatch.setenv("RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP", "1")
    candidate = evolution.RubricRules(
        ["moderate safety", "strong safety"],
        ["helpful"],
        [],
    )
    promoted = {
        "outcome_safety_rule": ["moderate safety", "strong safety"],
        "outcome_useful_rule": ["helpful"],
        "reasoning_process_rule": [],
        "refusal_rule": [],
    }
    evidence = {
        "outcome_safety_rule": [
            {"rule_index": 0.0, "gap": 0.6, "good_pass_rate": 1.0},
            {"rule_index": 1.0, "gap": 1.0, "good_pass_rate": 1.0},
        ],
        "outcome_useful_rule": [
            {"rule_index": 0.0, "gap": 0.8, "good_pass_rate": 1.0},
        ],
    }
    bounded = evolution._limit_promoted_rules(candidate, promoted, evidence)
    assert bounded["outcome_safety_rule"] == ["strong safety"]
    assert sum(len(values) for values in bounded.values()) == 1


def test_variance_weight_uses_complete_group_and_updates_next_batch(monkeypatch, tmp_path):
    @dataclass
    class Rules:
        outcome_safety_rule: list
        outcome_useful_rule: list
        reasoning_process_rule: list
        refusal_rule: list

        @property
        def all_rules(self):
            return {
                "outcome_safety_rule": self.outcome_safety_rule,
                "outcome_useful_rule": self.outcome_useful_rule,
                "reasoning_process_rule": self.reasoning_process_rule,
                "refusal_rule": self.refusal_rule,
            }

    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_START_STEP", "100")
    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "100")
    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS", "8")
    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_EMA_DECAY", "0")
    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_ALPHA", "0.4")
    monkeypatch.setenv("RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA", "0.3")
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path))
    evolution._VARIANCE_WEIGHT_STATE = {"schema_version": 1, "profiles": {}}
    monkeypatch.setattr(evolution, "_save_variance_weight_state", lambda state: None)

    rules = Rules(["critical safety"], [], [], [])
    infos = [
        {"uid": "same-question", "critical_safety_rule_count": 1}
        for _ in range(8)
    ]
    results = [
        {
            "rubric_rule_passes": {"outcome_safety_rule": [0]},
            "judge_parse_valid": True,
            "judge_failed": False,
        }
        for _ in range(8)
    ]

    # The current group has already been scored. This call only updates state
    # that _apply_variance_rule_weights reads for a later batch.
    evolution._update_variance_rule_weights(
        scopes=["ordinary_harm"] * 8,
        extra_infos=infos,
        rules_list=[rules] * 8,
        results=results,
    )
    weights = evolution._apply_variance_rule_weights(
        info=infos[0], scope="ordinary_harm", rules=rules
    )

    # Beta smoothing gives p=0.1 and variance=0.36. The harmful-safety
    # failure term raises rather than suppresses this all-fail critical rule.
    assert weights["outcome_safety_rule"][0] == pytest.approx(1.214)


def test_online_wrapper_preserves_base_refusal_reward_and_delays_new_rules(monkeypatch, tmp_path):
    @dataclass
    class BaseRules:
        outcome_safety_rule: list
        outcome_useful_rule: list
        reasoning_process_rule: list
        refusal_rule: list

        @property
        def all_rules(self):
            return {
                "outcome_safety_rule": self.outcome_safety_rule,
                "outcome_useful_rule": self.outcome_useful_rule,
                "reasoning_process_rule": self.reasoning_process_rule,
                "refusal_rule": self.refusal_rule,
            }

    class StubJudge:
        def judge_batch(self, prompts, responses, rules, extra_infos):
            return [
                {
                    "outcome_safety_rule": [1 if index < 2 else 0],
                    "outcome_useful_rule": [],
                    "reasoning_process_rule": [],
                    "refusal_rule": [],
                    "_parse_valid": True,
                }
                for index in range(len(prompts))
            ]

    class StubBase:
        RubricRules = BaseRules

        def __init__(self):
            self.seen_extra_infos = []
            self.judge = StubJudge()

        @staticmethod
        def _build_rules(info):
            return BaseRules(
                list(info.get("outcome_safety_rule", [])),
                list(info.get("outcome_useful_rule", [])),
                list(info.get("reasoning_process_rule", [])),
                list(info.get("refusal_rule", [])),
            )

        def _get_judge_backend(self):
            return self.judge

        def compute_score_batch(self, **kwargs):
            assert kwargs.pop("_skip_evolve_dispatch") is True
            self.seen_extra_infos.append(kwargs["extra_infos"])
            return [
                {
                    "score": 0.75,
                    "rubric_rule_passes": {
                        "outcome_safety_rule": [1] * len(info.get("outcome_safety_rule", [])),
                        "outcome_useful_rule": [],
                        "reasoning_process_rule": [],
                        "refusal_rule": [1] * len(info.get("refusal_rule", [])),
                    },
                    "rubric_rule_totals": {
                        "outcome_safety_rule": len(info.get("outcome_safety_rule", [])),
                        "refusal_rule": len(info.get("refusal_rule", [])),
                    },
                    "judge_parse_valid": True,
                    "judge_failed": False,
                }
                for info in kwargs["extra_infos"]
            ]

    base = StubBase()
    cache = {}
    monkeypatch.setattr(evolution, "_get_base_guard_impl", lambda: base)
    monkeypatch.setattr(evolution, "_get_evolved_cache", lambda epoch: cache)
    monkeypatch.setattr(evolution, "_save_evolved_rules", lambda epoch, value: None)
    monkeypatch.setattr(
        evolution,
        "_generate_evolved_rules",
        lambda **kwargs: {
            "outcome_safety_rule": ["new validated safety rule"],
            "outcome_useful_rule": [],
            "reasoning_process_rule": [],
            "trajectory_quality": [1, 1, 0, 0],
            "provider": "test",
            "model": "test",
            "generation_error": "",
        },
    )
    monkeypatch.setattr(evolution, "_append_evolve_audit", lambda record: None)
    monkeypatch.setenv("RUBRIC_EVOLVE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_START_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EPOCH", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_START_STEP", "100")
    monkeypatch.setenv("RUBRIC_EVOLVE_INTERVAL_STEPS", "20")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_LABEL_COUNT", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_MIN_GAP", "0.5")
    monkeypatch.setenv("RUBRIC_EVOLVE_SAVE_DIR", str(tmp_path))
    monkeypatch.delenv("RUBRIC_EVOLVE_STEPS", raising=False)
    evolution._EVOLVE_DIRTY = 0

    infos = [
        {
            "prompt": f"request {index}",
            "outcome_safety_rule": ["base safety"],
            "refusal_rule": ["base refusal"],
        }
        for index in range(4)
    ]
    sources = ["harm_sampled"] * 4
    responses = [f"response {index}" for index in range(4)]

    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "100")
    first = evolution.compute_score_batch(sources, responses, [None] * 4, infos)
    assert first[0]["score"] == pytest.approx(0.75)
    assert base.seen_extra_infos[0][0]["refusal_rule"] == ["base refusal"]
    assert base.seen_extra_infos[0][0]["outcome_safety_rule"] == ["base safety"]
    assert cache["ordinary_harm"]["outcome_safety_rule"] == ["new validated safety rule"]

    monkeypatch.setenv("RUBRIC_GLOBAL_STEP", "101")
    evolution.compute_score_batch(sources, responses, [None] * 4, infos)
    assert base.seen_extra_infos[1][0]["refusal_rule"] == ["base refusal"]
    assert base.seen_extra_infos[1][0]["outcome_safety_rule"] == [
        "base safety",
        "new validated safety rule",
    ]


def test_evolution_prompt_requests_quality_labels_and_uses_real_newlines():
    messages = evolution._build_evolve_prompt(
        prompt="A representative request",
        instruction_prompt="Follow the policy",
        dataset_label="ordinary_harm",
        existing_rules=evolution.RubricRules(["existing"], [], []),
        responses=["safe response", "unsafe response"],
    )
    content = messages[1]["content"]
    assert "trajectory_quality" in content
    assert "direct actionable harmful instructions" in content
    assert "refusal_rule must be an empty list" in content
    assert (chr(92) + "n") not in content


def test_trajectory_truncation_preserves_beginning_and_final_answer(monkeypatch):
    monkeypatch.setattr(evolution, "_get_evolve_tokenizer", lambda: None)
    trajectory = "BEGIN-" + ("x" * 200) + "-FINAL-ANSWER"
    bounded = evolution._truncate_evolution_trajectory(trajectory, max_tokens=20)
    assert bounded.startswith("BEGIN-")
    assert bounded.endswith("-FINAL-ANSWER")
    assert "middle omitted" in bounded


def test_two_stage_generation_freezes_labels_and_enforces_rule_budget(monkeypatch):
    calls = []

    def fake_call(messages, max_tokens):
        calls.append(messages)
        if len(calls) == 1:
            return (
                '{"trajectory_quality":[1,1,0,0],'
                '"failure_modes":["safe_helpful","safe_helpful",'
                '"harmful_enablement","partial_compliance"]}',
                "quota",
                "qwen3.7-plus",
            )
        return (
            '{"outcome_safety_rule":["s1","s2"],'
            '"outcome_useful_rule":["u1","u2"],'
            '"reasoning_process_rule":["r1","r2"],'
            '"refusal_rule":["must not survive"]}',
            "quota",
            "qwen3.7-plus",
        )

    monkeypatch.setattr(evolution, "_call_evolve_model", fake_call)
    monkeypatch.setenv("RUBRIC_EVOLVE_TWO_STAGE", "1")
    monkeypatch.setenv("RUBRIC_EVOLVE_MAX_RULES", "2")
    monkeypatch.setenv("RUBRIC_EVOLVE_TOTAL_MAX_RULES", "4")
    generated = evolution._generate_evolved_rules(
        prompt="harmful request",
        instruction_prompt="instruction",
        dataset_label="ordinary_harm",
        existing_rules=evolution.RubricRules([], [], [], []),
        responses=["good 1", "good 2", "bad 1", "bad 2"],
    )

    assert len(calls) == 2
    assert generated["trajectory_quality"] == [1, 1, 0, 0]
    assert generated["outcome_safety_rule"] == ["s1", "s2"]
    assert generated["outcome_useful_rule"] == ["u1", "u2"]
    assert generated["reasoning_process_rule"] == []
    assert generated["refusal_rule"] == []
    assert sum(len(generated[key]) for key in evolution._normalize_rules_dict({})) == 4


def test_replay_generation_reuses_fixed_proposal_labels_without_relabeling(monkeypatch):
    calls = []

    def fake_call(messages, max_tokens):
        calls.append(messages)
        return (
            '{"outcome_safety_rule":["separate harmful enablement"],'
            '"outcome_useful_rule":[],"reasoning_process_rule":[],"refusal_rule":[]}',
            "quota",
            "qwen3.7-plus",
        )

    monkeypatch.setattr(evolution, "_call_evolve_model", fake_call)
    monkeypatch.setenv("RUBRIC_EVOLVE_TWO_STAGE", "1")
    fixed = {
        "trajectory_quality": [1, 0],
        "failure_modes": ["safe_helpful", "harmful_enablement"],
    }
    generated = evolution._generate_evolved_rules(
        prompt="representative request",
        instruction_prompt="instruction",
        dataset_label="ordinary_harm",
        existing_rules=evolution.RubricRules([], [], [], []),
        responses=["good", "bad"],
        fixed_diagnosis=fixed,
    )

    # The replay pool has already been labeled independently. Only the rule
    # writer call is made, and it cannot silently relabel proposal examples.
    assert len(calls) == 1
    assert generated["trajectory_quality"] == [1, 0]
    assert generated["failure_modes"] == ["safe_helpful", "harmful_enablement"]
