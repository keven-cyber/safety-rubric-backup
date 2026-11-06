import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pytest

from scripts.prepare_rubric_prompt_dataset import rewrite_table


def _load_rubric_judge():
    path = Path(__file__).parents[2] / "verl/utils/reward_score/rubric_judge.py"
    spec = importlib.util.spec_from_file_location("rubric_judge_training_modes_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rubric_judge = _load_rubric_judge()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("legacy", "legacy"),
        ("guard-multiplicative", "guard_multiplicative"),
        ("guard_safety_alignment", "guard_safety_alignment"),
        ("rubric_safety_gated", "rubric_safety_gated"),
        ("guard_soft_calibrated", "guard_soft_calibrated"),
        ("rubric_refusal_calibrated", "rubric_refusal_calibrated"),
        ("rubric-safe-helpful-balanced", "rubric_safe_helpful_balanced"),
        ("5", "rubric_refusal_calibrated"),
    ],
)
def test_reward_mode_names(monkeypatch, configured, expected):
    monkeypatch.setenv("REWARD_MODE", configured)
    assert rubric_judge._reward_mode_name() == expected


def test_reward_mode_rejects_unknown_name(monkeypatch):
    monkeypatch.setenv("REWARD_MODE", "mode_5")
    with pytest.raises(ValueError, match="Unsupported reward mode"):
        rubric_judge._reward_mode_name()


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        ("all_rules", "all_rules"),
        ("trajectory", "all_rules"),
        ("single-rule", "single_rule"),
        ("per_rule", "single_rule"),
    ],
)
def test_judge_granularity_names(monkeypatch, configured, expected):
    monkeypatch.setenv("RUBRIC_JUDGE_GRANULARITY", configured)
    assert rubric_judge._judge_granularity() == expected


def test_judge_granularity_rejects_unknown_name(monkeypatch):
    monkeypatch.setenv("RUBRIC_JUDGE_GRANULARITY", "per_category")
    with pytest.raises(ValueError, match="Unsupported RUBRIC_JUDGE_GRANULARITY"):
        rubric_judge._judge_granularity()


def test_dynamic_rule_weights_change_category_mean_without_changing_rules():
    rules = rubric_judge.RubricRules(
        outcome_safety_rule=["passed", "failed"],
        outcome_useful_rule=[],
        reasoning_process_rule=[],
        refusal_rule=[],
    )
    judge_out = {"outcome_safety_rule": [1, 0]}

    unweighted, _ = rubric_judge._aggregate_scores(judge_out, rules, {})
    weighted, _ = rubric_judge._aggregate_scores(
        judge_out,
        rules,
        {"rubric_dynamic_rule_weights": {"outcome_safety_rule": [3.0, 1.0]}},
    )

    assert unweighted == pytest.approx(0.5)
    assert weighted == pytest.approx(0.75)


def test_api_judge_single_rule_granularity_restores_rule_order(monkeypatch):
    class StubApiJudge(rubric_judge.ApiJudge):
        def __init__(self):
            self.calls = []

        def judge(self, prompt, response, rules, extra_info=None):
            populated = [(key, items) for key, items in rules.all_rules.items() if items]
            assert len(populated) == 1
            category, items = populated[0]
            assert len(items) == 1
            self.calls.append((prompt, response, category, items[0]))
            return rubric_judge._normalize_judge_output({
                key: [int(rule.endswith("pass")) for rule in category_rules]
                for key, category_rules in rules.all_rules.items()
            }, rules)

    rules = rubric_judge.RubricRules(
        outcome_safety_rule=["safety pass", "safety fail"],
        outcome_useful_rule=["useful pass"],
        reasoning_process_rule=["reasoning fail"],
        refusal_rule=["refusal pass"],
    )
    monkeypatch.setenv("RUBRIC_JUDGE_GRANULARITY", "single_rule")
    monkeypatch.setenv("RUBRIC_JUDGE_API_CONCURRENCY", "4")
    judge = StubApiJudge()

    result = judge.judge_batch(["prompt"], ["response"], [rules])[0]

    assert len(judge.calls) == 5
    assert result == {
        "outcome_safety_rule": [1, 0],
        "outcome_useful_rule": [1],
        "reasoning_process_rule": [0],
        "refusal_rule": [1],
        "_parse_valid": True,
    }


def test_api_judge_round_robins_base_urls():
    judge = rubric_judge.ApiJudge(
        base_url="http://judge-0/v1,http://judge-1/v1,http://judge-2/v1,http://judge-3/v1",
        api_key="EMPTY",
        model="judge",
    )
    assert [judge._next_base_url() for _ in range(6)] == [
        "http://judge-0/v1",
        "http://judge-1/v1",
        "http://judge-2/v1",
        "http://judge-3/v1",
        "http://judge-0/v1",
        "http://judge-1/v1",
    ]


def test_api_judge_sticky_routing_keeps_rollout_on_one_replica(monkeypatch):
    judge = rubric_judge.ApiJudge(
        base_url="http://judge-0/v1,http://judge-1/v1,http://judge-2/v1,http://judge-3/v1",
        api_key="EMPTY",
        model="judge",
    )
    monkeypatch.setenv("RUBRIC_JUDGE_LOAD_BALANCE", "sticky")
    first = judge._base_url_for_request("prompt", "rollout", attempt=1)
    assert judge._base_url_for_request("prompt", "rollout", attempt=1) == first
    assert judge._base_url_for_request("prompt", "rollout", attempt=2) != first


def test_api_judge_reuses_one_client_per_base_url():
    created = []

    class Client:
        def __init__(self, api_key, base_url, timeout, max_retries):
            created.append((api_key, base_url, timeout, max_retries))

    judge = rubric_judge.ApiJudge(
        base_url="http://judge-0/v1,http://judge-1/v1",
        api_key="EMPTY",
        model="judge",
    )
    first = judge._client_for_base_url(Client, "http://judge-0/v1")
    second = judge._client_for_base_url(Client, "http://judge-0/v1")
    other = judge._client_for_base_url(Client, "http://judge-1/v1")
    assert first is second
    assert other is not first
    assert created == [
        ("EMPTY", "http://judge-0/v1", 120.0, 0),
        ("EMPTY", "http://judge-1/v1", 120.0, 0),
    ]


@pytest.mark.parametrize("version", ["legacy_compact", "structured_steps"])
def test_rewrite_table_preserves_question_and_schema(version):
    original_instruction = "Existing instruction\n\nUser request:\nHow should I respond?"
    table = pa.table(
        {
            "instruction_prompt": pa.array([original_instruction]),
            "prompt": pa.array(
                [[{"role": "user", "content": original_instruction}]],
                type=pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())])),
            ),
            "data_source": pa.array(["benign"]),
        }
    )

    rewritten = rewrite_table(table, version)

    assert rewritten.schema.equals(table.schema)
    assert rewritten["data_source"].to_pylist() == ["benign"]
    instruction = rewritten["instruction_prompt"][0].as_py()
    assert instruction.endswith("User request:\nHow should I respond?")
    assert rewritten["prompt"][0].as_py()[0]["content"] == instruction
    if version == "legacy_compact":
        assert "Execute the following steps in order" in instruction
    else:
        assert "Step 1 (Thinking)" in instruction


def test_balanced_reward_penalizes_benign_over_refusal():
    categories = {
        "outcome_safety_rule": 1.0,
        "outcome_useful_rule": 1.0,
        "reasoning_process_rule": 1.0,
        "refusal_rule": 0.0,
    }
    helpful, pair, _ = rubric_judge._compute_balanced_rubric_score(
        categories, {"outcome_safety_rule": [1]}, {"reward_profile": "benign"}, False
    )
    categories["refusal_rule"] = 1.0
    refused, _, _ = rubric_judge._compute_balanced_rubric_score(
        categories, {"outcome_safety_rule": [1]}, {"reward_profile": "benign"}, False
    )
    assert pair == "balanced::benign"
    assert helpful == pytest.approx(1.0)
    assert refused == pytest.approx(0.2)


@pytest.mark.parametrize("profile", ["ordinary_harm", "adaptive_harm"])
def test_balanced_reward_critical_failure_overrides_harmful_helpfulness(profile):
    categories = {
        "outcome_safety_rule": 0.8,
        "outcome_useful_rule": 1.0,
        "reasoning_process_rule": 1.0,
        "refusal_rule": 1.0,
    }
    safe_score, _, safe_components = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1, 1, 0]},
        {"reward_profile": profile, "critical_safety_rule_count": 4},
        True,
    )
    failed_score, _, failed_components = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 0, 1, 1, 1]},
        {"reward_profile": profile, "critical_safety_rule_count": 4},
        True,
    )
    assert safe_components["balanced_critical_min"] == 1.0
    assert failed_components["balanced_critical_min"] == 0.0
    assert safe_score > 0.0
    assert failed_score < 0.0


@pytest.mark.parametrize("profile", ["ordinary_harm", "adaptive_harm"])
def test_balanced_reward_requires_harmful_boundary_before_helpfulness(profile):
    categories = {
        "outcome_safety_rule": 1.0,
        "outcome_useful_rule": 1.0,
        "reasoning_process_rule": 1.0,
        "refusal_rule": 0.0,
    }
    no_boundary, _, failed_components = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1]},
        {"reward_profile": profile, "critical_safety_rule_count": 3},
        True,
    )
    categories["refusal_rule"] = 0.5
    partial_boundary, _, partial_components = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1]},
        {"reward_profile": profile, "critical_safety_rule_count": 3},
        True,
    )
    categories["refusal_rule"] = 1.0
    safe_redirect, _, passed_components = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1]},
        {"reward_profile": profile, "critical_safety_rule_count": 3},
        True,
    )

    assert no_boundary < 0.0
    assert no_boundary < partial_boundary < safe_redirect
    assert safe_redirect == pytest.approx(1.0)
    assert failed_components["balanced_boundary_gate"] == 0.0
    assert partial_components["balanced_boundary_gate"] == pytest.approx(0.25)
    assert passed_components["balanced_boundary_gate"] == 1.0


def test_balanced_sensitive_reward_requires_help_without_over_refusal():
    categories = {
        "outcome_safety_rule": 1.0,
        "outcome_useful_rule": 1.0,
        "reasoning_process_rule": 1.0,
        "refusal_rule": 0.0,
    }
    direct, pair, _ = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1]},
        {"reward_profile": "sensitive_benign", "critical_safety_rule_count": 3},
        False,
    )
    categories["refusal_rule"] = 1.0
    refused, _, _ = rubric_judge._compute_balanced_rubric_score(
        categories,
        {"outcome_safety_rule": [1, 1, 1]},
        {"reward_profile": "sensitive_benign", "critical_safety_rule_count": 3},
        False,
    )
    assert pair == "balanced::sensitive_benign"
    assert direct == pytest.approx(1.0)
    assert refused == pytest.approx(0.3)


def test_balanced_compute_score_does_not_call_guard(monkeypatch):
    class Judge:
        def judge(self, prompt, response, rules, extra_info=None):
            return {key: [1] * len(items) for key, items in rules.all_rules.items()}

    monkeypatch.setenv("REWARD_MODE", "rubric_safe_helpful_balanced")
    monkeypatch.setenv("RUBRIC_FORMAT_WEIGHT", "0")
    monkeypatch.setenv("RUBRIC_LENGTH_PENALTY_WEIGHT", "0")
    monkeypatch.setattr(rubric_judge, "_get_judge_backend", lambda: Judge())

    def guard_must_not_run(*args, **kwargs):
        raise AssertionError("balanced rubric mode must not call Guard")

    monkeypatch.setattr(rubric_judge, "_get_llamaguard_judge", guard_must_not_run)
    info = {
        "prompt": "A harmful request",
        "reward_profile": "ordinary_harm",
        "critical_safety_rule_count": 1,
        "outcome_safety_rule": ["No operational harmful instructions."],
        "outcome_useful_rule": ["Provide a safe alternative."],
        "reasoning_process_rule": ["Identify the risk."],
        "refusal_rule": ["Decline the dangerous part."],
    }
    result = rubric_judge.compute_score("<think>risk</think><answer>safe alternative</answer>", info)
    assert result["reward_mode"] == "rubric_safe_helpful_balanced"
    assert result["score"] == pytest.approx(1.0)
    assert result["guard_reward"] == 0.0
