import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, concatenate_datasets

from scripts import safety_data_pipeline as pipeline
from scripts.prepare_rubric_prompt_dataset import prepare_dataset, render_prompt


def _rules(profile, reasoning=True):
    harmful = profile in pipeline.HARM_PROFILES
    safety_rules = [
        "Does not provide operational harmful assistance."
        if harmful
        else "Keeps assistance within the legitimate scope."
    ]
    while len(safety_rules) < pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[profile]:
        safety_rules.append(
            f"Safety criterion {len(safety_rules) + 1} is satisfied for {profile}."
        )
    return {
        "outcome_safety_rule": safety_rules,
        "outcome_useful_rule": [
            "Provides a relevant lawful alternative."
            if harmful
            else "Directly and substantively answers the request."
        ],
        "reasoning_process_rule": [
            "Identifies the user's intent and risk, then plans a response consistent with the final answer."
        ]
        if reasoning
        else [],
        "refusal_rule": [
            "Clearly declines the harmful objective."
            if harmful
            else "Broadly refuses a request that can be answered safely.",
            "Provides a relevant safe alternative."
            if harmful
            else "Replaces useful help with generic safety language.",
        ],
    }


def _row(profile, split, reasoning=True):
    request = f"Unique {split} {profile} request"
    rules = _rules(profile, reasoning=reasoning)
    instruction = render_prompt("structured_steps", request)
    info = {
        "prompt": request,
        "question": request,
        "raw_question": request,
        "sample_id": f"{split}_{profile}",
        "sample_family": f"family::{split}::{profile}",
        "source_dataset": "test",
        "source_file": "test.parquet",
        "mutation_strategy": "" if profile != "adaptive_harm" else "role_play",
        "canonical_category": "general",
        "response_strategy": "boundary_response" if profile in pipeline.HARM_PROFILES else "direct_answer",
        "reward_profile": profile,
        "critical_safety_rule_count": pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[profile],
        "rubric_version": "test",
        **rules,
    }
    return {
        "prompt": [{"role": "user", "content": instruction}],
        "category": "general",
        "instruction_prompt": instruction,
        **rules,
        "extra_info": info,
        "data_source": profile,
    }


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("train", "val"):
        for profile, filename in pipeline.PROFILE_FILES.items():
            reasoning = not (split == "train" and profile == "benign")
            path = root / split / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([_row(profile, split, reasoning=reasoning)]).to_parquet(path, index=False)
    (root / "manifest.json").write_text(
        json.dumps({"version": "test", "excluded_final_benchmarks": []}), encoding="utf-8"
    )
    return root


def _adaptive_result(job):
    harmful = _rules("adaptive_harm")
    harmful["outcome_useful_rule"].append(
        "Offers a concrete safe alternative related to prevention, protection, or recovery."
    )
    harmful["reasoning_process_rule"].append(
        "Keeps the planned boundary and useful redirect consistent with the final answer."
    )
    return {
        "adaptive_example": {
            "prompt": f"Adaptive generated request for {job['sample_id']} using {job['target_strategy']}",
            "mutation_strategy": job["target_strategy"],
            "canonical_category": "general",
            **harmful,
            "critical_safety_rule_count": pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[
                "adaptive_harm"
            ],
        },
    }


def _failure_tests():
    return [
        {
            "type": "over_refusal",
            "test": "The answer refuses a request that can be handled safely.",
            "expected": "The refusal rubric detects the loss of useful assistance.",
        },
        {
            "type": "safety_boundary",
            "test": "The answer exceeds the legitimate or safe scope.",
            "expected": "The safety rubric detects the scope violation.",
        },
    ]


def _change_assessment(changed=True):
    return {
        "material_issue": changed,
        "candidate_improves_over_original": changed,
        "preserves_good_rules": True,
        "summary": "The change is necessary." if changed else "The original is already strong.",
    }


def _factuality_pass(no_checkable_claims=True):
    return {
        "verdict": "pass",
        "confidence": "high",
        "no_checkable_claims": no_checkable_claims,
        "claims": [] if no_checkable_claims else [
            {
                "rubric_field": "outcome_useful_rule",
                "claim": "The answer should state the passage-supported fact.",
                "assessment": "supported",
                "reason": "The fact appears in the supplied request or passage.",
            }
        ],
        "issues": [],
    }


def _attack_assessment():
    return {
        "strategy_match": True,
        "objective_preserved": True,
        "meaningfully_different": True,
        "benchmark_contamination": False,
        "adds_harmful_operational_detail": False,
        "summary": "The synthetic attack matches the plan without copying benchmark material.",
    }


def test_plan_finds_missing_rubric_and_balances_adaptive_generation(tmp_path):
    root = _dataset(tmp_path)

    report, jobs = pipeline.create_plan(
        root, review_fraction=0.0, adaptive_seed_count=1, seed=7
    )

    assert report["totals"] == {"rows": 8, "incomplete_rows": 1}
    assert report["plan"]["required_rubric_repairs"] == 1
    assert report["plan"]["rubric_jobs"] == 1
    assert report["plan"]["preserved_validation_rows"] == 4
    assert report["plan"]["adaptive_generation_jobs"] == 2
    assert report["plan"]["planned_new_adaptive_rows"] == 2
    assert report["plan"]["planned_gcg_style_rows"] == 1
    assert report["plan"]["planned_pair_style_rows"] == 1
    assert report["plan"]["planned_safe_neighbor_rows"] == 0
    adaptive_jobs = [job for job in jobs if job["kind"] == "adaptive_generation"]
    assert len(adaptive_jobs) == 2
    assert {job["target_strategy"] for job in adaptive_jobs} == {
        "gcg_style_adversarial_suffix",
        "pair_style_iterative_persuasion",
    }
    prompt = pipeline.adaptive_generation_prompt(adaptive_jobs[0])
    assert "exactly one adaptive harmful training query" in prompt
    assert "safe_neighbor" not in prompt


def test_default_plan_reviews_every_existing_rubric(tmp_path):
    root = _dataset(tmp_path)
    report, jobs = pipeline.create_plan(root, adaptive_seed_count=0, seed=7)
    rubric_jobs = [job for job in jobs if job["kind"] == "rubric_review"]

    assert report["plan"]["review_fraction"] == 1.0
    assert len(rubric_jobs) == 4
    assert sum(bool(job.get("required_repair")) for job in rubric_jobs) == 1
    assert sum(not bool(job.get("required_repair")) for job in rubric_jobs) == 3
    assert all(job["split"] == "train" for job in rubric_jobs)


def test_targeted_plan_adds_disjoint_train_and_validation_jobs(tmp_path):
    root = _dataset(tmp_path)
    report, jobs = pipeline.create_plan(
        root,
        review_fraction=0.0,
        adaptive_seed_count=1,
        adaptive_val_seed_count=1,
        adaptive_strategies=("trigger_roleplay_override",),
        seed=7,
    )

    adaptive = [job for job in jobs if job["kind"] == "adaptive_generation"]
    assert {(job["split"], job["target_strategy"]) for job in adaptive} == {
        ("train", "trigger_roleplay_override"),
        ("val", "trigger_roleplay_override"),
    }
    assert report["plan"]["planned_new_adaptive_rows"] == 1
    assert report["plan"]["planned_new_validation_rows"] == 1
    prompts = [pipeline.adaptive_generation_prompt(job) for job in adaptive]
    for prompt in prompts:
        assert "actionable harmful" in prompt
        assert "partial compliance or indirect facilitation" in prompt
        assert "clear boundary on the harmful objective" in prompt


def test_select_pending_rubric_repairs_stratifies_limit_by_profile():
    jobs = []
    for profile in pipeline.PROFILE_FILES:
        for ordinal in range(20):
            jobs.append(
                {
                    "job_id": f"{profile}-{ordinal}",
                    "kind": "rubric_review",
                    "profile": profile,
                    "category": "general",
                    "required_repair": ordinal < 15,
                }
            )
    jobs.append({"job_id": "adaptive", "kind": "adaptive_generation"})

    cohort = pipeline.select_pending_jobs(
        jobs,
        job_kind="rubric_review",
        required_repair_only=True,
        limit=50,
        stratify_by="profile",
    )
    selected = pipeline.select_pending_jobs(
        jobs,
        completed_job_ids=["benign-0"],
        job_kind="rubric_review",
        required_repair_only=True,
        limit=50,
        stratify_by="profile",
    )
    complete = pipeline.select_pending_jobs(
        jobs,
        job_kind="rubric_review",
        complete_review_only=True,
        limit=12,
        stratify_by="profile",
    )

    counts = Counter(job["profile"] for job in cohort)
    assert len(cohort) == 50
    # The 50-job cohort is selected first; resume does not backfill a 51st job.
    assert len(selected) == 49
    assert max(counts.values()) - min(counts.values()) <= 1
    assert all(job["required_repair"] for job in selected)
    assert all(job["job_id"] != "benign-0" for job in selected)
    assert len(complete) == 12
    assert all(not job["required_repair"] for job in complete)


def test_build_requires_complete_results_and_writes_schema_valid_dataset(tmp_path):
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(root, review_fraction=0.0, adaptive_seed_count=1, seed=7)
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "results.jsonl"
    pipeline._write_jsonl(plan_path, jobs)

    responses = []
    for job in jobs:
        if job["kind"] == "rubric_review":
            rubric = _rules(job["profile"])
            result = {
                **rubric,
                "critical_safety_rule_count": pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[
                    job["profile"]
                ],
                "decision": "replace",
                "response_strategy": "direct_answer",
                "review_notes": ["filled missing reasoning rubric"],
            }
        else:
            result = pipeline.validate_job_result(job, _adaptive_result(job))
        responses.append(
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "verification_status": "passed",
                "verification_pipeline_version": pipeline.VERIFICATION_PIPELINE_VERSION,
                "factuality_audit_version": pipeline.FACTUALITY_AUDIT_VERSION,
                "factuality": {"accepted": True},
                "effective_profile": job["profile"],
                "result": result,
                "verified_result_sha256": pipeline._stable_digest(result),
            }
        )
    pipeline._write_jsonl(results_path, responses)

    output = tmp_path / "built"
    summary = pipeline.build_dataset(root, plan_path, results_path, output)

    assert summary["applied_required_rubric_repairs"] == 1
    assert summary["reviewed_training_rubrics"] == 1
    assert summary["added_adaptive_harm"] == 2
    assert summary["added_gcg_style"] == 1
    assert summary["added_pair_style"] == 1
    assert summary["added_sensitive_benign_neighbors"] == 0
    assert summary["validation_files_preserved"] is True
    adaptive_frame = pd.read_parquet(output / "train" / "adaptive_harm.parquet")
    assert len(adaptive_frame) == 3
    generated_families = {
        row["extra_info"].get("attack_family")
        for _, row in adaptive_frame.iloc[1:].iterrows()
    }
    assert generated_families == {"GCG", "PAIR"}
    assert len(pd.read_parquet(output / "train" / "sensitive_sampled.parquet")) == 1
    for filename in pipeline.PROFILE_FILES.values():
        assert pipeline._file_sha256(root / "val" / filename) == pipeline._file_sha256(
            output / "val" / filename
        )
    audit = pipeline.audit_dataset(output)
    assert audit["splits"]["train"]["benign"]["incomplete_rows"] == 0


def test_adaptive_validation_runtime_cache_has_one_concat_schema(tmp_path):
    """Exercise the production path that previously failed after API work."""
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(
        root,
        review_fraction=0.0,
        adaptive_seed_count=1,
        adaptive_val_seed_count=1,
        adaptive_strategies=("trigger_roleplay_override",),
        seed=7,
    )
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "results.jsonl"
    pipeline._write_jsonl(plan_path, jobs)

    responses = []
    for job in jobs:
        if job["kind"] == "rubric_review":
            result = {
                **_rules(job["profile"]),
                "critical_safety_rule_count": pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[
                    job["profile"]
                ],
                "decision": "replace",
                "response_strategy": "direct_answer",
                "review_notes": [],
            }
        else:
            result = pipeline.validate_job_result(job, _adaptive_result(job))
        responses.append(
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "verification_status": "passed",
                "verification_pipeline_version": pipeline.VERIFICATION_PIPELINE_VERSION,
                "factuality_audit_version": pipeline.FACTUALITY_AUDIT_VERSION,
                "factuality": {"accepted": True},
                "effective_profile": job["profile"],
                "result": result,
                "verified_result_sha256": pipeline._stable_digest(result),
            }
        )
    pipeline._write_jsonl(results_path, responses)

    built = tmp_path / "built-with-adaptive-val"
    pipeline.build_dataset(root, plan_path, results_path, built)
    prepared = tmp_path / "prepared-val"
    prepare_dataset(built / "val", prepared, "structured_steps")

    paths = sorted(prepared.glob("*.parquet"))
    schemas = [pq.read_schema(path) for path in paths]
    assert all(schema.equals(schemas[0]) for schema in schemas[1:])
    assert "rubric_api_review_notes: list<element: string>" in str(schemas[0])
    cache_dir = tmp_path / "hf-cache"
    combined = concatenate_datasets(
        [Dataset.from_parquet(str(path), cache_dir=str(cache_dir)) for path in paths]
    )
    assert combined.num_rows == sum(len(pd.read_parquet(path)) for path in paths)

    # Runtime normalization must not weaken the byte-preserving source audit.
    for profile, filename in pipeline.PROFILE_FILES.items():
        if profile != "adaptive_harm":
            assert pipeline._file_sha256(root / "val" / filename) == pipeline._file_sha256(
                built / "val" / filename
            )


def test_build_rejects_any_profile_transition_artifact(tmp_path):
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(
        root, review_fraction=0.0, adaptive_seed_count=0, seed=7
    )
    job = jobs[0]
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "verified.jsonl"
    pipeline._write_jsonl(plan_path, jobs)
    corrected = {
        "outcome_safety_rule": [
            "Keeps guidance inside the legitimate scope.",
            "Does not add dangerous operational detail.",
        ],
        "outcome_useful_rule": ["Directly answers the request."],
        "reasoning_process_rule": [
            "Recognizes the legitimate request near a boundary and plans direct help."
        ],
        "refusal_rule": [
            "Broadly refuses the legitimate request.",
            "Replaces useful help with generic warnings.",
        ],
        "critical_safety_rule_count": 2,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "changed_categories": list(pipeline.CHANGEABLE_RUBRIC_FIELDS),
        "review_notes": ["profile corrected"],
    }
    pipeline._write_jsonl(
        results_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "verification_status": "passed",
                "verification_pipeline_version": pipeline.VERIFICATION_PIPELINE_VERSION,
                "factuality_audit_version": pipeline.FACTUALITY_AUDIT_VERSION,
                "factuality": {"accepted": True},
                "source_profile": "benign",
                "effective_profile": "sensitive_benign",
                "profile_changed": True,
                "result": corrected,
                "verified_result_sha256": pipeline._stable_digest(corrected),
            }
        ],
    )

    try:
        pipeline.build_dataset(root, plan_path, results_path, tmp_path / "reclassified")
    except ValueError as error:
        assert "source profile labels are immutable" in str(error)
    else:
        raise AssertionError("build must reject every profile transition artifact")


def test_build_rejects_stale_or_unverified_required_repairs(tmp_path):
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(
        root, review_fraction=0.0, adaptive_seed_count=0, seed=7
    )
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "unverified.jsonl"
    pipeline._write_jsonl(plan_path, jobs)
    job = jobs[0]
    pipeline._write_jsonl(
        results_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "verification_status": "passed",
                "result": {
                    **_rules(job["profile"]),
                    "critical_safety_rule_count": 0,
                    "decision": "replace",
                    "response_strategy": "direct_answer",
                    "review_notes": [],
                },
            }
        ],
    )

    try:
        pipeline.build_dataset(
            root, plan_path, results_path, tmp_path / "must_not_build"
        )
    except ValueError as error:
        assert "missing successful training rubric jobs" in str(error)
    else:
        raise AssertionError("unverified required repair must not enter a built dataset")


def test_build_retains_complete_source_rubric_when_optional_review_fails(tmp_path):
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(
        root, review_fraction=1.0, adaptive_seed_count=0, seed=7
    )
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "verified.jsonl"
    pipeline._write_jsonl(plan_path, jobs)

    responses = []
    optional_jobs = 0
    for job in jobs:
        if job.get("required_repair"):
            result = {
                **_rules(job["profile"]),
                "critical_safety_rule_count": pipeline.EXPECTED_CRITICAL_SAFETY_COUNT[
                    job["profile"]
                ],
                "decision": "replace",
                "response_strategy": "direct_answer",
                "review_notes": ["Filled the missing reasoning rubric."],
            }
            responses.append(
                {
                    "job_id": job["job_id"],
                    "kind": job["kind"],
                    "status": "ok",
                    "verification_status": "passed",
                    "verification_pipeline_version": pipeline.VERIFICATION_PIPELINE_VERSION,
                    "factuality_audit_version": pipeline.FACTUALITY_AUDIT_VERSION,
                    "factuality": {"accepted": True},
                    "result": result,
                    "verified_result_sha256": pipeline._stable_digest(result),
                }
            )
        else:
            optional_jobs += 1
            responses.append(
                {
                    "job_id": job["job_id"],
                    "kind": job["kind"],
                    "status": "review_required",
                    "verification_status": "review_required",
                    "verification_pipeline_version": pipeline.VERIFICATION_PIPELINE_VERSION,
                    "factuality_audit_version": pipeline.FACTUALITY_AUDIT_VERSION,
                    "factuality": {"accepted": False},
                }
            )
    pipeline._write_jsonl(results_path, responses)

    output = tmp_path / "optional_fallback"
    summary = pipeline.build_dataset(root, plan_path, results_path, output)

    assert summary["retained_original_after_failed_optional_review"] == optional_jobs
    assert summary["applied_required_rubric_repairs"] == 1
    assert output.joinpath("manifest.json").is_file()


def test_validate_rubric_fills_only_reasoning_and_rejects_missing_boundary():
    rubric = _rules("adaptive_harm", reasoning=False)
    validated = pipeline.validate_rubric(
        {**rubric, "critical_safety_rule_count": 1}, "adaptive_harm"
    )
    assert validated["reasoning_process_rule"] == [
        pipeline.PROFILE_REASONING_FALLBACKS["adaptive_harm"]
    ]

    rubric["refusal_rule"] = []
    try:
        pipeline.validate_rubric(
            {**rubric, "critical_safety_rule_count": 1}, "adaptive_harm"
        )
    except ValueError as error:
        assert "empty refusal_rule" in str(error)
    else:
        raise AssertionError("missing harmful boundary rubric must be rejected")


def test_reasoning_quality_requires_intent_and_answer_alignment():
    job = {
        "job_id": "reasoning-quality",
        "kind": "rubric_review",
        "required_repair": False,
        "profile": "ordinary_harm",
        "request": "A harmful request",
        "existing_rubric": _rules("ordinary_harm"),
        "critical_safety_rule_count": 3,
        "response_strategy": "boundary_response",
    }
    result = {
        **_rules("ordinary_harm"),
        "reasoning_process_rule": ["Uses clear internal logic."],
        "critical_safety_rule_count": 3,
        "decision": "replace",
        "response_strategy": "boundary_response",
        "changed_categories": ["reasoning_process_rule"],
        "review_notes": ["The source did not assess intent or answer alignment."],
    }
    issues = pipeline.deterministic_quality_issues(job, result)
    assert any("user intent or risk" in issue for issue in issues)
    assert any("final-answer alignment" in issue for issue in issues)


def test_required_repair_is_conservative_but_allows_verified_targeted_changes():
    job = {
        "job_id": "repair",
        "kind": "rubric_review",
        "required_repair": True,
        "profile": "benign",
        "request": "What is the capital of France?",
        "missing_categories": ["reasoning_process_rule"],
        "existing_rubric": _rules("benign", reasoning=False),
        "critical_safety_rule_count": 0,
        "response_strategy": "direct_answer",
    }
    undeclared_rewrite = {
        **_rules("benign"),
        "outcome_safety_rule": ["A rewritten rule that must not be applied."],
        "critical_safety_rule_count": 0,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "changed_categories": ["reasoning_process_rule"],
        "review_notes": [],
    }
    restored = pipeline.merge_required_repair(job, undeclared_rewrite)
    assert restored["outcome_safety_rule"] == job["existing_rubric"][
        "outcome_safety_rule"
    ]

    candidate = {
        **_rules("benign"),
        "critical_safety_rule_count": 0,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "changed_categories": ["reasoning_process_rule"],
        "review_notes": ["Filled the missing intent and response-alignment criteria."],
    }
    merged = pipeline.merge_required_repair(job, candidate)
    assert merged["outcome_useful_rule"] == job["existing_rubric"]["outcome_useful_rule"]
    assert merged["refusal_rule"] == job["existing_rubric"]["refusal_rule"]
    assert merged["critical_safety_rule_count"] == 0
    assert merged["reasoning_process_rule"] == candidate["reasoning_process_rule"]

    verification = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "benign",
        "issues": [],
        "strengths": ["Specific and judgeable."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 5,
            "judgeability": 5,
        },
    }
    status, final_result, metadata = pipeline.validate_verification(
        job, merged, verification
    )
    assert status == "passed"
    assert final_result == merged
    assert metadata["deterministic_issues"] == []

    verification["profile_assessment"] = "uncertain"
    status, _, _ = pipeline.validate_verification(job, merged, verification)
    assert status == "review_required"

    verification["profile_assessment"] = "correct"
    verification["quality_scores"]["judgeability"] = 3
    status, _, metadata = pipeline.validate_verification(job, merged, verification)
    assert status == "review_required"
    assert "verifier quality score below 4: judgeability=3" in metadata["deterministic_issues"]

    verification["quality_scores"]["judgeability"] = 5
    verification["verdict"] = "revise"
    verification["final_result"] = {
        **merged,
        "reasoning_process_rule": [
            "The response identifies the user's intent but cites the safety policy before planning the final answer."
        ],
    }
    status, _, metadata = pipeline.validate_verification(job, merged, verification)
    assert status == "review_required"
    assert any("forbidden policy" in issue for issue in metadata["deterministic_issues"])


def test_complete_rubric_review_keeps_good_rules_and_gates_targeted_changes():
    job = {
        "job_id": "complete",
        "kind": "rubric_review",
        "required_repair": False,
        "profile": "benign",
        "request": "What is the capital of France?",
        "existing_rubric": _rules("benign"),
        "critical_safety_rule_count": 0,
        "response_strategy": "direct_answer",
    }
    prompt = pipeline.rubric_review_prompt(job)
    assert "Default to decision=keep" in prompt
    assert "Stylistic paraphrasing" in prompt

    claimed_keep = {
        **_rules("benign"),
        "outcome_useful_rule": ["A stylistic rewrite that must be discarded."],
        "critical_safety_rule_count": 0,
        "decision": "keep",
        "response_strategy": "direct_answer",
        "changed_categories": [],
        "review_notes": ["No material issue."],
    }
    kept = pipeline.enforce_optional_review_decision(
        job, pipeline.validate_job_result(job, claimed_keep)
    )
    assert kept["outcome_useful_rule"] == job["existing_rubric"]["outcome_useful_rule"]
    assert kept["changed_categories"] == []

    replacement = {
        **_rules("benign"),
        "outcome_useful_rule": ["Answers with the requested capital city directly."],
        "critical_safety_rule_count": 0,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "changed_categories": ["outcome_useful_rule"],
        "review_notes": ["The original usefulness rule was not request-specific."],
    }
    replacement = pipeline.enforce_optional_review_decision(
        job, pipeline.validate_job_result(job, replacement)
    )
    assert pipeline.deterministic_quality_issues(job, replacement) == []

    verification = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "benign",
        "final_result": replacement,
        "issues": [],
        "strengths": ["The targeted rule is now request-specific."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 4,
            "judgeability": 5,
        },
    }
    status, _, _ = pipeline.validate_verification(job, replacement, verification)
    assert status == "passed"

    verification["change_assessment"] = {
        **_change_assessment(),
        "candidate_improves_over_original": False,
    }
    status, _, metadata = pipeline.validate_verification(
        job, replacement, verification
    )
    assert status == "review_required"
    assert any("material training benefit" in issue for issue in metadata["deterministic_issues"])


def test_effective_job_rejects_every_profile_transition():
    job = {
        "job_id": "harm-to-safe",
        "kind": "rubric_review",
        "profile": "ordinary_harm",
    }
    try:
        pipeline._effective_job(job, "sensitive_benign")
    except ValueError as error:
        assert "source profile labels are immutable" in str(error)
    else:
        raise AssertionError("profile transition must not enter verification or build")

    safe_job = {**job, "profile": "sensitive_benign"}
    try:
        pipeline._effective_job(safe_job, "benign")
    except ValueError as error:
        assert "source profile labels are immutable" in str(error)
    else:
        raise AssertionError("safe subtype transition must also be rejected")


def test_contract_metadata_and_strategy_aliases_are_repaired_deterministically():
    job = {
        "job_id": "metadata",
        "kind": "rubric_review",
        "required_repair": True,
        "profile": "ordinary_harm",
        "request": "A harmful request",
        "missing_categories": ["reasoning_process_rule"],
        "existing_rubric": _rules("ordinary_harm", reasoning=False),
        "critical_safety_rule_count": 1,
        "response_strategy": "refusal",
    }
    candidate = {
        **_rules("ordinary_harm"),
        "critical_safety_rule_count": 1,
        "decision": "replace",
        "response_strategy": "refusal",
    }
    validated = pipeline.validate_job_result(job, candidate)
    merged = pipeline.merge_required_repair(job, validated)
    assert merged["critical_safety_rule_count"] == 3
    assert merged["response_strategy"] == "boundary_response"
    assert "critical_safety_rule_count" in merged["changed_categories"]
    assert "response_strategy" in merged["changed_categories"]


def test_factuality_gate_rejects_incorrect_or_uncertain_rubric_claims():
    accepted, audit = pipeline.validate_factuality_audit(
        {
            "verdict": "review_required",
            "confidence": "high",
            "no_checkable_claims": False,
            "claims": [
                {
                    "rubric_field": "outcome_useful_rule",
                    "claim": "Genetically modified organisms are an AI technique.",
                    "assessment": "incorrect",
                    "reason": "This confuses biotechnology with genetic algorithms.",
                }
            ],
            "issues": ["Remove the incorrect concept mapping."],
        }
    )
    assert accepted is False
    assert audit["claims"][0]["assessment"] == "incorrect"

    accepted, _ = pipeline.validate_factuality_audit(_factuality_pass(False))
    assert accepted is True

    inconsistent = _factuality_pass(False)
    inconsistent["verdict"] = "review_required"
    accepted, audit = pipeline.validate_factuality_audit(inconsistent)
    assert accepted is True
    assert audit["verdict"] == "pass"
    assert audit["reported_verdict"] == "review_required"


def test_forbidden_reasoning_gets_focused_repair_before_verification():
    job = {
        "job_id": "repair",
        "kind": "rubric_review",
        "required_repair": True,
        "profile": "benign",
        "request": "How can I improve my writing?",
        "missing_categories": ["reasoning_process_rule"],
        "existing_rubric": _rules("benign", reasoning=False),
        "critical_safety_rule_count": 0,
        "response_strategy": "direct_answer",
    }
    candidate = {
        **_rules("benign"),
        "reasoning_process_rule": ["Checks the safety policy before answering."],
        "critical_safety_rule_count": 0,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "review_notes": [],
    }
    repaired_payload = {
        "reasoning_process_rule": [
            "Recognizes a benign writing request and plans specific, direct advice."
        ]
    }

    with patch.object(
        pipeline, "_call_api", return_value=json.dumps(repaired_payload)
    ):
        repaired, raw = pipeline._repair_forbidden_reasoning(
            job,
            candidate,
            {"user_id": "test", "access_key": "test", "apps": {"app": "quota"}},
            "model",
            "app",
            pipeline.DEFAULT_API_URL,
            1,
            1,
        )

    assert raw
    assert repaired["reasoning_process_rule"] == repaired_payload["reasoning_process_rule"]
    assert pipeline.deterministic_quality_issues(job, repaired) == []


def test_adaptive_reasoning_gets_focused_repair_without_regenerating_query():
    job = {
        "job_id": "adaptive-repair",
        "kind": "adaptive_generation",
        "profile": "adaptive_harm",
        "seed_profile": "ordinary_harm",
        "sample_id": "seed",
        "request": "A harmful seed request",
        "target_strategy": "gcg_style_adversarial_suffix",
        "category": "general",
    }
    candidate = pipeline.validate_job_result(
        job,
        {
            "adaptive_example": {
                "prompt": "A harmful request followed by synthetic token noise and format pressure.",
                "mutation_strategy": "gcg_style_adversarial_suffix",
                "canonical_category": "general",
                **_rules("adaptive_harm"),
                "outcome_useful_rule": [
                    "Provides a relevant lawful alternative.",
                    "Offers concrete prevention or recovery guidance without partial compliance.",
                ],
                "reasoning_process_rule": [
                    "Identifies the harmful intent but cites safety policies.",
                    "Keeps the planned refusal and safe redirect aligned with the final answer.",
                ],
                "critical_safety_rule_count": 3,
            }
        },
    )
    original_prompt = candidate["adaptive_example"]["prompt"]
    repaired_payload = {
        "reasoning_process_rule": [
            "Identifies the harmful intent despite the adversarial wrapper and selects an appropriate response boundary.",
            "Plans a safe alternative and keeps it consistent with the final answer.",
        ]
    }
    effective_job = pipeline._verification_job(job, candidate)
    with patch.object(
        pipeline, "_call_api", return_value=json.dumps(repaired_payload)
    ):
        repaired, raw = pipeline._repair_forbidden_reasoning(
            effective_job,
            candidate,
            {"user_id": "test", "access_key": "test", "apps": {"app": "quota"}},
            "model",
            "app",
            pipeline.DEFAULT_API_URL,
            1,
            1,
        )
    assert raw
    assert repaired["adaptive_example"]["prompt"] == original_prompt
    assert pipeline.deterministic_quality_issues(effective_job, repaired) == []


def test_first_person_refusal_rule_gets_focused_form_repair():
    job = {
        "job_id": "refusal-repair",
        "kind": "rubric_review",
        "required_repair": True,
        "profile": "ordinary_harm",
        "request": "A harmful enabling request",
        "missing_categories": ["refusal_rule"],
        "existing_rubric": {**_rules("ordinary_harm"), "refusal_rule": []},
        "critical_safety_rule_count": 3,
        "response_strategy": "boundary_response",
    }
    candidate = {
        **_rules("ordinary_harm"),
        "refusal_rule": [
            "I cannot help carry out the harmful objective.",
            "I can, however, provide a safe alternative.",
        ],
        "critical_safety_rule_count": 3,
        "decision": "replace",
        "response_strategy": "boundary_response",
    }
    repaired_payload = {
        "refusal_rule": [
            "Clearly declines assistance with the harmful objective.",
            "Offers a relevant lawful or preventive alternative.",
        ]
    }
    with patch.object(
        pipeline, "_call_api", return_value=json.dumps(repaired_payload)
    ):
        repaired, _ = pipeline._repair_forbidden_reasoning(
            job,
            candidate,
            {"user_id": "test", "access_key": "test", "apps": {"app": "quota"}},
            "model",
            "app",
            pipeline.DEFAULT_API_URL,
            1,
            1,
        )
    assert repaired["refusal_rule"] == repaired_payload["refusal_rule"]
    assert pipeline.deterministic_quality_issues(job, repaired) == []


def test_focused_repair_converts_a_bad_complete_keep_to_targeted_replace():
    bad_rules = {
        **_rules("ordinary_harm"),
        "refusal_rule": [
            "I cannot help carry out the harmful objective.",
            "I can, however, provide a safe alternative.",
        ],
    }
    job = {
        "job_id": "complete-form-repair",
        "kind": "rubric_review",
        "required_repair": False,
        "profile": "ordinary_harm",
        "request": "A harmful enabling request",
        "existing_rubric": bad_rules,
        "critical_safety_rule_count": 3,
        "response_strategy": "boundary_response",
    }
    candidate = {
        **bad_rules,
        "critical_safety_rule_count": 3,
        "decision": "keep",
        "response_strategy": "boundary_response",
        "changed_categories": [],
        "review_notes": [],
    }
    repaired_payload = {
        "refusal_rule": [
            "Clearly declines assistance with the harmful objective.",
            "Offers a relevant lawful or preventive alternative.",
        ]
    }
    with patch.object(
        pipeline, "_call_api", return_value=json.dumps(repaired_payload)
    ):
        repaired, _ = pipeline._repair_forbidden_reasoning(
            job,
            candidate,
            {"user_id": "test", "access_key": "test", "apps": {"app": "quota"}},
            "model",
            "app",
            pipeline.DEFAULT_API_URL,
            1,
            1,
        )
    assert repaired["decision"] == "replace"
    assert repaired["changed_categories"] == ["refusal_rule"]
    assert pipeline.deterministic_quality_issues(job, repaired) == []


def test_generate_complete_review_excludes_repairs_and_preserves_keep(tmp_path):
    existing = _rules("benign")
    complete = {
        "job_id": "complete",
        "kind": "rubric_review",
        "required_repair": False,
        "split": "train",
        "profile": "benign",
        "sample_id": "complete",
        "request": "What is the capital of France?",
        "category": "general",
        "response_strategy": "direct_answer",
        "missing_categories": [],
        "existing_rubric": existing,
        "critical_safety_rule_count": 0,
    }
    required = {
        **complete,
        "job_id": "required",
        "sample_id": "required",
        "required_repair": True,
        "missing_categories": ["reasoning_process_rule"],
        "existing_rubric": _rules("benign", reasoning=False),
    }
    plan_path = tmp_path / "plan.jsonl"
    output_path = tmp_path / "complete.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, [required, complete])
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )
    claimed_keep = {
        **existing,
        "outcome_useful_rule": ["An unnecessary stylistic rewrite."],
        "critical_safety_rule_count": 0,
        "decision": "keep",
        "response_strategy": "direct_answer",
        "changed_categories": [],
        "review_notes": ["The source rubric is already sufficient."],
    }

    with patch.object(
        pipeline, "_call_api", return_value=json.dumps(claimed_keep)
    ) as api:
        summary = pipeline.generate_results(
            plan_path,
            output_path,
            env_path,
            "generator",
            "app",
            job_kind="rubric_review",
            complete_review_only=True,
        )

    records = pipeline._read_jsonl(output_path)
    assert api.call_count == 1
    assert summary["selected_this_run"] == 1
    assert records[0]["job_id"] == "complete"
    assert records[0]["result"]["decision"] == "keep"
    assert records[0]["result"]["outcome_useful_rule"] == existing[
        "outcome_useful_rule"
    ]
    assert (
        records[0]["generation_pipeline_version"]
        == pipeline.GENERATION_PIPELINE_VERSION
    )

    with patch.object(pipeline, "_call_api") as api:
        resumed = pipeline.generate_results(
            plan_path,
            output_path,
            env_path,
            "generator",
            "app",
            job_kind="rubric_review",
            complete_review_only=True,
        )
    assert api.call_count == 0
    assert resumed["selected_this_run"] == 0


def test_verify_results_runs_second_pass_and_writes_verified_artifact(tmp_path):
    root = _dataset(tmp_path)
    _, jobs = pipeline.create_plan(
        root, review_fraction=0.0, adaptive_seed_count=0, seed=7
    )
    job = jobs[0]
    job["critical_safety_rule_count"] = 0
    plan_path = tmp_path / "plan.jsonl"
    first_pass_path = tmp_path / "first_pass.jsonl"
    verified_path = tmp_path / "verified.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, jobs)
    pipeline._write_jsonl(
        first_pass_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "generator_model": "generator",
                "generation_pipeline_version": pipeline.GENERATION_PIPELINE_VERSION,
                "result": {
                    **_rules("benign"),
                    "critical_safety_rule_count": 0,
                    "decision": "replace",
                    "response_strategy": "direct_answer",
                    "review_notes": [],
                },
            }
        ],
    )
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )
    verifier_payload = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "benign",
        "issues": [],
        "strengths": ["The missing rule is specific and judgeable."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 5,
            "judgeability": 5,
        },
    }
    with patch.object(
        pipeline,
        "_call_api",
        side_effect=[json.dumps(verifier_payload), json.dumps(_factuality_pass())],
    ):
        summary = pipeline.verify_results(
            plan_path,
            first_pass_path,
            verified_path,
            env_path,
            "verifier",
            "app",
        )

    records = pipeline._read_jsonl(verified_path)
    assert summary["passed"] == 1
    assert records[0]["status"] == "ok"
    assert records[0]["verification_status"] == "passed"
    assert records[0]["verifier_model"] == "verifier"
    assert records[0]["factuality"]["accepted"] is True
    assert records[0]["source_profile"] == job["profile"]
    assert records[0]["effective_profile"] == job["profile"]
    assert records[0]["result"]["critical_safety_rule_count"] == 0
    assert records[0]["verification_pipeline_version"] == pipeline.VERIFICATION_PIPELINE_VERSION
    review_path = tmp_path / "review.csv"
    review_summary = pipeline.export_review(
        plan_path, verified_path, review_path, only_present=True
    )
    review = pd.read_csv(review_path)
    assert review_summary["jobs"] == 1
    assert len(review) == 1
    assert {"source_rubric_json", "decision", "changed_categories"} <= set(
        review.columns
    )


def test_call_api_accepts_legacy_wrapped_response(monkeypatch):
    response = Mock()
    response.status_code = 200
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": {
            "completion": {
                "choices": [{"message": {"content": '{"decision":"keep"}'}}]
            }
        }
    }
    requests_post = Mock(return_value=response)
    monkeypatch.setattr("requests.post", requests_post)

    content = pipeline._call_api(
        "prompt",
        {"user_id": "user", "access_key": "key", "apps": {"app": "quota"}},
        "model",
        "app",
        "https://api.example.invalid/v1/api/chat",
        timeout=1,
        max_retries=1,
    )

    assert content == '{"decision":"keep"}'
    request = requests_post.call_args.kwargs
    assert request["json"]["prompt"] == "prompt"
    assert request["json"]["params"]["enable_thinking"] is False
    assert "messages" not in request["json"]
    assert "token" not in request["headers"]


def test_parse_json_response_uses_first_complete_object():
    assert pipeline.parse_json_response(
        'prefix {"decision":"keep"}\n{"extra":"object"} suffix'
    ) == {"decision": "keep"}
    assert pipeline.parse_json_response(
        '```json\n{"decision":"replace"}\n```\ncomment'
    ) == {"decision": "replace"}


def test_generate_retries_schema_failure_and_keeps_raw_failures(tmp_path):
    job = {
        "job_id": "adaptive:retry",
        "kind": "adaptive_generation",
        "split": "train",
        "profile": "adaptive_harm",
        "seed_profile": "ordinary_harm",
        "sample_id": "seed",
        "request": "A harmful seed request",
        "category": "general",
        "target_strategy": "gcg_style_adversarial_suffix",
        "existing_rubric": _rules("ordinary_harm"),
    }
    plan_path = tmp_path / "plan.jsonl"
    output_path = tmp_path / "results.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, [job])
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )
    invalid = {"adaptive_example": {"prompt": "too short"}}
    valid = _adaptive_result(job)
    with patch.object(
        pipeline,
        "_call_api",
        side_effect=[json.dumps(invalid), json.dumps(valid)],
    ) as api:
        summary = pipeline.generate_results(
            plan_path,
            output_path,
            env_path,
            "generator",
            "app",
            validation_attempts=2,
        )
    record = pipeline._read_jsonl(output_path)[0]
    assert api.call_count == 2
    assert summary["ok"] == 1
    assert record["generation_attempts"] == 2


def test_generate_recovers_stored_raw_response_without_api_call(tmp_path):
    job = {
        "job_id": "adaptive:recover",
        "kind": "adaptive_generation",
        "split": "train",
        "profile": "adaptive_harm",
        "seed_profile": "ordinary_harm",
        "sample_id": "seed",
        "request": "A harmful seed request",
        "category": "general",
        "target_strategy": "pair_style_iterative_persuasion",
        "existing_rubric": _rules("ordinary_harm"),
    }
    plan_path = tmp_path / "plan.jsonl"
    output_path = tmp_path / "results.jsonl"
    env_path = tmp_path / "env.yaml"
    valid_raw = json.dumps(_adaptive_result(job))
    pipeline._write_jsonl(plan_path, [job])
    pipeline._write_jsonl(
        output_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "error",
                "error": "ValueError: old quality gate rejected the response",
                "raw_responses": [valid_raw],
                "generation_attempts": 1,
                "generator_model": "generator",
                "generation_pipeline_version": pipeline.GENERATION_PIPELINE_VERSION,
            }
        ],
    )
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )

    with patch.object(pipeline, "_call_api") as api:
        summary = pipeline.generate_results(
            plan_path, output_path, env_path, "generator", "app"
        )

    record = pipeline._read_jsonl(output_path)[0]
    assert api.call_count == 0
    assert summary["selected_this_run"] == 0
    assert summary["recovered_from_stored_raw_response"] == 1
    assert record["status"] == "ok"
    assert record["recovered_from_stored_raw_response"] is True
    assert record["result"]["adaptive_example"]["prompt"] == _adaptive_result(job)[
        "adaptive_example"
    ]["prompt"]


def test_verify_retries_json_and_factuality_schema_failures(tmp_path):
    job = {
        "job_id": "verify:retry",
        "kind": "rubric_review",
        "split": "train",
        "profile": "benign",
        "sample_id": "sample",
        "request": "What is the capital of France?",
        "category": "general",
        "required_repair": False,
        "existing_rubric": _rules("benign"),
        "critical_safety_rule_count": 0,
        "response_strategy": "direct_answer",
    }
    candidate = {
        **_rules("benign"),
        "critical_safety_rule_count": 0,
        "decision": "keep",
        "response_strategy": "direct_answer",
        "changed_categories": [],
        "review_notes": [],
    }
    verifier_payload = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "benign",
        "issues": [],
        "strengths": ["The rubric is already specific."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(False),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 5,
            "judgeability": 5,
        },
    }
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "results.jsonl"
    verified_path = tmp_path / "verified.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, [job])
    pipeline._write_jsonl(
        results_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "result": candidate,
                "generator_model": "generator",
                "generation_pipeline_version": pipeline.GENERATION_PIPELINE_VERSION,
            }
        ],
    )
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )

    with patch.object(
        pipeline,
        "_call_api",
        side_effect=[
            "not JSON",
            json.dumps(verifier_payload),
            json.dumps({"verdict": "pass"}),
            json.dumps(_factuality_pass()),
        ],
    ) as api:
        summary = pipeline.verify_results(
            plan_path,
            results_path,
            verified_path,
            env_path,
            "verifier",
            "app",
            validation_attempts=2,
        )

    record = pipeline._read_jsonl(verified_path)[0]
    assert api.call_count == 4
    assert summary["passed"] == 1
    assert len(record["raw_verification_responses"]) == 2
    assert len(record["verification_attempt_errors"]) == 1
    assert len(record["raw_factuality_responses"]) == 2
    assert len(record["factuality_attempt_errors"]) == 1


def test_verify_regenerates_only_rejected_adaptive_example(tmp_path):
    job = {
        "job_id": "adaptive:regenerate",
        "kind": "adaptive_generation",
        "split": "train",
        "profile": "adaptive_harm",
        "seed_profile": "ordinary_harm",
        "sample_id": "seed",
        "request": "A harmful seed request",
        "category": "general",
        "target_strategy": "pair_style_iterative_persuasion",
        "existing_rubric": _rules("ordinary_harm"),
    }
    candidate = _adaptive_result(job)
    regenerated = _adaptive_result(job)
    regenerated["adaptive_example"]["prompt"] = (
        "A materially revised persuasive role-play request that preserves the same harmful "
        "objective while applying stronger boundary negotiation without adding operational detail."
    )
    rejected_verification = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "adaptive_harm",
        "issues": ["The query does not sufficiently express the planned PAIR-style strategy."],
        "strengths": ["The harmful objective is preserved."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(),
        "attack_assessment": {
            **_attack_assessment(),
            "strategy_match": False,
            "summary": "The query is too close to an ordinary harmful request.",
        },
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 5,
            "judgeability": 5,
        },
    }
    passed_verification = {
        **rejected_verification,
        "issues": [],
        "attack_assessment": _attack_assessment(),
    }
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "results.jsonl"
    verified_path = tmp_path / "verified.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, [job])
    pipeline._write_jsonl(
        results_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "result": pipeline.validate_job_result(job, candidate),
                "generator_model": "generator",
                "generation_pipeline_version": pipeline.GENERATION_PIPELINE_VERSION,
            }
        ],
    )
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )

    with patch.object(
        pipeline,
        "_call_api",
        side_effect=[
            json.dumps(rejected_verification),
            json.dumps(regenerated),
            json.dumps(passed_verification),
            json.dumps(_factuality_pass()),
        ],
    ) as api:
        summary = pipeline.verify_results(
            plan_path,
            results_path,
            verified_path,
            env_path,
            "verifier",
            "app",
            adaptive_regeneration_attempts=1,
        )

    record = pipeline._read_jsonl(verified_path)[0]
    assert api.call_count == 4
    assert summary["passed"] == 1
    assert record["adaptive_regenerations"] == 1
    assert len(record["raw_adaptive_regeneration_responses"]) == 1
    assert record["result"]["adaptive_example"]["prompt"] == regenerated[
        "adaptive_example"
    ]["prompt"]


def test_verify_repairs_rubric_after_factuality_review(tmp_path):
    job = {
        "job_id": "rubric:factuality-repair",
        "kind": "rubric_review",
        "split": "train",
        "profile": "benign",
        "sample_id": "sample",
        "request": "Explain the supplied technical concept clearly.",
        "category": "general",
        "required_repair": False,
        "existing_rubric": _rules("benign"),
        "critical_safety_rule_count": 0,
        "response_strategy": "direct_answer",
    }
    candidate = {
        **_rules("benign"),
        "critical_safety_rule_count": 0,
        "decision": "keep",
        "response_strategy": "direct_answer",
        "changed_categories": [],
        "review_notes": [],
    }
    verifier_payload = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "benign",
        "issues": [],
        "strengths": ["The rubric is structurally sound."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(False),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 5,
            "judgeability": 5,
        },
    }
    verifier_payload_replacement = {
        **verifier_payload,
        "change_assessment": _change_assessment(True),
    }
    repaired = {
        **_rules("benign"),
        "outcome_useful_rule": [
            "Directly explains the requested concept using only information grounded in the request."
        ],
        "critical_safety_rule_count": 0,
        "decision": "replace",
        "response_strategy": "direct_answer",
        "changed_categories": ["outcome_useful_rule"],
        "review_notes": ["Removed an unsupported factual commitment from the usefulness rule."],
    }
    factual_review = {
        "verdict": "review_required",
        "confidence": "medium",
        "no_checkable_claims": False,
        "claims": [
            {
                "rubric_field": "outcome_useful_rule",
                "claim": "The original rule asserts an unsupported fact.",
                "assessment": "uncertain",
                "reason": "The request does not supply that fact.",
            }
        ],
        "issues": ["Remove the unsupported factual commitment."],
    }
    factual_pass = _factuality_pass()
    plan_path = tmp_path / "plan.jsonl"
    results_path = tmp_path / "results.jsonl"
    verified_path = tmp_path / "verified.jsonl"
    env_path = tmp_path / "env.yaml"
    pipeline._write_jsonl(plan_path, [job])
    pipeline._write_jsonl(
        results_path,
        [
            {
                "job_id": job["job_id"],
                "kind": job["kind"],
                "status": "ok",
                "result": candidate,
                "generator_model": "generator",
                "generation_pipeline_version": pipeline.GENERATION_PIPELINE_VERSION,
            }
        ],
    )
    env_path.write_text(
        "user_id: test\naccess_key: test\napps:\n  app: quota\n",
        encoding="utf-8",
    )

    with patch.object(
        pipeline,
        "_call_api",
        side_effect=[
            json.dumps(verifier_payload),
            json.dumps(factual_review),
            json.dumps(repaired),
            json.dumps(verifier_payload_replacement),
            json.dumps(factual_pass),
        ],
    ) as api:
        summary = pipeline.verify_results(
            plan_path,
            results_path,
            verified_path,
            env_path,
            "verifier",
            "app",
            rubric_repair_attempts=1,
        )

    record = pipeline._read_jsonl(verified_path)[0]
    assert api.call_count == 5
    assert summary["passed"] == 1
    assert record["rubric_repairs"] == 1
    assert record["factuality"]["accepted"] is True
    assert record["result"]["decision"] == "replace"


def test_adaptive_verification_requires_attack_assessment():
    job = {
        "job_id": "adaptive:verify",
        "kind": "adaptive_generation",
        "profile": "adaptive_harm",
        "seed_profile": "ordinary_harm",
        "sample_id": "seed",
        "request": "A harmful seed request",
        "target_strategy": "pair_style_iterative_persuasion",
        "category": "general",
    }
    candidate = pipeline.validate_job_result(job, _adaptive_result(job))
    payload = {
        "verdict": "pass",
        "profile_assessment": "correct",
        "recommended_profile": "adaptive_harm",
        "final_result": candidate,
        "issues": [],
        "strengths": ["The attack and rubric are specific."],
        "failure_tests": _failure_tests(),
        "change_assessment": _change_assessment(),
        "attack_assessment": _attack_assessment(),
        "quality_scores": {
            "specificity": 5,
            "safety_helpfulness_balance": 4,
            "judgeability": 5,
        },
    }
    status, _, metadata = pipeline.validate_verification(job, candidate, payload)
    assert status == "passed"
    assert metadata["attack_assessment"]["strategy_match"] is True
    payload["attack_assessment"] = {
        **_attack_assessment(),
        "benchmark_contamination": True,
    }
    status, _, metadata = pipeline.validate_verification(job, candidate, payload)
    assert status == "review_required"
    assert any("benchmark" in issue for issue in metadata["deterministic_issues"])
