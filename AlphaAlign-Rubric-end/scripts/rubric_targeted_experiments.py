#!/usr/bin/env python3
"""Evidence-driven RubricAlign experiment matrix after the E0--E7 audit."""
import argparse
import json
import shlex


TARGETED_STRATEGIES = ",".join(
    (
        "trigger_roleplay_override",
        "encoded_transformation_wrapper",
        "benign_research_pretext",
        "multi_turn_persuasion_summary",
        "implicit_chained_enablement",
        "refusal_leakage_pressure",
    )
)

# review_fraction, train hard-negative seeds, validation seeds, revision, gate
ARMS = {
    "A": ("hard_negative", "0.0", "80", "8", "old", "0"),
    "B": ("critical_rubric", "1.0", "0", "0", "critical_content", "0"),
    "C": ("hard_negative_critical_rubric", "1.0", "80", "8", "critical_content", "0"),
    "D1": ("combined_conservative_gate", "1.0", "80", "8", "critical_content", "critical"),
    "D2": ("combined_strict_gate", "1.0", "80", "8", "critical_content", "strict"),
    "E": ("combined_strict_more_hard", "1.0", "120", "8", "critical_content", "strict"),
}


def configuration(arm):
    name, review_fraction, adaptive_seeds, adaptive_val_seeds, revision, gate = ARMS[arm]
    env = {
        "RUBRIC_TARGETED_EXPERIMENT_ID": arm,
        "TRAIN_EXPERIMENT_NAME": f"rubricalign_targeted_{arm}_{name}",
        "RUBRIC_REVISION_MODE": revision,
        "RUBRIC_REVIEW_FRACTION": review_fraction,
        "RUBRIC_ADAPTIVE_SEEDS": adaptive_seeds,
        "RUBRIC_ADAPTIVE_VAL_SEEDS": adaptive_val_seeds,
        "RUBRIC_ADAPTIVE_STRATEGIES": TARGETED_STRATEGIES,
        "RUBRIC_CRITICAL_FAILURE_GATE": gate,
        "RUBRIC_STRICT_GATE_SAFETY_THRESHOLD": "1.0",
        "RUBRIC_STRICT_GATE_BOUNDARY_THRESHOLD": "1.0",
        "RUBRIC_CRITICAL_GATE_SCORE": "-1.0",
        "REWARD_MODE": "rubric_safe_helpful_balanced",
        "RUBRIC_GUARD_ENABLE": "0",
        "RUBRIC_EVOLVE": "0",
        "TRAINING_DATA_VERSION": "new_v3",
        "TRAIN_PROMPT_VERSION": "structured_steps",
        "RUBRIC_JUDGE_GRANULARITY": "single_rule",
        "RUBRIC_JUDGE_STRICT_PARSE": "1",
        "RUBRIC_JUDGE_GUIDED_OUTPUT": "1",
        "RUBRIC_JUDGE_DROP_FAILED": "1",
        "RUBRIC_REQUIRE_OSS": "1",
        "RUBRIC_INTERMEDIATE_RECORD_ENABLE": "0",
        "RUBRIC_LIGHT_DIAGNOSTICS_ENABLE": "1",
        "RUBRIC_LIGHT_DIAGNOSTIC_STEPS": "300,600,900,990",
        "RUBRIC_LIGHT_DIAGNOSTIC_MAX_PER_PROFILE": "8",
        "TRAIN_LEARNING_RATE": "1e-6",
        "TRAIN_USE_KL_LOSS": "False",
        "TRAIN_TOTAL_EPOCHS": "3",
        "TRAIN_SAVE_FREQ": "30",
        "TRAIN_TEST_FREQ": "30",
        "TRAIN_SAVE_BEST_CKPT": "True",
        "TRAIN_BEST_METRIC_KEY": "val-core/balanced/safety_constrained_score",
        "TRAIN_BEST_METRIC_MODE": "max",
        "FINAL_EVAL_EXACT_TRIO": "1",
        "FINAL_SAFETY_EVAL_STEPS": "900,990,best",
        "FINAL_EVAL_MIN_STEP": "0",
        "FINAL_SAFETY_EVAL_ENABLE": "1",
        "FINAL_GENERAL_EVAL_ENABLE": "1",
        "RUBRIC_EVAL_ONLY": "0",
    }
    return {
        "schema_version": 1,
        "arm": arm,
        "name": name,
        "environment": env,
        "resource_pool": "wings-L20Z-T04-agentic",
        "nodes": 1,
        "gpus": 8,
        "source_dataset": "updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1",
        "seed_policy": "unchanged repository default; no seed sweep",
        "guard_supervision": False,
        "official_benchmark_training_data": False,
        "final_checkpoints": [900, 990, "best"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--shell", action="store_true")
    args = parser.parse_args()
    config = configuration(args.arm)
    if args.shell:
        for key, value in config["environment"].items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        print(json.dumps(config, indent=2))
