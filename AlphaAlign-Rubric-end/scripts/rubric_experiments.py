#!/usr/bin/env python3
"""Versioned eight-arm experiment matrix; no API, GPU or secret access."""
import argparse
import json
import shlex
from pathlib import Path

ARMS = {
    "E0": ("old_baseline", "old", "1e-6", "False", "0"),
    "E1": ("safety_semantics", "safety_semantics", "1e-6", "False", "0"),
    "E2": ("decision_boundaries", "decision_boundaries", "1e-6", "False", "0"),
    "E3": ("combined", "combined", "1e-6", "False", "0"),
    "E4": ("old_half_lr", "old", "5e-7", "False", "0"),
    "E5": ("old_kl", "old", "1e-6", "True", "0"),
    "E6": ("old_critical_gate", "old", "1e-6", "False", "1"),
    "E7": ("old_critical_gate_half_lr", "old", "5e-7", "False", "1"),
}


def configuration(arm):
    name, revision, lr, kl, gate = ARMS[arm]
    env = {
        "RUBRIC_EXPERIMENT_ID": arm, "TRAIN_EXPERIMENT_NAME": f"rubricalign_{arm}_{name}",
        "RUBRIC_REVISION_MODE": revision, "TRAIN_LEARNING_RATE": lr,
        "TRAIN_USE_KL_LOSS": kl, "TRAIN_KL_LOSS_COEF": "0.001",
        "RUBRIC_CRITICAL_FAILURE_GATE": gate,
        "REWARD_MODE": "rubric_safe_helpful_balanced", "RUBRIC_GUARD_ENABLE": "0",
        "RUBRIC_EVOLVE": "0", "TRAINING_DATA_VERSION": "new_v3",
        "TRAIN_PROMPT_VERSION": "structured_steps", "RUBRIC_JUDGE_GRANULARITY": "single_rule",
        "RUBRIC_JUDGE_STRICT_PARSE": "1", "RUBRIC_JUDGE_GUIDED_OUTPUT": "1",
        "RUBRIC_JUDGE_DROP_FAILED": "1",
        "RUBRIC_REQUIRE_OSS": "1",
        "RUBRIC_INTERMEDIATE_RECORD_ENABLE": "0", "TRAIN_TOTAL_EPOCHS": "3",
        "TRAIN_SAVE_FREQ": "30", "TRAIN_TEST_FREQ": "30", "TRAIN_SAVE_BEST_CKPT": "True",
        "TRAIN_BEST_METRIC_KEY": "val-core/balanced/pareto_score", "TRAIN_BEST_METRIC_MODE": "max",
        "FINAL_EVAL_EXACT_TRIO": "1", "FINAL_SAFETY_EVAL_STEPS": "900,990,best",
        "FINAL_EVAL_MIN_STEP": "0", "FINAL_SAFETY_EVAL_ENABLE": "1",
        "FINAL_GENERAL_EVAL_ENABLE": "1", "RUBRIC_EVAL_ONLY": "0",
    }
    return {"schema_version": 1, "arm": arm, "name": name, "environment": env,
            "resource_pool": "wings-L20Z-T04-agentic", "nodes": 1, "gpus": 8,
            "source_dataset": "updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1",
            "seed_policy": "unchanged repository defaults; no seed sweep",
            "guard_alignment": "not_empirically_validated",
            "api_revision": revision != "old", "final_checkpoints": [900, 990, "best"]}


def check_task_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Primus task configuration must be an object")
    recovery = config.get("rubric_eval", {"enabled": False})
    if not isinstance(recovery, dict) or recovery.get("enabled") is not False:
        raise ValueError("Training experiments cannot use an eval-only Primus configuration")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm", choices=ARMS)
    parser.add_argument("--shell", action="store_true")
    parser.add_argument("--check-task-config", type=Path)
    args = parser.parse_args()
    config = configuration(args.arm)
    if args.check_task_config:
        check_task_config(args.check_task_config)
    if args.shell:
        for key, value in config["environment"].items():
            print(f"export {key}={shlex.quote(value)}")
    else:
        print(json.dumps(config, indent=2))
