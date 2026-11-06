#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec bash "${PROJECT_ROOT}/train_val.sh" --help
fi
export REWARD_MODE=rubric_safe_helpful_balanced
export RUBRIC_GUARD_ENABLE=0
export RUBRIC_EVOLVE=0
export TRAINING_DATA_VERSION=new_v3
export RUBRIC_DATASET_DIR_NAME=updated_training_rubric_dataset_safety_v4
export RUBRIC_JUDGE_STRICT_PARSE=1
export RUBRIC_JUDGE_GUIDED_OUTPUT="${RUBRIC_JUDGE_GUIDED_OUTPUT:-1}"
export RUBRIC_REQUIRE_OSS="${RUBRIC_REQUIRE_OSS:-1}"
export RUBRIC_INTERMEDIATE_RECORD_ENABLE="${RUBRIC_INTERMEDIATE_RECORD_ENABLE:-0}"
export ROLLOUT_TRACE_ENABLE="${RUBRIC_INTERMEDIATE_RECORD_ENABLE}"
export ROLLOUT_TRACE_MAX_PER_STEP=0
source "${PROJECT_ROOT}/scripts/rubric_eval_mode.sh"
rubric_configure_eval_mode
dataset="${FILE_PATH:-${PROJECT_ROOT}/Evaluation/${RUBRIC_DATASET_DIR_NAME}}"
if [ "${RUBRIC_EVAL_ONLY}" != "1" ] && [ ! -f "${dataset}/manifest.json" ]; then
  python3 "${PROJECT_ROOT}/scripts/prepare_safety_rubric_v4.py" \
    --source "${PROJECT_ROOT}/Evaluation/updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1" \
    --output "${dataset}"
fi
export FILE_PATH="${dataset}"
source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
# A new data experiment must not silently auto-resume an old v3 checkpoint.
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints}/safety_v4/${RUBRIC_RUN_ID}}"
exec bash "${PROJECT_ROOT}/train_val.sh" "$@"
