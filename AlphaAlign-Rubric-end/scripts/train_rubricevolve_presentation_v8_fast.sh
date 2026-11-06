#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "${RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE:-0}" = "1" ]; then
  export RUBRIC_JUDGE_SERVING_PROFILE=legacy_compact_v8
elif [ "${RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE:-0}" = "1" ]; then
  export RUBRIC_JUDGE_SERVING_PROFILE=grouped_context_v8
else
  export RUBRIC_JUDGE_SERVING_PROFILE=batched_v8
fi
export RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT=48
export RUBRIC_JUDGE_API_CONCURRENCY=48
export RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE=1
export RUBRIC_DURABLE_TRACKING_ENABLE=1
export RUBRIC_VALIDATION_QUARANTINE_ENABLE=1
export RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE=1
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR:?Missing persistent checkpoint directory}/presentation_v8_batched_${PRIMUS_JOB_SUBMIT_TIMESTAMP:?Missing submission timestamp}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
if [ -n "${RUBRIC_RECOVERY_SOURCE_CHECKPOINT:-}" ]; then
  python3 "${SCRIPT_DIR}/../verl/utils/checkpoint_recovery.py" \
    --source "${RUBRIC_RECOVERY_SOURCE_CHECKPOINT}" --destination "${CKPT_PATH}" \
    --persistent-root "${PRIMUS_SAVE_CHECKPOINT_DIR}"
  export RUBRIC_CHECKPOINT_RECOVERY_ENABLE=1
  export TRAIN_RESUME_MODE=auto
else
  export RUBRIC_CHECKPOINT_RECOVERY_ENABLE=0
  export TRAIN_RESUME_MODE=disable
fi
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8.sh" "$@"
