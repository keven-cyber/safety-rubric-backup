#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
case "${1:-}" in g1|g3) ;; *) echo 'Expected g1 or g3 frozen review' >&2; exit 2;; esac
: "${RUBRIC_FROZEN_REVIEW_SOURCE_ROOT:?Missing explicit completed source root}"
: "${RUBRIC_RECOVERY_SOURCE_CHECKPOINT:?Missing complete1170 preservation source}"
if [ "${RUBRIC_RECOVERY_SOURCE_CHECKPOINT}" != "${RUBRIC_FROZEN_REVIEW_SOURCE_ROOT}/global_step_1170" ]; then
  echo 'Frozen review requires the completed1170 source of this exact job' >&2
  exit 2
fi
export RUBRIC_FROZEN_REVIEW_ENABLE=1
export RUBRIC_FROZEN_REVIEW_AUTHORIZATION=user_authorized_new_generations_20261102
export RUBRIC_EVAL_ONLY=0
export RUBRIC_INTERMEDIATE_RECORD_ENABLE=1
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8_legacy.sh" "$@"
