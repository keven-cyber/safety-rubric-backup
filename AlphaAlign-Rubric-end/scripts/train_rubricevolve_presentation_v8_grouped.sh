#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE=1
export RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING=1
export RUBRIC_JUDGE_THROUGHPUT_CANARY_ENABLE=0
# This is a new full-rubric evidence protocol, not the ungrounded legacy flag.
# Original strict semantic/impact checks and the real restored8x8 gate still run.
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8_fast.sh" "$@"
