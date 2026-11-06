#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_JUDGE_THROUGHPUT_CANARY_ENABLE=1
# The canary selects144 only after exact raw-score/reward/update equivalence
# and a measured speed benefit. Otherwise the existing48-worker path remains.
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8_fast.sh" "$@"
