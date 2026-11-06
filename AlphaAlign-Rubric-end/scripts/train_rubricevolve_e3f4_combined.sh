#!/usr/bin/env bash
# Combined replay-balance and benign penalty arm.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_EXPERIMENT_NAME=rubricevolve-e3f4-combined
export RUBRIC_TARGETED_EXPERIMENT_ID=e3f4_combined
export RUBRIC_EVOLVE_REPLAY_MODE_BALANCE=1
export RUBRIC_EVOLVE_ANTI_REFUSAL_PENALTY_HIGH=1
exec bash "${SCRIPT_DIR}/train_rubricevolve_e3_refusal_common.sh" "$@"
