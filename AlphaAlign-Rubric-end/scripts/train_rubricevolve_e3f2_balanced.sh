#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_EXPERIMENT_NAME=rubricevolve-e3f2-balanced
export RUBRIC_TARGETED_EXPERIMENT_ID=e3f2_balanced
export RUBRIC_EVOLVE_REPLAY_MODE_BALANCE=1
export RUBRIC_EVOLVE_ANTI_REFUSAL_PENALTY_HIGH=0
exec bash "${SCRIPT_DIR}/train_rubricevolve_e3_refusal_common.sh" "$@"
