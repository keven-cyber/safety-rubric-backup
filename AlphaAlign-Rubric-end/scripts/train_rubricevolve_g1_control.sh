#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repaired evolution control, not the untouched historical non-evolution run.
export TRAIN_EXPERIMENT_NAME=rubricevolve-g1-control
export RUBRIC_TARGETED_EXPERIMENT_ID=g1_control
export RUBRIC_EVOLVE_REWARD_MODE=legacy
export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0
export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_GROUP_RANK_GUARD=0
exec bash "${SCRIPT_DIR}/train_rubricevolve_guarded_residual.sh" "$@"
