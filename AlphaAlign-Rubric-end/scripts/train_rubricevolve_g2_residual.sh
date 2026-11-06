#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_EXPERIMENT_NAME=rubricevolve-g2-residual
export RUBRIC_TARGETED_EXPERIMENT_ID=g2_residual
export RUBRIC_EVOLVE_REWARD_MODE=bounded_residual
export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0
export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=1
export RUBRIC_EVOLVE_GROUP_RANK_GUARD=0
exec bash "${SCRIPT_DIR}/train_rubricevolve_guarded_residual.sh" "$@"
