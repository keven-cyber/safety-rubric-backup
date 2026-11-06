#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_SELECTION_V2_ENABLE=1
export RUBRIC_SELECTION_STRICT_BEST=1
export TRAIN_BEST_METRIC_KEY=val-core/selection_v2/rank_score
export TRAIN_EXPERIMENT_NAME=rubricevolve-g3-calibrated-selection
export RUBRIC_TARGETED_EXPERIMENT_ID=g3_calibrated_selection
export RUBRIC_BENIGN_CONTRAST_ENABLE=1
export RUBRIC_EVOLVE_REWARD_MODE=bounded_residual
export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0.25
export RUBRIC_EVOLVE_STD_FLOOR_SCOPE=affected_groups
export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=1
export RUBRIC_EVOLVE_GROUP_RANK_GUARD=0
exec bash "${SCRIPT_DIR}/train_rubricevolve_guarded_residual.sh" "$@"
