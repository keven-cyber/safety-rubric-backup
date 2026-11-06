#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Same G1 data/optimizer, opt-in calibrated benign-only within-prompt correction.
# No external test set or Guard contributes to training reward.
export RUBRIC_BENIGN_CONTRAST_ENABLE=1
export RUBRIC_SELECTION_V2_ENABLE=1
export RUBRIC_SELECTION_STRICT_BEST=1
export TRAIN_BEST_METRIC_KEY=val-core/selection_v2/rank_score
export TRAIN_EXPERIMENT_NAME=rubricevolve-g1-calibrated-selection
export RUBRIC_TARGETED_EXPERIMENT_ID=g1_calibrated_selection
export RUBRIC_EVOLVE_REWARD_MODE=legacy
export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0
export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_GROUP_RANK_GUARD=0
exec bash "${SCRIPT_DIR}/train_rubricevolve_guarded_residual.sh" "$@"
