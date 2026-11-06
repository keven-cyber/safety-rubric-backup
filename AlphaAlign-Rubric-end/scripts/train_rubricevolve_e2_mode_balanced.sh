#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# E2: retain rare harmful failure modes in the replay bank, instead of letting
# the most recent common failure mode dominate all proposal/held-out examples.
export TRAIN_EXPERIMENT_NAME=rubricevolve-e2-mode-balanced
export RUBRIC_TARGETED_EXPERIMENT_ID=e2_mode_balanced
export RUBRIC_EVOLVE_EXPERIMENT_FULL=1
export RUBRIC_EVOLVE_ALLOWED_PROFILES=ordinary_harm,adaptive_harm
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES=outcome_safety_rule
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_REPLAY_MODE_BALANCE=1
export RUBRIC_EVOLVE_REPLAY_MAX_PER_LABEL=48
export RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL=12
export RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL=6
export RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP=1
exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
