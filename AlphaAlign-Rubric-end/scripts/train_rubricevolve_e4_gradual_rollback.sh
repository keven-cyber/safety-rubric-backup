#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# E4: phase-in each promoted safety rule, and retire it if fresh labeled
# rollouts show a supported safety/helpfulness regression.
export TRAIN_EXPERIMENT_NAME=rubricevolve-e4-gradual-rollback
export RUBRIC_TARGETED_EXPERIMENT_ID=e4_gradual_rollback
export RUBRIC_EVOLVE_EXPERIMENT_FULL=1
export RUBRIC_EVOLVE_ALLOWED_PROFILES=ordinary_harm,adaptive_harm
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES=outcome_safety_rule
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=1
export RUBRIC_EVOLVE_GRADUAL_INITIAL_WEIGHT=0.25
export RUBRIC_EVOLVE_GRADUAL_WARMUP_STEPS=120
export RUBRIC_EVOLVE_ROLLBACK_MIN_SUPPORT=4
export RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP=1
export RUBRIC_EVOLVE_MIN_GAP=0.50
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE=0.875
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE=0.125
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS=1
exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
