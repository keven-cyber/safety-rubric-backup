#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# E1: one high-precision safety rule at a time, on residual harmful failures.
export TRAIN_EXPERIMENT_NAME=rubricevolve-e1-precision
export RUBRIC_TARGETED_EXPERIMENT_ID=e1_precision
export RUBRIC_EVOLVE_EXPERIMENT_FULL=1
export RUBRIC_EVOLVE_ALLOWED_PROFILES=ordinary_harm,adaptive_harm
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES=outcome_safety_rule
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP=1
export RUBRIC_EVOLVE_MIN_GAP=0.50
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE=0.875
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE=0.125
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS=1
exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
