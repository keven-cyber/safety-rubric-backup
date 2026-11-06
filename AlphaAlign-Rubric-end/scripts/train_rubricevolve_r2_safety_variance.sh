#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Same residual-safety evolution with bounded 8-rollout variance calibration.
export TRAIN_EXPERIMENT_NAME="rubricevolve-r2-safety-variance"
export RUBRIC_EVOLVE_ALLOWED_PROFILES="ordinary_harm,adaptive_harm"
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES="outcome_safety_rule"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=1
export RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS=8
export RUBRIC_EVOLVE_VARIANCE_ALPHA=0.20
export RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA=0.40
export RUBRIC_EVOLVE_VARIANCE_EMA_DECAY=0.90
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN=0.75
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX=1.25
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS=1

exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
