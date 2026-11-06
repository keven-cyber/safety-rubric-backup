#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Highest-confidence arm: repair only residual harmful-policy blind spots.
export TRAIN_EXPERIMENT_NAME="rubricevolve-r1-safety-blindspot"
export RUBRIC_EVOLVE_ALLOWED_PROFILES="ordinary_harm,adaptive_harm"
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES="outcome_safety_rule"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_MIN_GAP=0.40
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE=0.80
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE=0.20
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS=1

exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
