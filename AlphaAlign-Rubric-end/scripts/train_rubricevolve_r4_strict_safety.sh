#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Precision-first arm: only a rule with perfect held-out good/bad behavior may
# enter the reward, with a second evidence-guided rewrite before giving up.
export TRAIN_EXPERIMENT_NAME="rubricevolve-r4-strict-safety"
export RUBRIC_EVOLVE_ALLOWED_PROFILES="ordinary_harm,adaptive_harm"
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES="outcome_safety_rule"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_MIN_GAP=0.60
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE=1.00
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE=0.00
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS=2
export RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL=10
export RUBRIC_EVOLVE_REPLAY_MIN_PROPOSAL_PER_LABEL=3

exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
