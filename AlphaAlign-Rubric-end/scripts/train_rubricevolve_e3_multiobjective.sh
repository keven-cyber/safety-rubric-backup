#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# E3: separate safety and anti-over-refusal banks; held-out contrasts remain
# profile-specific, so a benign rule cannot be promoted from harmful examples.
export TRAIN_EXPERIMENT_NAME=rubricevolve-e3-multiobjective
export RUBRIC_TARGETED_EXPERIMENT_ID=e3_multiobjective
export RUBRIC_EVOLVE_EXPERIMENT_FULL=1
export RUBRIC_EVOLVE_ALLOWED_PROFILES=ordinary_harm,adaptive_harm,benign,sensitive_benign
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES=outcome_safety_rule,outcome_useful_rule,refusal_rule
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE=0
export RUBRIC_EVOLVE_CROSS_PROFILE_GUARD=1
export RUBRIC_EVOLVE_CROSS_PROFILE_MIN_GOOD=4
export RUBRIC_EVOLVE_CROSS_PROFILE_MAX_GOOD=8
export RUBRIC_EVOLVE_CROSS_PROFILE_MIN_PASS_RATE=0.80
export RUBRIC_EVOLVE_MAX_PROFILES_PER_STEP=4
export RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP=1
export RUBRIC_EVOLVE_MIN_GAP=0.45
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE=0.8
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE=0.2
exec bash "${SCRIPT_DIR}/train_online_rubric_evolve.sh" "$@"
