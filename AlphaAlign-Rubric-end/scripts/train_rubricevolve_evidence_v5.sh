#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
arm="${1:?Usage: train_rubricevolve_evidence_v5.sh g1|g3}"
shift
case "${arm}" in
  g1)
    export RUBRIC_EVOLVE_REWARD_MODE=legacy
    export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0
    export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=0
    ;;
  g3)
    export RUBRIC_EVOLVE_REWARD_MODE=bounded_residual
    export RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=0.25
    export RUBRIC_EVOLVE_STD_FLOOR_SCOPE=affected_groups
    export RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=1
    ;;
  *) echo "Unknown arm: ${arm}" >&2; exit 2 ;;
esac
export TRAIN_EXPERIMENT_NAME="rubricevolve-${arm}-evidence-v5"
export RUBRIC_TARGETED_EXPERIMENT_ID="${arm}_evidence_v5"
export RUBRIC_SELECTION_V2_ENABLE=1
export RUBRIC_SELECTION_ORDER_V5_ENABLE=1
export RUBRIC_SELECTION_STRICT_BEST=1
export TRAIN_BEST_METRIC_KEY=val-core/selection_v2/rank_score
export RUBRIC_BENIGN_CONTRAST_ENABLE=1
export RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE=1
export RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE=1
export RUBRIC_EVOLVE_COLLECT_START_STEP=120
export RUBRIC_EVOLVE_COLLECT_INTERVAL_STEPS=20
export RUBRIC_EVOLVE_MIN_DISTINCT_PROMPTS=2
export RUBRIC_EVOLVE_GROUP_RANK_GUARD=0
# Never silently remove a failed Judge sample/group to complete training.
export RUBRIC_JUDGE_DROP_FAILED=0
export RUBRIC_EVIDENCE_V5_PREFLIGHT=1
exec bash "${SCRIPT_DIR}/train_rubricevolve_guarded_residual.sh" "$@"
