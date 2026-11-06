#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arm="${1:?Usage: train_rubricevolve_targeted_refusal.sh g1|g3}"
shift
case "${arm}" in
  g1) baseline=train_rubricevolve_g1_control.sh ;;
  g3) baseline=train_rubricevolve_g3_stdfloor.sh ;;
  *) echo "Unknown historical arm: ${arm}" >&2; exit 2 ;;
esac
# Explicit isolation from newer v4/v5 training and checkpoint-selection flags.
export RUBRIC_BENIGN_CONTRAST_ENABLE=0
export RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE=0
export RUBRIC_SELECTION_V2_ENABLE=0
export RUBRIC_SELECTION_ORDER_V5_ENABLE=0
export RUBRIC_SELECTION_STRICT_BEST=0
export RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE=0
export RUBRIC_EVIDENCE_V5_PREFLIGHT=0
export RUBRIC_EVOLVE_STD_FLOOR_SCOPE=global
export TRAIN_BEST_METRIC_KEY=val-core/balanced/constraint_rank_score
export RUBRIC_JUDGE_DROP_FAILED=0
if [ "${RUBRIC_GRADED_IMPACT_ENABLE:-0}" = "1" ]; then
  # Keep historical runs unchanged; the new arm archives and quarantines whole
  # prompt groups on unknown scores, never substitutes a semantic verdict.
  export RUBRIC_JUDGE_DROP_FAILED=1
fi
export RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE=1
export RUBRIC_TARGETED_REFUSAL_ENABLE=1
export RUBRIC_TARGETED_REFUSAL_ADV_CAP=0.05
export RUBRIC_TARGETED_REFUSAL_START_STEP=360
export RUBRIC_TARGETED_REFUSAL_INTERVAL=4
exec bash "${SCRIPT_DIR}/${baseline}" "$@"
