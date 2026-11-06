#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
arm="${1:?Usage: train_rubricevolve_agentic_eight.sh f1_g1|f3_g3|p1_earlytarget|p2_klanchor|p3_combined|p4_midfloor}"
shift
baseline="$(python3 "${SCRIPT_DIR}/agentic_eight.py" "${arm}" --baseline)"
export RUBRIC_AGENTIC_EIGHT_ARM="${arm}"
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8_legacy.sh" "${baseline}" "$@"
