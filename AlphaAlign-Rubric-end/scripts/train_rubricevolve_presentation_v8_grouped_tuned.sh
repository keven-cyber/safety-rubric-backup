#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_GROUPED_TUNING_ENABLE=1
export RUBRIC_GROUPED_TIMING_ENABLE=1
export RUBRIC_GROUPED_API_CONCURRENCY=48
export RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE=0
export RUBRIC_GROUPED_WIRE_FORMAT=object
exec bash "${SCRIPT_DIR}/train_rubricevolve_presentation_v8_grouped.sh" "$@"
