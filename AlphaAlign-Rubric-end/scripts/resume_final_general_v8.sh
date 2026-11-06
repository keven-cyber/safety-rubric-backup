#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_ROOT
export RUBRIC_TARGETED_EXPERIMENT_ID="${1:?Expected g1 or g3}_final_general_resume"
case "$1" in g1|g3) ;; *) exit 2;; esac
export RUBRIC_INTERMEDIATE_RECORD_ENABLE=1 RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING=1
source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
trap 'python3 "${PROJECT_ROOT}/scripts/run_evidence.py" finish --run-dir "${RUBRIC_RUN_DIR}" --exit-code "$?"' EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
exec > >(tee -a "${RUBRIC_RUN_DIR}/launcher.log") 2>&1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
python3 -m verl.utils.final_stage_resume
