#!/usr/bin/env bash
# One invocation = one experiment, not eight jobs sharing a checkpoint directory.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  echo 'Usage: bash scripts/train_rubric_experiment.sh E0..E7 [--dry-run]'
  echo 'E1/E2/E3 use configs/primus_api_auth.json unless SAFETY_PIPELINE_ENV_FILE overrides it.'
  exit 0
fi
arm="$1"
shift
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo '[ERROR] Only --dry-run is accepted after the experiment ID.' >&2
  exit 2
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  exec python3 "${PROJECT_ROOT}/scripts/rubric_experiments.py" "${arm}"
fi
# Closed registry, shell-quoted values: never evaluate API/model output.
task_config_args=(--shell)
if [[ -n "${PRIMUS_JOB_CONFIG_FILE_PATH:-}" ]]; then
  task_config_args+=(--check-task-config "${PRIMUS_JOB_CONFIG_FILE_PATH}")
fi
settings="$(python3 "${PROJECT_ROOT}/scripts/rubric_experiments.py" "${arm}" "${task_config_args[@]}")"
eval "${settings}"
if [[ "${RUBRIC_REVISION_MODE}" != old ]]; then
  export SAFETY_PIPELINE_ENV_FILE="${SAFETY_PIPELINE_ENV_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}"
  [[ -f "${SAFETY_PIPELINE_ENV_FILE}" ]] || { echo '[ERROR] API config not found.' >&2; exit 2; }
fi
if [[ "${RUBRIC_EVIDENCE_INITIALIZED:-0}" == 1 ]]; then
  echo '[ERROR] Start each experiment in a fresh process without a reused evidence run.' >&2
  exit 2
fi
export JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-3}" JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
export JUDGE_GPU_IDS="${JUDGE_GPU_IDS:-0,1,2}"
export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-4,5,6,7}" TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ACTOR_PARAM_OFFLOAD=false ACTOR_OPTIMIZER_OFFLOAD=false
export RUBRIC_DATASET_DIR_NAME=updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1
export SOURCE_DATA="${PROJECT_ROOT}/Evaluation/${RUBRIC_DATASET_DIR_NAME}"
[[ -f "${SOURCE_DATA}/manifest.json" ]] || { echo '[ERROR] Old source dataset missing.' >&2; exit 2; }
source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
python3 "${PROJECT_ROOT}/scripts/rubric_experiments.py" "${arm}" > "${RUBRIC_RUN_DIR}/experiment.json"
if [[ "${RUBRIC_REVISION_MODE}" != old ]]; then
  exec bash "${PROJECT_ROOT}/scripts/train_api_rubric.sh"
fi
export FILE_PATH="${SOURCE_DATA}"
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR:-${RUBRIC_RUN_DIR}/checkpoints}/${arm}/${RUBRIC_RUN_ID}"
exec bash "${PROJECT_ROOT}/train_val.sh"
