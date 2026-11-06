#!/usr/bin/env bash
# One invocation builds one immutable dataset revision and trains one arm.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
  echo 'Usage: bash scripts/train_targeted_rubric_experiment.sh A|B|C|D1|D2|E [--dry-run]'
  exit 0
fi
arm="$1"
shift
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--dry-run" ) ]]; then
  echo '[ERROR] Only --dry-run is accepted after the experiment ID.' >&2
  exit 2
fi
if [[ "${1:-}" == "--dry-run" ]]; then
  exec python3 "${PROJECT_ROOT}/scripts/rubric_targeted_experiments.py" "${arm}"
fi

settings="$(python3 "${PROJECT_ROOT}/scripts/rubric_targeted_experiments.py" "${arm}" --shell)"
eval "${settings}"
export SAFETY_PIPELINE_ENV_FILE="${SAFETY_PIPELINE_ENV_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}"
[[ -f "${SAFETY_PIPELINE_ENV_FILE}" ]] || { echo '[ERROR] API config not found.' >&2; exit 2; }
if [[ "${RUBRIC_EVIDENCE_INITIALIZED:-0}" == 1 ]]; then
  echo '[ERROR] Start each experiment in a fresh process.' >&2
  exit 2
fi

export JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-3}" JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
export JUDGE_GPU_IDS="${JUDGE_GPU_IDS:-0,1,2}"
export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-4,5,6,7}" TRAIN_N_GPUS_PER_NODE="${TRAIN_N_GPUS_PER_NODE:-4}"
export ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-1}"
export ACTOR_PARAM_OFFLOAD=false ACTOR_OPTIMIZER_OFFLOAD=false
export RUBRIC_DATASET_DIR_NAME=updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1
export SOURCE_DATA="${PROJECT_ROOT}/Evaluation/${RUBRIC_DATASET_DIR_NAME}"
[[ -f "${SOURCE_DATA}/manifest.json" ]] || { echo '[ERROR] Source dataset missing.' >&2; exit 2; }

source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
python3 "${PROJECT_ROOT}/scripts/rubric_targeted_experiments.py" "${arm}" \
  > "${RUBRIC_RUN_DIR}/experiment.json"
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR:-${RUBRIC_RUN_DIR}/checkpoints}/targeted_${arm}/${RUBRIC_RUN_ID}"
exec bash "${PROJECT_ROOT}/scripts/train_api_rubric.sh"
