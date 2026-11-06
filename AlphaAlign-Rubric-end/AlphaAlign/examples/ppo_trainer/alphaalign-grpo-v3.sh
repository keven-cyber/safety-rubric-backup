#!/bin/bash
# AlphaAlign baseline on the RubricAlign v3 dataset.
#
# The reward is the FsfairX preference model; the algorithm is GRPO, matching
# examples/ppo_trainer/qwen3B-instruct-GRPO.sh. Everything that is not part of
# the method itself (dataset, batch size, rollout count, learning rate, prompt
# length, epochs, checkpoint cadence) is aligned with the RubricAlign launcher so
# only the reward and the optimisation constraint differ.
#
# This script runs with AlphaAlign's own verl copy: `python3 -m` puts the
# AlphaAlign root first on sys.path.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# The dataset adapter and the shared Evaluation assets live in the parent repo.
RUBRIC_PROJECT_ROOT="${RUBRIC_PROJECT_ROOT:-$(cd "${PROJECT_ROOT}/.." && pwd)}"

export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/alphaalign_v3}}"
export API_LOG_DIR="${API_LOG_DIR:-/root/logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/eval_outputs}"

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_temp}"
mkdir -p "${RAY_TMPDIR}"
if [ "${CLEAR_RAY_TMPDIR:-1}" = "1" ]; then
  rm -rf "${RAY_TMPDIR:?}"/*
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export RAY_gcs_server_request_timeout_seconds="${RAY_gcs_server_request_timeout_seconds:-300}"

mkdir -p "${CKPT_PATH}" "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}"

export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${MODEL_BASE_PATH}/policy}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_BASE_PATH}/reward_model}"

if [ ! -f "${POLICY_MODEL_PATH}/config.json" ]; then
  echo "[ERROR] Policy model not found at POLICY_MODEL_PATH=${POLICY_MODEL_PATH}" >&2
  exit 1
fi
if [ ! -f "${REWARD_MODEL_PATH}/config.json" ]; then
  echo "[ERROR] FsfairX reward model not found at REWARD_MODEL_PATH=${REWARD_MODEL_PATH}" >&2
  echo "[ERROR] train_val_alphaalign.sh downloads it; set REWARD_MODEL_ID to change the source." >&2
  exit 1
fi

# =========================
# Dataset: RubricAlign v3, AlphaAlign prompt/label form
# =========================
# AlphaAlign's reward function extracts \boxed{} from <answer> tags, scores a
# <safety_reasoning> format term, and branches on safe::/unsafe:: ground truth.
# The adapter therefore emits the AlphaAlign variant, not a bare prompt.
BASELINE_PROMPT_SOURCE="${BASELINE_PROMPT_SOURCE:-raw_question}"
BASELINE_VARIANT="alphaalign"
case "${BASELINE_PROMPT_SOURCE}" in
  raw_question) ;;
  *)
    echo "[ERROR] AlphaAlign requires BASELINE_PROMPT_SOURCE=raw_question; its own instruction" >&2
    echo "[ERROR] template replaces the RubricAlign one and its reward depends on that template." >&2
    exit 2
    ;;
esac

# Use the completed enriched corpus for comparable RubricAlign baselines. A
# fallback is opt-in so a missing upload cannot silently change the train set.
RUBRIC_DATASET_DIR_NAME="${RUBRIC_DATASET_DIR_NAME:-updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1}"
RUBRIC_DATASET_FALLBACK_DIR_NAME="${RUBRIC_DATASET_FALLBACK_DIR_NAME:-}"
if [ -z "${RUBRIC_SOURCE_DATASET_DIR:-}" ]; then
  dataset_candidates=()
  for dataset_name in "${RUBRIC_DATASET_DIR_NAME}" "${RUBRIC_DATASET_FALLBACK_DIR_NAME}"; do
    [ -n "${dataset_name}" ] || continue
    if [ -n "${RUBRIC_DATASET_ROOT:-}" ]; then
      dataset_candidates+=("${RUBRIC_DATASET_ROOT}/${dataset_name}")
    fi
    dataset_candidates+=("${RUBRIC_PROJECT_ROOT}/Evaluation/${dataset_name}")
    if [ -n "${PRIMUS_OSS_WORKSPACE_DIR:-}" ]; then
      dataset_candidates+=(
        "${PRIMUS_OSS_WORKSPACE_DIR}/datasets/${dataset_name}"
        "${PRIMUS_OSS_WORKSPACE_DIR}/data/${dataset_name}"
        "${PRIMUS_OSS_WORKSPACE_DIR}/${dataset_name}"
      )
    fi
  done
  for candidate in "${dataset_candidates[@]}"; do
    if [ -d "${candidate}/train" ] && [ -d "${candidate}/val" ]; then
      RUBRIC_SOURCE_DATASET_DIR="${candidate}"
      break
    fi
  done
  RUBRIC_SOURCE_DATASET_DIR="${RUBRIC_SOURCE_DATASET_DIR:-${dataset_candidates[0]}}"
fi

if [ ! -d "${RUBRIC_SOURCE_DATASET_DIR}/train" ]; then
  echo "[ERROR] RubricAlign v3 dataset not found: ${RUBRIC_SOURCE_DATASET_DIR}" >&2
  echo "[ERROR] Set RUBRIC_DATASET_ROOT to the mounted dataset parent, or RUBRIC_SOURCE_DATASET_DIR directly." >&2
  exit 1
fi

BASELINE_DATA_PATH="${BASELINE_DATA_PATH:-${RAY_TMPDIR}/baseline_datasets/${BASELINE_VARIANT}}"
if [ "${BASELINE_DATA_REBUILD:-1}" = "1" ] || [ ! -f "${BASELINE_DATA_PATH}/manifest.json" ]; then
  python3 "${RUBRIC_PROJECT_ROOT}/scripts/prepare_baseline_prompt_dataset.py" \
    --source-dir "${RUBRIC_SOURCE_DATASET_DIR}" \
    --output-dir "${BASELINE_DATA_PATH}" \
    --prompt-source "${BASELINE_PROMPT_SOURCE}" \
    --variant "${BASELINE_VARIANT}"
fi

TRAIN_DATA_PATH="${BASELINE_DATA_PATH}/train"
VAL_DATA_PATH="${BASELINE_DATA_PATH}/val"
train_files="['${TRAIN_DATA_PATH}/benign_sampled.parquet','${TRAIN_DATA_PATH}/sensitive_sampled.parquet','${TRAIN_DATA_PATH}/harm_sampled.parquet','${TRAIN_DATA_PATH}/adaptive_harm.parquet']"
test_files="['${VAL_DATA_PATH}/benign_sampled.parquet','${VAL_DATA_PATH}/sensitive_sampled.parquet','${VAL_DATA_PATH}/harm_sampled.parquet','${VAL_DATA_PATH}/adaptive_harm.parquet']"

# =========================
# Shared training budget (kept identical to the RubricAlign launcher)
# =========================
TRAIN_MAX_PROMPT_LENGTH="${TRAIN_MAX_PROMPT_LENGTH:-1536}"
TRAIN_MAX_RESPONSE_LENGTH="${TRAIN_MAX_RESPONSE_LENGTH:-2048}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
TRAIN_TOTAL_EPOCHS="${TRAIN_TOTAL_EPOCHS:-3}"
TRAIN_TOTAL_STEPS="${TRAIN_TOTAL_STEPS:-}"
TRAIN_SAVE_FREQ="${TRAIN_SAVE_FREQ:-30}"
TRAIN_TEST_FREQ="${TRAIN_TEST_FREQ:-40}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"

for name in TRAIN_TOTAL_EPOCHS TRAIN_SAVE_FREQ TRAIN_TEST_FREQ TRAIN_BATCH_SIZE ROLLOUT_N; do
  value="${!name}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
done
if [ -n "${TRAIN_TOTAL_STEPS}" ] && ! [[ "${TRAIN_TOTAL_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TRAIN_TOTAL_STEPS must be a positive integer when set, got '${TRAIN_TOTAL_STEPS}'." >&2
  exit 2
fi

trainer_step_override=()
if [ -n "${TRAIN_TOTAL_STEPS}" ]; then
  trainer_step_override+=("trainer.total_training_steps=${TRAIN_TOTAL_STEPS}")
fi

if [ -z "${TRAIN_N_GPUS_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TRAIN_N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  else
    TRAIN_N_GPUS_PER_NODE=1
  fi
fi
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${TRAIN_N_GPUS_PER_NODE}}"

echo "[INFO] ============================================="
echo "[INFO] AlphaAlign baseline (GRPO + FsfairX RM)"
echo "[INFO] ============================================="
echo "[INFO] verl root:        ${PROJECT_ROOT}"
echo "[INFO] source dataset:   ${RUBRIC_SOURCE_DATASET_DIR}"
echo "[INFO] baseline dataset: ${BASELINE_DATA_PATH} (variant=${BASELINE_VARIANT}, prompt_source=${BASELINE_PROMPT_SOURCE})"
echo "[INFO] policy:           ${POLICY_MODEL_PATH}"
echo "[INFO] reward model:     ${REWARD_MODEL_PATH}"
echo "[INFO] CKPT_PATH:        ${CKPT_PATH}"
echo "[INFO] GPUs:             ${CUDA_VISIBLE_DEVICES:-all} (n=${TRAIN_N_GPUS_PER_NODE}, rollout TP=${ROLLOUT_TP_SIZE})"
echo "[INFO] budget:           batch=${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}, lr=${ACTOR_LR}, epochs=${TRAIN_TOTAL_EPOCHS}, total_steps=${TRAIN_TOTAL_STEPS:-auto}, save_freq=${TRAIN_SAVE_FREQ}"
echo "[INFO] ============================================="

unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

cd "${PROJECT_ROOT}"

# Running from the AlphaAlign root normally puts its own verl first on sys.path,
# but a PEP 660 editable install of the parent repository registers a
# sys.meta_path finder that outranks the working directory. Silently training the
# wrong verl would be indistinguishable from a successful baseline run, so the
# resolved package is verified here.
resolved_verl="$(python3 -c 'import os, verl; print(os.path.abspath(verl.__file__))')"
expected_verl_root="${PROJECT_ROOT}/verl"
case "${resolved_verl}" in
  "${expected_verl_root}"/*)
    echo "[INFO] verl package:     ${resolved_verl}"
    ;;
  *)
    echo "[ERROR] 'import verl' resolved to ${resolved_verl}" >&2
    echo "[ERROR] but AlphaAlign must train with ${expected_verl_root}." >&2
    echo "[ERROR] A PEP 660 editable install of another verl is shadowing it. Either" >&2
    echo "[ERROR]   pip uninstall verl" >&2
    echo "[ERROR] or run this baseline from a dedicated venv." >&2
    exit 1
    ;;
esac

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.temperature=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.val_batch_size="${BASELINE_VAL_BATCH_SIZE:-304}" \
    data.max_prompt_length="${TRAIN_MAX_PROMPT_LENGTH}" \
    data.max_response_length="${TRAIN_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=4 \
    data.return_raw_chat=True \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    reward_model.enable=True \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.input_tokenizer="${POLICY_MODEL_PATH}" \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu="${REWARD_MODEL_MICRO_BATCH_SIZE_PER_GPU:-4}" \
    reward_model.model.use_remove_padding=False \
    trainer.logger="${TRAIN_LOGGER_BACKENDS:-[wandb]}" \
    trainer.project_name="${TRAIN_PROJECT_NAME:-rubric_safety_baselines}" \
    trainer.experiment_name="${TRAIN_EXPERIMENT_NAME:-alphaalign_grpo_v3}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${TRAIN_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq="${TRAIN_SAVE_FREQ}" \
    trainer.test_freq="${TRAIN_TEST_FREQ}" \
    trainer.total_epochs="${TRAIN_TOTAL_EPOCHS}" \
    trainer.resume_mode="${TRAIN_RESUME_MODE:-auto}" \
    "${trainer_step_override[@]}" \
    trainer.log_val_generations="${BASELINE_LOG_VAL_GENERATIONS:-10}" \
    trainer.max_actor_ckpt_to_keep=-1 \
    trainer.default_local_dir="${CKPT_PATH}"
