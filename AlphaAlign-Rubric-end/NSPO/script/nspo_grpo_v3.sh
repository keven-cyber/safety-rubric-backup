#!/bin/bash
# NSPO baseline on the RubricAlign v3 dataset.
#
# NSPO = GRPO on a Qwen3Guard safety reward, with every actor update projected
# into the null space of the MLP input activations measured on a general
# instruction corpus. The reward and the projection are the method; the dataset,
# batch size, rollout count, learning rate, prompt length, epochs and checkpoint
# cadence are aligned with the RubricAlign launcher.
#
# This script runs with NSPO's own verl 0.5 copy, which is not interchangeable
# with the parent repository's verl 0.2 (tensordict <=0.6.2 vs >=0.8.0).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NSPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VERL_ROOT="${NSPO_ROOT}/verl"
# The dataset adapter and the shared Evaluation assets live in the parent repo.
RUBRIC_PROJECT_ROOT="${RUBRIC_PROJECT_ROOT:-$(cd "${NSPO_ROOT}/.." && pwd)}"

export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${NSPO_ROOT}/checkpoints/nspo_v3}}"
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
if [ ! -f "${POLICY_MODEL_PATH}/config.json" ]; then
  echo "[ERROR] Policy model not found at POLICY_MODEL_PATH=${POLICY_MODEL_PATH}" >&2
  exit 1
fi

# =========================
# Guard reward endpoint
# =========================
export NSPO_GUARD_API_BASE="${NSPO_GUARD_API_BASE:-http://127.0.0.1:52001/v1}"
export NSPO_GUARD_MODEL="${NSPO_GUARD_MODEL:-qwen3_guard_gen_8b}"
export NSPO_GUARD_API_KEY="${NSPO_GUARD_API_KEY:-EMPTY}"
export NSPO_GUARD_CONTENT_FORMAT="${NSPO_GUARD_CONTENT_FORMAT:-text}"
export NSPO_GUARD_MAX_WORKERS="${NSPO_GUARD_MAX_WORKERS:-64}"

guard_probe_url="${NSPO_GUARD_API_BASE%/v1}/v1/models"
if ! curl -fsS "${guard_probe_url}" >/dev/null 2>&1; then
  echo "[ERROR] Guard endpoint is not reachable: ${guard_probe_url}" >&2
  echo "[ERROR] Start it first (train_val_nspo.sh does this) or point NSPO_GUARD_API_BASE elsewhere." >&2
  exit 1
fi
if ! curl -fsS "${guard_probe_url}" 2>/dev/null | grep -Fq "\"${NSPO_GUARD_MODEL}\""; then
  echo "[ERROR] Guard endpoint does not serve model '${NSPO_GUARD_MODEL}'." >&2
  echo "[ERROR] Served models: $(curl -fsS "${guard_probe_url}" 2>/dev/null)" >&2
  exit 1
fi

# =========================
# Null-space projection corpus
# =========================
export NSPO_NULLSPACE_ENABLE="${NSPO_NULLSPACE_ENABLE:-1}"
export NSPO_NULLSPACE_MODEL_PATH="${NSPO_NULLSPACE_MODEL_PATH:-${POLICY_MODEL_PATH}}"
if [ "${NSPO_NULLSPACE_ENABLE}" != "0" ]; then
  if [ -z "${NSPO_NULLSPACE_DATASET_DIR:-}" ]; then
    echo "[ERROR] NSPO_NULLSPACE_DATASET_DIR is unset." >&2
    echo "[ERROR] Build it with scripts/prepare_nspo_nullspace_dataset.py, or set NSPO_NULLSPACE_ENABLE=0" >&2
    echo "[ERROR] to run plain GRPO on the same reward (ablation only, not NSPO)." >&2
    exit 1
  fi
  if [ ! -f "${NSPO_NULLSPACE_DATASET_DIR}/train/nullspace.parquet" ]; then
    echo "[ERROR] Null-space corpus incomplete: ${NSPO_NULLSPACE_DATASET_DIR}/train/nullspace.parquet is missing." >&2
    exit 1
  fi
  export NSPO_NULLSPACE_DATASET_DIR
else
  echo "[WARN] NSPO_NULLSPACE_ENABLE=0: the null-space projection is off, so this run is GRPO+Guard, not NSPO."
fi
export NSPO_AMEND_INTERVAL="${NSPO_AMEND_INTERVAL:-5}"

# =========================
# Dataset: RubricAlign v3, baseline prompt form
# =========================
# NSPO's reward is a Qwen3Guard verdict on the response alone, so it imposes no
# output format and the plain variant is used.
BASELINE_PROMPT_SOURCE="${BASELINE_PROMPT_SOURCE:-raw_question}"
BASELINE_VARIANT="plain"
case "${BASELINE_PROMPT_SOURCE}" in
  raw_question|instruction) ;;
  *)
    echo "[ERROR] BASELINE_PROMPT_SOURCE must be raw_question or instruction." >&2
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

BASELINE_DATA_PATH="${BASELINE_DATA_PATH:-${RAY_TMPDIR}/baseline_datasets/${BASELINE_VARIANT}_${BASELINE_PROMPT_SOURCE}}"
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
echo "[INFO] NSPO baseline (GRPO + Qwen3Guard reward + null-space projection)"
echo "[INFO] ============================================="
echo "[INFO] verl root:        ${VERL_ROOT}"
echo "[INFO] source dataset:   ${RUBRIC_SOURCE_DATASET_DIR}"
echo "[INFO] baseline dataset: ${BASELINE_DATA_PATH} (variant=${BASELINE_VARIANT}, prompt_source=${BASELINE_PROMPT_SOURCE})"
echo "[INFO] policy:           ${POLICY_MODEL_PATH}"
echo "[INFO] guard:            ${NSPO_GUARD_API_BASE} (${NSPO_GUARD_MODEL})"
echo "[INFO] null-space:       enable=${NSPO_NULLSPACE_ENABLE}, corpus=${NSPO_NULLSPACE_DATASET_DIR:-<off>}, amend every ${NSPO_AMEND_INTERVAL} steps"
echo "[INFO] CKPT_PATH:        ${CKPT_PATH}"
echo "[INFO] GPUs:             ${CUDA_VISIBLE_DEVICES:-all} (n=${TRAIN_N_GPUS_PER_NODE}, rollout TP=${ROLLOUT_TP_SIZE})"
echo "[INFO] budget:           batch=${TRAIN_BATCH_SIZE}, n=${ROLLOUT_N}, lr=${ACTOR_LR}, epochs=${TRAIN_TOTAL_EPOCHS}, total_steps=${TRAIN_TOTAL_STEPS:-auto}, save_freq=${TRAIN_SAVE_FREQ}"
echo "[INFO] ============================================="

unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

# `python3 -m` from the verl root puts NSPO's verl 0.5 first on sys.path, ahead of
# any verl installed for the parent repository. NSPO_PYTHON_BIN is the interpreter
# that actually satisfies verl 0.5's tensordict>=0.8.0 requirement.
NSPO_PYTHON_BIN="${NSPO_PYTHON_BIN:-python3}"
echo "[INFO] interpreter:      ${NSPO_PYTHON_BIN}"
cd "${VERL_ROOT}"

# A PEP 660 editable install of the parent repository's verl 0.2 registers a
# sys.meta_path finder that outranks the working directory. Training NSPO against
# verl 0.2 would fail in confusing ways or, worse, run without the null-space
# hooks, so the resolved package is verified here.
resolved_verl="$("${NSPO_PYTHON_BIN}" -c 'import os, verl; print(os.path.abspath(verl.__file__))')"
expected_verl_root="${VERL_ROOT}/verl"
case "${resolved_verl}" in
  "${expected_verl_root}"/*)
    echo "[INFO] verl package:     ${resolved_verl}"
    ;;
  *)
    echo "[ERROR] 'import verl' resolved to ${resolved_verl}" >&2
    echo "[ERROR] but NSPO must train with ${expected_verl_root} (verl 0.5)." >&2
    echo "[ERROR] Another verl is shadowing it. Use a dedicated interpreter:" >&2
    echo "[ERROR]   bash train_val_nspo.sh --python /path/to/venv/bin/python" >&2
    exit 1
    ;;
esac

PYTHONUNBUFFERED=1 "${NSPO_PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${NSPO_ROOT}/script/safe_reward.py" \
    custom_reward_function.name=compute_score_batched \
    actor_rollout_ref.rollout.temperature=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.val_batch_size="${BASELINE_VAL_BATCH_SIZE:-304}" \
    data.max_prompt_length="${TRAIN_MAX_PROMPT_LENGTH}" \
    data.max_response_length="${TRAIN_MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=4 \
    data.truncation='error' \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.nccl_timeout=7200 \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding="${NSPO_USE_REMOVE_PADDING:-0}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger="${TRAIN_LOGGER_BACKENDS:-[wandb]}" \
    trainer.project_name="${TRAIN_PROJECT_NAME:-rubric_safety_baselines}" \
    trainer.experiment_name="${TRAIN_EXPERIMENT_NAME:-nspo_grpo_v3}" \
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
