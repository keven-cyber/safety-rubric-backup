#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Optional base env
if [ -f "${PROJECT_ROOT}/verl/base.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/verl/base.sh"
fi

# =========================
# Train/val 基础设置（保留 train_val.sh 关键设置）
# =========================
export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR:-${CKPT_PATH:-${PROJECT_ROOT}/checkpoints/qwen25_3B_instruct}}"
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

mkdir -p "${MODEL_BASE_PATH}/guard" "${MODEL_BASE_PATH}/judge" "${MODEL_BASE_PATH}/policy" \
         "${CKPT_PATH}" "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${RAY_TMPDIR}"

export GUARD_MODEL_PATH="${GUARD_MODEL_PATH:-${MODEL_BASE_PATH}/guard}"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${MODEL_BASE_PATH}/judge}"
export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${MODEL_BASE_PATH}/policy}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_BASE_PATH}/reward_model}"

# =========================
# Reward / format 设置
# =========================
# reward_multiple:
#   0 -> existing paradigm (controlled by RUBRIC_SAFE_PRED_REWARD_MODE + RUBRIC_GUARD_ENABLE)
#   1 -> multiplicative paradigm in rubric_judge.py
#        (Guard supervision forced ON for unsafe-labeled samples)
#   2 -> safety-alignment paradigm in rubric_judge.py
#        Rur = (useful条目分数总和 + reasoning条目分数总和) / (useful条目数 + reasoning条目数)
#        unsafe样本: safe=>safety*Rur, unsafe=>(safety-1)*Rur
#        safe样本: safe且no_refuse=>Rur, 否则=>Rur-1
# RUBRIC_MULTIPLE_FAIL_VALUE:
#   only effective when reward_multiple=1
#   pass multiplier fixed to +1; fail multiplier is this value (e.g., -1 or 0)
export reward_multiple="${reward_multiple:-1}"
export RUBRIC_MULTIPLE_FAIL_VALUE="${RUBRIC_MULTIPLE_FAIL_VALUE:--1}"
export RUBRIC_SAFE_PRED_REWARD_MODE="${RUBRIC_SAFE_PRED_REWARD_MODE:-0}"
export RUBRIC_FORMAT_REQUIRE_THINK="${RUBRIC_FORMAT_REQUIRE_THINK:-1}"
export RUBRIC_LENGTH_PENALTY_WEIGHT="${RUBRIC_LENGTH_PENALTY_WEIGHT:-0.1}"
export RUBRIC_LENGTH_TOKENIZER_PATH="${RUBRIC_LENGTH_TOKENIZER_PATH:-${JUDGE_MODEL_PATH}}"
# 全局 reasoning rubric（可由环境变量控制）
export GLOBAL_REASONING_RULE_ENABLE="${GLOBAL_REASONING_RULE_ENABLE:-1}"
export GLOBAL_REASONING_RULE_TEXT="${GLOBAL_REASONING_RULE_TEXT:-The reasoning process must include a concise final answer}"
export GLOBAL_REASONING_RULE_WEIGHT="${GLOBAL_REASONING_RULE_WEIGHT:-2}"

if [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" = "1" ]; then
  # mode=1: <think><prediction><answer>
  export RUBRIC_FORMAT_REQUIRE_PREDICTION_TAG="${RUBRIC_FORMAT_REQUIRE_PREDICTION_TAG:-1}"
  export RUBRIC_FORMAT_WEIGHT="${RUBRIC_FORMAT_WEIGHT:-0.3}"
else
  # mode=0: <think><answer>
  export RUBRIC_FORMAT_REQUIRE_PREDICTION_TAG="${RUBRIC_FORMAT_REQUIRE_PREDICTION_TAG:-0}"
  export RUBRIC_FORMAT_WEIGHT="${RUBRIC_FORMAT_WEIGHT:-0.1}"
fi

# Guard 四象限奖励（可外部覆盖）
export RUBRIC_SAFE_PRED_REWARD_SAFE_SAFE="${RUBRIC_SAFE_PRED_REWARD_SAFE_SAFE:-0.0}"
export RUBRIC_SAFE_PRED_REWARD_SAFE_UNSAFE="${RUBRIC_SAFE_PRED_REWARD_SAFE_UNSAFE:--0.5}"
export RUBRIC_SAFE_PRED_REWARD_UNSAFE_SAFE="${RUBRIC_SAFE_PRED_REWARD_UNSAFE_SAFE:-0.5}"
export RUBRIC_SAFE_PRED_REWARD_UNSAFE_UNSAFE="${RUBRIC_SAFE_PRED_REWARD_UNSAFE_UNSAFE:--0.25}"

# Rubric evolve（默认关闭，开启时从指定 epoch 生效）
export RUBRIC_EVOLVE="${RUBRIC_EVOLVE:-0}"
export RUBRIC_EVOLVE_START_EPOCH="${RUBRIC_EVOLVE_START_EPOCH:-2}"
export RUBRIC_EVOLVE_DIRECT="${RUBRIC_EVOLVE_DIRECT:-1}"
export RUBRIC_EVOLVE_PRINT="${RUBRIC_EVOLVE_PRINT:-1}"
export RUBRIC_EVOLVE_TRAJ_MAX_TOKENS="${RUBRIC_EVOLVE_TRAJ_MAX_TOKENS:-400}"
export RUBRIC_EVOLVE_TOKENIZER_PATH="${RUBRIC_EVOLVE_TOKENIZER_PATH:-${JUDGE_MODEL_PATH}}"

# =========================
# Judge / Guard API 设置（默认适配 Qwen/Qwen3Guard-Gen-8B）
# =========================
export RUBRIC_JUDGE_BACKEND="${RUBRIC_JUDGE_BACKEND:-local_api}"
export RUBRIC_JUDGE_LOCAL_API_BASE="${RUBRIC_JUDGE_LOCAL_API_BASE:-http://127.0.0.1:8000/v1}"
export RUBRIC_JUDGE_LOCAL_API_KEY="${RUBRIC_JUDGE_LOCAL_API_KEY:-EMPTY}"
export RUBRIC_JUDGE_LOCAL_API_MODEL="${RUBRIC_JUDGE_LOCAL_API_MODEL:-qwen3_32B_instruct}"
export RUBRIC_JUDGE_VLLM_MAX_MODEL_LEN="${RUBRIC_JUDGE_VLLM_MAX_MODEL_LEN:-4096}"
export RUBRIC_JUDGE_MAX_NEW_TOKENS="${RUBRIC_JUDGE_MAX_NEW_TOKENS:-512}"
export RUBRIC_JUDGE_API_CONCURRENCY="${RUBRIC_JUDGE_API_CONCURRENCY:-8}"
export RUBRIC_JUDGE_DEBUG="${RUBRIC_JUDGE_DEBUG:-0}"
export RUBRIC_JUDGE_DEBUG_EVERY="${RUBRIC_JUDGE_DEBUG_EVERY:-50}"
export RUBRIC_JUDGE_API_MAX_RETRIES="${RUBRIC_JUDGE_API_MAX_RETRIES:-3}"
export RUBRIC_JUDGE_API_RETRY_BACKOFF="${RUBRIC_JUDGE_API_RETRY_BACKOFF:-1.0}"
export RUBRIC_JUDGE_API_RETRY_BACKOFF_MAX="${RUBRIC_JUDGE_API_RETRY_BACKOFF_MAX:-10.0}"

export RUBRIC_GUARD_ENABLE="${RUBRIC_GUARD_ENABLE:-${RUBRIC_LLAMAGUARD_ENABLE:-1}}"
export RUBRIC_GUARD_BATCH_SIZE="${RUBRIC_GUARD_BATCH_SIZE:-${RUBRIC_LLAMAGUARD_BATCH_SIZE:-8}}"
export RUBRIC_GUARD_MAX_NEW_TOKENS="${RUBRIC_GUARD_MAX_NEW_TOKENS:-${RUBRIC_LLAMAGUARD_MAX_NEW_TOKENS:-32}}"
export RUBRIC_GUARD_BACKEND="${RUBRIC_GUARD_BACKEND:-${RUBRIC_LLAMAGUARD_BACKEND:-local_api}}"
export RUBRIC_GUARD_LOCAL_API_BASE="${RUBRIC_GUARD_LOCAL_API_BASE:-${RUBRIC_LLAMAGUARD_LOCAL_API_BASE:-http://127.0.0.1:8001/v1}}"
export RUBRIC_GUARD_LOCAL_API_KEY="${RUBRIC_GUARD_LOCAL_API_KEY:-${RUBRIC_LLAMAGUARD_LOCAL_API_KEY:-EMPTY}}"
export RUBRIC_GUARD_LOCAL_API_MODEL="${RUBRIC_GUARD_LOCAL_API_MODEL:-${RUBRIC_LLAMAGUARD_LOCAL_API_MODEL:-qwen3_guard_gen_8b}}"

# backward-compatible aliases (rubric_judge.py supports both key sets)
export RUBRIC_LLAMAGUARD_ENABLE="${RUBRIC_LLAMAGUARD_ENABLE:-${RUBRIC_GUARD_ENABLE}}"
export RUBRIC_LLAMAGUARD_BATCH_SIZE="${RUBRIC_LLAMAGUARD_BATCH_SIZE:-${RUBRIC_GUARD_BATCH_SIZE}}"
export RUBRIC_LLAMAGUARD_MAX_NEW_TOKENS="${RUBRIC_LLAMAGUARD_MAX_NEW_TOKENS:-${RUBRIC_GUARD_MAX_NEW_TOKENS}}"
export RUBRIC_LLAMAGUARD_BACKEND="${RUBRIC_LLAMAGUARD_BACKEND:-${RUBRIC_GUARD_BACKEND}}"
export RUBRIC_LLAMAGUARD_LOCAL_API_BASE="${RUBRIC_LLAMAGUARD_LOCAL_API_BASE:-${RUBRIC_GUARD_LOCAL_API_BASE}}"
export RUBRIC_LLAMAGUARD_LOCAL_API_KEY="${RUBRIC_LLAMAGUARD_LOCAL_API_KEY:-${RUBRIC_GUARD_LOCAL_API_KEY}}"
export RUBRIC_LLAMAGUARD_LOCAL_API_MODEL="${RUBRIC_LLAMAGUARD_LOCAL_API_MODEL:-${RUBRIC_GUARD_LOCAL_API_MODEL}}"

export WANDB_LOG_TRAIN_METRICS="${WANDB_LOG_TRAIN_METRICS:-1}"
export REWARD_NUM_EXAMINE="${REWARD_NUM_EXAMINE:-1}"

# =========================
# 数据集按 mode 自动切换
# =========================
if [ -z "${FILE_PATH:-}" ]; then
  if [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" = "1" ]; then
    FILE_PATH="${PROJECT_ROOT}/Evaluation/Rubric_pipline_with_prediction_new"
  else
    FILE_PATH="${PROJECT_ROOT}/Evaluation/Rubric_pipline_without_prediction"
  fi
fi

if [ ! -d "${FILE_PATH}" ]; then
  echo "[ERROR] FILE_PATH not found: ${FILE_PATH}" >&2
  exit 1
fi

export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-$FILE_PATH}"

echo "[INFO] reward_multiple=${reward_multiple}, RUBRIC_MULTIPLE_FAIL_VALUE=${RUBRIC_MULTIPLE_FAIL_VALUE}, RUBRIC_SAFE_PRED_REWARD_MODE=${RUBRIC_SAFE_PRED_REWARD_MODE}, RUBRIC_LENGTH_PENALTY_WEIGHT=${RUBRIC_LENGTH_PENALTY_WEIGHT}, FILE_PATH=${FILE_PATH}, RUBRIC_EVOLVE=${RUBRIC_EVOLVE}"
echo "[INFO] POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "[INFO] JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}"
echo "[INFO] GUARD_MODEL_PATH=${GUARD_MODEL_PATH}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"

benign_train_path="$FILE_PATH/benign_sampled.parquet"
harm_train_path="$FILE_PATH/harm_sampled.parquet"
enhanced_harm_train_path="$FILE_PATH/enhanced_harm_sampled.parquet"
sensitive_benign_train_path="$FILE_PATH/sensitive_sampled.parquet"
xstest_benign_train_path="$FILE_PATH/xstest_benign.parquet"
xtest_harm_train_path="$FILE_PATH/xtest_harm.parquet"

benign_val_path="$FILE_PATH/benign_val.parquet"
harm_val_path="$FILE_PATH/harm_val.parquet"
enhanced_harm_val_path="$FILE_PATH/enhanced_harm_val.parquet"
sensitive_benign_val_path="$FILE_PATH/sensitive_val.parquet"

train_files="['$benign_train_path','$harm_train_path','$enhanced_harm_train_path','$sensitive_benign_train_path','$xstest_benign_train_path','$xtest_harm_train_path']"
test_files="['$benign_val_path','$harm_val_path','$enhanced_harm_val_path','$sensitive_benign_val_path']"

unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

if [ -z "${TRAIN_N_GPUS_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TRAIN_N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  else
    TRAIN_N_GPUS_PER_NODE=1
  fi
fi
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${TRAIN_N_GPUS_PER_NODE}}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${PROJECT_ROOT}/verl/utils/reward_score/rubric_judge.py" \
    custom_reward_function.name=compute_score_batch \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.temperature=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    data.return_raw_chat=True \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    trainer.logger=['wandb'] \
    trainer.project_name="${TRAIN_PROJECT_NAME:-rubric_safety_2}" \
    trainer.experiment_name="${TRAIN_EXPERIMENT_NAME:-qwen3B_instruct_GRPO}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${TRAIN_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=40 \
    trainer.total_epochs=2 \
    trainer.log_val_generations=10 \
    trainer.max_actor_ckpt_to_keep=-1 \
    trainer.default_local_dir="${CKPT_PATH}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    reward_model.enable=False \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.use_remove_padding=True \
    trainer.save_best_ckpt="${TRAIN_SAVE_BEST_CKPT:-True}" \
    trainer.best_metric_key="${TRAIN_BEST_METRIC_KEY:-val-core/harm/rubric_plus_unsafe}" \
    trainer.best_metric_mode="${TRAIN_BEST_METRIC_MODE:-max}"
