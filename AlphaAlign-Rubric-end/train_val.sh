#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Policy 模型可配置参数（最小改动）
POLICY_MODEL_ID="${POLICY_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
POLICY_MODEL_DIR="${POLICY_MODEL_DIR:-}"
SKIP_POLICY_DOWNLOAD="${SKIP_POLICY_DOWNLOAD:-0}"
GUARD_MODEL_ID="${GUARD_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
GUARD_SERVED_MODEL_NAME="${GUARD_SERVED_MODEL_NAME:-qwen3_guard_gen_8b}"

usage() {
  cat <<'USAGE'
Usage: bash train_val.sh [options]

Options:
  -p, --policy-model <model_id>   Policy 模型 ID（用于 modelscope 下载）
      --policy-dir <dir>          Policy 下载目录（默认: $MODEL_BASE_PATH/policy）
      --policy-path <dir>         已有本地 Policy 路径（用于训练）
      --skip-policy-download      跳过 Policy 下载（需提供 --policy-path 或预先存在目录）
  -h, --help                      显示帮助
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--policy-model)
      POLICY_MODEL_ID="${2:?missing value for --policy-model}"
      shift 2
      ;;
    --policy-dir)
      POLICY_MODEL_DIR="${2:?missing value for --policy-dir}"
      shift 2
      ;;
    --policy-path)
      export POLICY_MODEL_PATH="${2:?missing value for --policy-path}"
      shift 2
      ;;
    --skip-policy-download)
      SKIP_POLICY_DOWNLOAD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# 安装依赖
pip install modelscope
pip install --upgrade --force-reinstall modelscope
ray stop --force

# 定义模型和 checkpoint 基础路径
export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
if [ -z "${POLICY_MODEL_DIR}" ]; then
  POLICY_MODEL_DIR="${MODEL_BASE_PATH}/policy"
fi
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR:-${CKPT_PATH:-${VERL_HOME}/checkpoints/qwen25_3B_instruct}}"
export API_LOG_DIR="${API_LOG_DIR:-/root/logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-${VERL_HOME}}/eval_outputs}"
export RUBRIC_EVOLVE_DATASET_DIR="${RUBRIC_EVOLVE_DATASET_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-${VERL_HOME}}/data}"
export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-${RUBRIC_EVOLVE_DATASET_DIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray_temp}"

# 创建必要目录
mkdir -p "${MODEL_BASE_PATH}/guard" "${MODEL_BASE_PATH}/judge" "${POLICY_MODEL_DIR}" \
         "${CKPT_PATH}" "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${RUBRIC_EVOLVE_DATASET_DIR}" "${RAY_TMPDIR}"
rm -rf "${RAY_TMPDIR:?}"/*

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-300}"
export RAY_gcs_server_request_timeout_seconds="${RAY_gcs_server_request_timeout_seconds:-300}"

echo "[INFO] ============================================="
echo "[INFO] Model Base Path: ${MODEL_BASE_PATH}"
echo "[INFO] Checkpoint Path: ${CKPT_PATH}"
echo "[INFO] Policy Path: ${POLICY_MODEL_PATH:-${POLICY_MODEL_DIR}}"
echo "[INFO] Guard Model ID: ${GUARD_MODEL_ID}"
echo "[INFO] Guard Served Name: ${GUARD_SERVED_MODEL_NAME}"
echo "[INFO] Eval Output Dir: ${EVAL_OUTPUT_DIR}"
echo "[INFO] Rubric Evolve Dataset Dir: ${RUBRIC_EVOLVE_DATASET_DIR}"
echo "[INFO] ============================================="

# 下载 Guard/Judge/Policy
modelscope download --model "${GUARD_MODEL_ID}" --local_dir "${MODEL_BASE_PATH}/guard"
modelscope download --model Qwen/Qwen3-32B --local_dir "${MODEL_BASE_PATH}/judge"

if [ "${SKIP_POLICY_DOWNLOAD}" = "1" ]; then
  echo "[INFO] Skip policy download (SKIP_POLICY_DOWNLOAD=1)"
else
  echo "[INFO] Downloading Policy model: ${POLICY_MODEL_ID} ..."
  modelscope download --model "${POLICY_MODEL_ID}" --local_dir "${POLICY_MODEL_DIR}"
fi

export GUARD_MODEL_PATH="${GUARD_MODEL_PATH:-${MODEL_BASE_PATH}/guard}"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${MODEL_BASE_PATH}/judge}"
export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${POLICY_MODEL_DIR}}"
export RUBRIC_GUARD_LOCAL_API_MODEL="${RUBRIC_GUARD_LOCAL_API_MODEL:-${GUARD_SERVED_MODEL_NAME}}"
# backward-compatible alias for older env key names
export RUBRIC_LLAMAGUARD_LOCAL_API_MODEL="${RUBRIC_LLAMAGUARD_LOCAL_API_MODEL:-${RUBRIC_GUARD_LOCAL_API_MODEL}}"

cd "${VERL_HOME}"

# 启动 vLLM Guard 服务
GUARD_CUDA_DEVICES="${GUARD_CUDA_DEVICES:-0,1}"
JUDGE_CUDA_DEVICES="${JUDGE_CUDA_DEVICES:-2,3,4,5}"
TRAIN_CUDA_DEVICES="${TRAIN_CUDA_DEVICES:-6,7}"

export VLLM_LOGGING_LEVEL=WARNING
CUDA_VISIBLE_DEVICES="${GUARD_CUDA_DEVICES}" nohup python -m vllm.entrypoints.openai.api_server \
  --model "${GUARD_MODEL_PATH}" \
  --served-model-name "${GUARD_SERVED_MODEL_NAME}" \
  --tensor-parallel-size 2 \
  --port 8001 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  > "${API_LOG_DIR}/rubric_guard_vllm.log" 2>&1 &

# 启动 vLLM Judge 服务
CUDA_VISIBLE_DEVICES="${JUDGE_CUDA_DEVICES}" nohup python -m vllm.entrypoints.openai.api_server \
  --model "${JUDGE_MODEL_PATH}" \
  --served-model-name qwen3_32B_instruct \
  --tensor-parallel-size 1 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85 \
  --port 8000 \
  > "${API_LOG_DIR}/rubric_judge_vllm.log" 2>&1 &

# 启动 GRPO 训练
CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_DEVICES}" bash examples/ppo_trainer/qwen3B-instruct-GRPO.sh

echo "[INFO] Training finished. Start merging all checkpoints under ${CKPT_PATH} ..."

BASE_MODEL_NAME="${POLICY_MODEL_ID##*/}"
MERGED_MODELS_DIR="${CKPT_PATH}/models"
mkdir -p "${MERGED_MODELS_DIR}"
MERGED_THIS_RUN=()

mapfile -t CKPT_DIRS < <(find "${CKPT_PATH}" -maxdepth 1 -mindepth 1 -type d -name "global_step_*" | sort -V)
if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
  echo "[ERROR] No checkpoint directories found in ${CKPT_PATH} (expected: global_step_*)" >&2
  exit 1
fi

for ckpt_dir in "${CKPT_DIRS[@]}"; do
  step_name="$(basename "${ckpt_dir}")"
  actor_dir="${ckpt_dir}/actor"
  if [ ! -d "${actor_dir}" ]; then
    echo "[ERROR] Missing actor dir for checkpoint: ${ckpt_dir}" >&2
    exit 1
  fi

  target_dir="${MERGED_MODELS_DIR}/${BASE_MODEL_NAME}-${step_name}"
  echo "[INFO] Merging ${step_name} -> ${target_dir}"
  python ./scripts/model_merger.py \
    --backend fsdp \
    --hf_model_path "${POLICY_MODEL_PATH}" \
    --local_dir "${actor_dir}" \
    --target_dir "${target_dir}"

  python ./scripts/move_tokenizer.py \
    --source "${POLICY_MODEL_PATH}" \
    --destination "${target_dir}"

  MERGED_THIS_RUN+=("${target_dir}")
done

echo "[INFO] Merge completed. Start safety evaluation ..."

SAFETY_DIR="${VERL_HOME}/Evaluation/safety"
if [ ! -d "${SAFETY_DIR}" ]; then
  echo "[ERROR] Safety evaluation directory not found: ${SAFETY_DIR}" >&2
  exit 1
fi

cd "${SAFETY_DIR}"

if [ "${#MERGED_THIS_RUN[@]}" -eq 0 ]; then
  echo "[ERROR] No models were merged in current run" >&2
  exit 1
fi

for model_dir in "${MERGED_THIS_RUN[@]}"; do
  model_name="$(basename "${model_dir}")"
  eval_log_file="${EVAL_OUTPUT_DIR}/${model_name}_eval.log"
  eval_result_summary="${EVAL_OUTPUT_DIR}/${model_name}_results_summary.txt"

  {
    echo "========================================"
    echo "Evaluation Results Summary"
    echo "Model: ${model_name}"
    echo "Timestamp: $(date)"
    echo "========================================"
    echo ""
  } > "${eval_result_summary}"

  echo "[INFO] Running generate.py for ${model_name}..."
  VLLM_WORKER_MULTIPROC_METHOD=spawn python generate.py --model_id "${model_dir}" 2>&1 | tee -a "${eval_log_file}" >> "${eval_result_summary}"

  echo "" >> "${eval_result_summary}"
  echo "-------- Evaluation Results --------" >> "${eval_result_summary}"

  echo "[INFO] Running eval_answer.py for ${model_name}..."
  python eval_answer.py --model_id "${model_name}" 2>&1 | tee -a "${eval_log_file}" >> "${eval_result_summary}"

  echo "" >> "${eval_result_summary}"
  echo "========================================" >> "${eval_result_summary}"
  echo "Evaluation completed at: $(date)" >> "${eval_result_summary}"

  echo "[INFO] Logs saved to: ${eval_log_file}"
  echo "[INFO] Summary saved to: ${eval_result_summary}"
done

echo "[INFO] Evaluation phase completed"
