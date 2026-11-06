#!/usr/bin/env bash
set -eo pipefail

# 训练阶段 - 模型下载和设置

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Policy 模型可配置参数（最小改动）
POLICY_MODEL_ID="${POLICY_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
POLICY_MODEL_DIR="${POLICY_MODEL_DIR:-}"
SKIP_POLICY_DOWNLOAD="${SKIP_POLICY_DOWNLOAD:-0}"
export VLLM_USE_CUSTOM_ALLREDUCE=0
GUARD_MODEL_ID="${GUARD_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
GUARD_SERVED_MODEL_NAME="${GUARD_SERVED_MODEL_NAME:-qwen3_guard_gen_8b}"
GUARD_GPU_IDS="${GUARD_GPU_IDS:-${SAFETY_GUARD_GPU_IDS:-3}}"
GUARD_TP_SIZE="${GUARD_TP_SIZE:-${SAFETY_GUARD_TP_SIZE:-1}}"
GUARD_PORT="${GUARD_PORT:-8001}"
FINAL_SAFETY_EVAL_ENABLE="${FINAL_SAFETY_EVAL_ENABLE:-1}"
FINAL_GENERAL_EVAL_ENABLE="${FINAL_GENERAL_EVAL_ENABLE:-1}"
FINAL_GENERAL_EVAL_BACKEND="${FINAL_GENERAL_EVAL_BACKEND:-vllm}"
FINAL_GENERAL_EVAL_DEVICE="${FINAL_GENERAL_EVAL_DEVICE:-cuda:0}"

case "${FINAL_SAFETY_EVAL_ENABLE}" in
  1|true|TRUE|yes|YES|on|ON) FINAL_SAFETY_EVAL_ENABLE=1 ;;
  0|false|FALSE|no|NO|off|OFF) FINAL_SAFETY_EVAL_ENABLE=0 ;;
  *)
    echo "[ERROR] FINAL_SAFETY_EVAL_ENABLE must be a boolean value." >&2
    exit 2
    ;;
esac

case "${FINAL_GENERAL_EVAL_ENABLE}" in
  1|true|TRUE|yes|YES|on|ON) FINAL_GENERAL_EVAL_ENABLE=1 ;;
  0|false|FALSE|no|NO|off|OFF) FINAL_GENERAL_EVAL_ENABLE=0 ;;
  *)
    echo "[ERROR] FINAL_GENERAL_EVAL_ENABLE must be a boolean value." >&2
    exit 2
    ;;
esac
case "${FINAL_GENERAL_EVAL_BACKEND}" in
  hf|vllm) ;;
  *)
    echo "[ERROR] FINAL_GENERAL_EVAL_BACKEND must be hf or vllm." >&2
    exit 2
    ;;
esac

# Semantic reward mode names are shared with qwen3B-instruct-GRPO.sh.
reward_mode_raw="${REWARD_MODE:-${reward_multiple:-guard_safety_alignment}}"
reward_mode_raw="$(tr '[:upper:]-' '[:lower:]_' <<< "${reward_mode_raw}")"
case "${reward_mode_raw}" in
  legacy|0|false|no|off) REWARD_MODE="legacy" ;;
  guard_multiplicative|1|true|yes|on) REWARD_MODE="guard_multiplicative" ;;
  guard_safety_alignment|2) REWARD_MODE="guard_safety_alignment" ;;
  rubric_safety_gated|3) REWARD_MODE="rubric_safety_gated" ;;
  guard_soft_calibrated|4) REWARD_MODE="guard_soft_calibrated" ;;
  rubric_refusal_calibrated|5) REWARD_MODE="rubric_refusal_calibrated" ;;
  rubric_safe_helpful_balanced) REWARD_MODE="rubric_safe_helpful_balanced" ;;
  *)
    echo "[ERROR] Unsupported reward mode: ${reward_mode_raw}" >&2
    exit 2
    ;;
esac
export REWARD_MODE
export reward_multiple="${REWARD_MODE}"

case "${REWARD_MODE}" in
  guard_multiplicative|guard_safety_alignment|guard_soft_calibrated)
    REWARD_MODE_NEEDS_GUARD=1
    ;;
  legacy)
    case "${RUBRIC_GUARD_ENABLE:-1}" in
      1|true|TRUE|yes|YES|on|ON) REWARD_MODE_NEEDS_GUARD=1 ;;
      *) REWARD_MODE_NEEDS_GUARD=0 ;;
    esac
    ;;
  *)
    REWARD_MODE_NEEDS_GUARD=0
    ;;
esac
export REWARD_MODE_NEEDS_GUARD
if [ "${REWARD_MODE_NEEDS_GUARD}" = "0" ]; then
    export RUBRIC_GUARD_ENABLE=0
fi
usage() {
  cat <<'USAGE'
Usage: bash train_val.sh [options]

Options:
  -p, --policy-model <model_id>   Policy 模型 ID（用于 modelscope 下载）
      --policy-dir <dir>          Policy 下载目录（默认: $MODEL_BASE_PATH/policy）
      --policy-path <dir>         已有本地 Policy 路径（用于训练）
      --skip-policy-download      跳过 Policy 下载（需提供 --policy-path 或预先存在目录）
  -h, --help                      显示帮助

Environment:
  TRAIN_PROMPT_VERSION            legacy_compact | structured_steps
  TRAINING_DATA_VERSION           origin | new | new_v2 | new_v3 (default: origin)
  TRAIN_MAX_PROMPT_LENGTH         new_v3 defaults to 1536; other versions default to 1024
  REWARD_MODE                     Semantic reward mode name (preferred)
  reward_multiple                 Alias for REWARD_MODE used by existing jobs
  RUBRIC_JUDGE_GRANULARITY        all_rules (default) | single_rule
  JUDGE_REPLICA_COUNT             Number of local Judge replicas (default: 3)
  JUDGE_TP_SIZE                   Tensor parallel size per Judge replica (default: 1)
  JUDGE_GPU_IDS                   Comma-separated Judge GPU IDs (default: 0,1,2)
  GUARD_GPU_IDS                   Qwen3Guard GPU IDs (default: 3)
  GUARD_TP_SIZE                   Qwen3Guard tensor parallel size (default: 1)
  GUARD_PORT                      Qwen3Guard API port (default: 8001)
  JUDGE_ENABLE_PREFIX_CACHING     Enable vLLM Judge prefix caching (default: 0)
  RUBRIC_JUDGE_LOAD_BALANCE       sticky (default) | round_robin
  RUBRIC_JUDGE_API_LOG_ENABLE     Successful Judge response logs (forced off when intermediate recording is off)
  TRAIN_GPU_IDS                   Comma-separated policy GPU IDs (default: 4,5,6,7)
  ACTOR_PARAM_OFFLOAD             true (default) | false
  ACTOR_OPTIMIZER_OFFLOAD         true (default) | false
  ROLLOUT_TP_SIZE                 Rollout tensor parallel size
  TRAIN_TOTAL_EPOCHS              Training epochs (default: 3)
  TRAIN_SAVE_FREQ                 Checkpoint save interval in steps (default: 40)
  FINAL_EVAL_MIN_STEP             Only merge/final-eval checkpoints at or above this step (default: 0)
  FINAL_SAFETY_EVAL_STEPS         Comma list of exact steps plus last,best (default: 600,900,last,best)
  SAFETY_EVAL_GPU_IDS             One GPU for final safety generation (default: first TRAIN_GPU_IDS GPU)
  FINAL_SAFETY_EVAL_ENABLE        Run final Qwen3Guard safety evaluation (default: 1)
  FINAL_GENERAL_EVAL_ENABLE       Run MMLU/GSM8K on best, falling back to last (default: 1)
  FINAL_GENERAL_EVAL_BACKEND      lm-eval backend: vllm or hf (default: vllm)
  FINAL_GENERAL_EVAL_DEVICE       Device used by the hf backend (default: cuda:0)
  FINAL_GENERAL_EVAL_GPU_IDS      GPUs for general evaluation (default: all released job GPUs)
  FINAL_GENERAL_EVAL_REPLICAS     vLLM model replicas (default: 1; Ray DP is opt-in)
  RUBRIC_INTERMEDIATE_RECORD_ENABLE  Detailed judge/rollout/validation records: 0 (default) | 1 (opt-in)
  ROLLOUT_TRACE_ENABLE            Save training rollout traces as JSONL (forced off by the switch above)
  RUBRIC_ARTIFACT_ROOT            Parent for unique run folders (default: Primus OSS/rubricalign_runs)
  RUBRIC_REQUIRE_OSS              Fail if the Primus mount is missing (default: 0; v4 launcher: 1)
  ROLLOUT_TRACE_MAX_PER_STEP      Soft trace limit preserving whole groups; 0 saves all (default: 0)
  RUBRIC_EVOLVE                   Enable online rubric evolution during policy training (default: 0)
  RUBRIC_EVOLVE_START_STEP        Policy warm-up step before the first proposal
  RUBRIC_EVOLVE_INTERVAL_STEPS     Steps between online rubric proposals
  RUBRIC_EVOLVE_PROVIDER          quota | openai | auto
  RUBRIC_EVOLVE_API_MODEL         Evolution writer model (online launcher: qwen3.7-plus)
  RUBRIC_EVOLVE_MIN_GAP           Minimum good-vs-bad Judge pass-rate gap for promotion
  RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE  Variance-aware delayed rule weighting: 0 (default) | 1
  RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS   Complete rollout-group size required for weighting (default: 8)
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

# Explicit completion of a previously omitted original best, using a fresh
# verified physical recovery copy. Never train, retarget, or use a linked view.
if [ "${RUBRIC_EVAL_USE_RECOVERY_COPY:-0}" = "1" ]; then
  if [ "${RUBRIC_EVAL_ONLY:-0}" != "1" ]; then
    echo '[ERROR] Recovery-copy evaluation must be evaluation-only.' >&2
    exit 2
  fi
  python3 - "${CKPT_PATH}" "${RUBRIC_RECOVERY_SOURCE_CHECKPOINT}" "${RUBRIC_EVAL_STEP}" <<'PYBEST'
import json,sys
from pathlib import Path
from verl.utils.best_checkpoint_reference import resolve
root, source, step = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
report=json.loads((root/'checkpoint_recovery.json').read_text())
if (report.get('status')!='ready' or report.get('restored_step')!=1170
        or report.get('source_checkpoint')!=str(source) or report.get('destination')!=str(root)
        or report.get('source_files_modified') is not False
        or report.get('source_and_copy_stream_sha256_equal') is not True or report.get('links_used') is not False):
    raise ValueError('Missing exact complete1170 physical recovery report')
info=json.loads((root/'best_checkpoint_info.json').read_text())
if info.get('best_metric_step')!=step or resolve(root,info)!=root/f'global_step_{step}':
    raise ValueError('Repair must evaluate the exact preserved original numbered best')
PYBEST
  export RUBRIC_EVAL_SOURCE_DIR="${CKPT_PATH}"
fi
source "${PROJECT_ROOT}/scripts/rubric_eval_mode.sh"
rubric_configure_eval_mode
if [ "${RUBRIC_EVAL_USE_RECOVERY_COPY:-0}" = "1" ]; then
  [ "${RUBRIC_EVAL_ONLY}" = "1" ] || { echo '[ERROR] Repair cannot enter training.' >&2; exit 2; }
  export FINAL_SAFETY_EVAL_STEPS=best
  export FINAL_EVAL_EXACT_TRIO=1
  export FINAL_EVAL_EXPECTED_LAST_STEP=1170
fi
source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
# Smoke validation must always exercise the configured trigger from a fresh
# optimizer step. Primus failover preserves /primus_oss/save_checkpoint, so an
# automatic resume can otherwise start after the trigger (for example at step
# 7 while the smoke trigger is step 2) and fail forever. Keep normal training's
# resume behavior unchanged.
if [ "${RUBRIC_EVOLVE_SMOKE_TEST:-0}" = "1" ]; then
  export TRAIN_RESUME_MODE=disable
  export CKPT_PATH="${RUBRIC_RUN_DIR}/smoke_checkpoints/${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID}"
  echo "[SMOKE] Fresh attempt=${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID}; resume=disable; checkpoint_dir=${CKPT_PATH}"
fi
# Persist a failure status even if dependency installation fails before services start.
trap 'python3 "${PROJECT_ROOT}/scripts/run_evidence.py" finish --run-dir "${RUBRIC_RUN_DIR}" --exit-code "$?"' EXIT
exec > >(tee -a "${RUBRIC_RUN_DIR}/launcher.log") 2>&1

# 安装依赖
python3 -m pip install fire tqdm
TRAIN_NUMPY_PIN="${TRAIN_NUMPY_PIN:-${LM_EVAL_NUMPY_PIN:-1.26.4}}"
# Keep ModelScope dependency resolution from upgrading NumPy and breaking
# binary extensions used later by the general-evaluation Transformers import.
modelscope_requirement=modelscope
if [ -n "${MODEL_SCOPE_PIN:-}" ]; then modelscope_requirement="modelscope==${MODEL_SCOPE_PIN}"; fi
python3 -m pip install --upgrade --force-reinstall \
  "numpy==${TRAIN_NUMPY_PIN}" \
  "${modelscope_requirement}"
# ModelScope dependency resolution may select huggingface-hub 1.x, while the
# image's transformers 4.57.x requires >=0.34.0,<1.0.
python3 -m pip install "huggingface_hub==${HF_HUB_PIN:-0.36.2}"
ray stop --force
# 定义模型和 checkpoint 基础路径，这tm是为什么？？？？？？？？？？
export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
if [ -z "${POLICY_MODEL_DIR}" ]; then
  POLICY_MODEL_DIR="${MODEL_BASE_PATH}/policy"
fi
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/rubricalign}}"
export API_LOG_DIR="${API_LOG_DIR:-/root/logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/eval_outputs}"
# export RAY_TMPDIR="${VERL_HOME}/ray_temp"
if [ -n "${CROSS_MODEL_EXPERIMENT_KEY:-}" ]; then
  # The fresh comparison already allocated a unique per-submission directory.
  # Keep it throughout the chain; no shared temporary directory is cleared.
  : "${RAY_TMPDIR:?Cross-model job requires its isolated Ray directory}"
  mkdir -p "${RAY_TMPDIR}"
else
  export RAY_TMPDIR=/tmp/ray_temp
  mkdir -p /tmp/ray_temp
  rm -rf /tmp/ray_temp/*
fi
unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export RAY_raylet_start_wait_time_s=300
export RAY_gcs_server_request_timeout_seconds=300

# 创建必要的目录结构
mkdir -p "$MODEL_BASE_PATH/guard"
mkdir -p "$MODEL_BASE_PATH/judge"
mkdir -p "$POLICY_MODEL_DIR"
mkdir -p "$CKPT_PATH"
mkdir -p "$API_LOG_DIR"
mkdir -p "$EVAL_OUTPUT_DIR"
mkdir -p "$RAY_TMPDIR"

echo "[INFO] ============================================="
echo "[INFO] 模型下载配置"
echo "[INFO] ============================================="
echo "[INFO] Model Base Path: $MODEL_BASE_PATH"
echo "[INFO] Checkpoint Path: $CKPT_PATH"
echo "[INFO] ============================================="

# 下载 Guard 模型
# echo "[INFO] Downloading Llama-Guard-3-8B..."
# modelscope download --model LLM-Research/Llama-Guard-3-8B --local_dir "$MODEL_BASE_PATH/guard"

if [ "${REWARD_MODE_NEEDS_GUARD}" = "1" ] || [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
  echo "[INFO] Downloading Guard Model for required training/evaluation: ${GUARD_MODEL_ID}..."
  if [ -n "${CROSS_MODEL_EXPERIMENT_KEY:-}" ]; then
    python3 "${PROJECT_ROOT}/scripts/cross_model_dependency_download.py" "${GUARD_MODEL_ID}" "${MODEL_BASE_PATH}/guard" "${CROSS_MODEL_RUN_DIR}/guard_snapshot.json"
  else
    modelscope download --model "${GUARD_MODEL_ID}" --local_dir "${MODEL_BASE_PATH}/guard"
  fi
fi

# 下载 Judge 模型 (Qwen3-32B)
if [ "${RUBRIC_EVAL_ONLY}" != "1" ]; then
  echo "[INFO] Downloading Qwen/Qwen3-32B..."
  if [ -n "${CROSS_MODEL_EXPERIMENT_KEY:-}" ]; then
    python3 "${PROJECT_ROOT}/scripts/cross_model_dependency_download.py" Qwen/Qwen3-32B "${MODEL_BASE_PATH}/judge" "${CROSS_MODEL_RUN_DIR}/judge_snapshot.json"
  else
    modelscope download --model Qwen/Qwen3-32B --local_dir "$MODEL_BASE_PATH/judge"
  fi
fi

# 下载 Policy 模型（可配置）
if [ "${SKIP_POLICY_DOWNLOAD}" = "1" ]; then
    echo "[INFO] Skip policy download (SKIP_POLICY_DOWNLOAD=1)"
else
    echo "[INFO] Downloading Policy model: ${POLICY_MODEL_ID} ..."
    modelscope download --model "${POLICY_MODEL_ID}" --local_dir "${POLICY_MODEL_DIR}"
fi

echo "[INFO] Model download completed"

# 设置环境变量 - 指向下载的模型目录
# modelscope 会直接在指定目录下存放模型文件
export GUARD_MODEL_PATH="$MODEL_BASE_PATH/guard"
export JUDGE_MODEL_PATH="$MODEL_BASE_PATH/judge"
export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${POLICY_MODEL_DIR}}"

# 创建 checkpoint 目录
# mkdir -p "$CKPT_PATH"

echo "[INFO] ================================="
echo "[INFO] 环境变量已设置"
echo "[INFO] ================================="
echo "[INFO] VERL_HOME=$VERL_HOME"
echo "[INFO] MODEL_BASE_PATH=$MODEL_BASE_PATH"
echo "[INFO] GUARD_MODEL_PATH=$GUARD_MODEL_PATH"
echo "[INFO] JUDGE_MODEL_PATH=$JUDGE_MODEL_PATH"
echo "[INFO] POLICY_MODEL_PATH=$POLICY_MODEL_PATH"
echo "[INFO] CKPT_PATH=$CKPT_PATH"
echo "[INFO] EVAL_OUTPUT_DIR=$EVAL_OUTPUT_DIR"
echo "[INFO] API_LOG_DIR=$API_LOG_DIR"
echo "[INFO] REWARD_MODE=$REWARD_MODE"
echo "[INFO] RUBRIC_JUDGE_GRANULARITY=${RUBRIC_JUDGE_GRANULARITY:-all_rules}"
echo "[INFO] TRAINING_DATA_VERSION=${TRAINING_DATA_VERSION:-origin}"
echo "[INFO] TRAIN_PROMPT_VERSION=${TRAIN_PROMPT_VERSION:-structured_steps}"
echo "[INFO] ================================="

# cd "$VERL_HOME"

# 启动 vLLM Guard 服务
echo "[INFO] Configuring vLLM Guard service..."
# bash examples/ppo_trainer/launch_vllm.sh
# 使用环境变量或默认值获取 guard 模型路径
GUARD_MODEL_PATH="${GUARD_MODEL_PATH:-/openbayes/input/input0/models/llamaguard}"
API_LOG_DIR="${API_LOG_DIR:-/root/logs}"

export VLLM_LOGGING_LEVEL=WARNING
guard_service_pid=""
judge_service_pids=()

if ! [[ "${GUARD_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${GUARD_PORT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] GUARD_TP_SIZE and GUARD_PORT must be positive integers." >&2
    exit 2
fi
IFS=',' read -r -a guard_gpu_array <<< "${GUARD_GPU_IDS}"
if [ "${#guard_gpu_array[@]}" -ne "${GUARD_TP_SIZE}" ]; then
    echo "[ERROR] GUARD_GPU_IDS has ${#guard_gpu_array[@]} GPUs; expected ${GUARD_TP_SIZE}." >&2
    exit 2
fi

stop_service_pids() {
    local pid
    for pid in "$@"; do
        if [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
    for pid in "$@"; do
        if [ -n "${pid}" ]; then
            wait "${pid}" 2>/dev/null || true
        fi
    done
}

cleanup_api_services() {
    local exit_code=$?
    stop_service_pids "${judge_service_pids[@]}"
    if [ -n "${guard_service_pid}" ]; then
        stop_service_pids "${guard_service_pid}"
    fi
    python3 "${PROJECT_ROOT}/scripts/run_evidence.py" finish --run-dir "${RUBRIC_RUN_DIR}" --exit-code "${exit_code}" || true
    python3 "${PROJECT_ROOT}/scripts/analyze_training_evidence.py" --run-dir "${RUBRIC_RUN_DIR}" \
      > "${RUBRIC_RUN_DIR}/analysis.log" 2>&1 || true
}
trap cleanup_api_services EXIT

# Evaluation-only runs must not download/start training judges or enter GRPO.
if [ "${RUBRIC_EVAL_ONLY}" != "1" ]; then
# 启动一个或多个 vLLM Judge 服务
echo "[INFO] Configuring vLLM Judge service..."
# bash examples/ppo_trainer/launch_guard.sh
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/openbayes/input/input0/models/qwen3_32B}"
API_LOG_DIR="${API_LOG_DIR:-/root/logs}"

export VLLM_LOGGING_LEVEL=WARNING
JUDGE_REPLICA_COUNT="${JUDGE_REPLICA_COUNT:-3}"
JUDGE_TP_SIZE="${JUDGE_TP_SIZE:-1}"
JUDGE_GPU_IDS="${JUDGE_GPU_IDS:-0,1,2}"
JUDGE_BASE_PORT="${JUDGE_BASE_PORT:-8000}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-16384}"
JUDGE_GPU_MEMORY_UTILIZATION="${JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
JUDGE_ENABLE_PREFIX_CACHING="${JUDGE_ENABLE_PREFIX_CACHING:-0}"

if ! [[ "${JUDGE_REPLICA_COUNT}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${JUDGE_TP_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] JUDGE_REPLICA_COUNT and JUDGE_TP_SIZE must be positive integers." >&2
    exit 2
fi

IFS=',' read -r -a judge_gpu_array <<< "${JUDGE_GPU_IDS}"
required_judge_gpus=$((JUDGE_REPLICA_COUNT * JUDGE_TP_SIZE))
if [ "${#judge_gpu_array[@]}" -ne "${required_judge_gpus}" ]; then
    echo "[ERROR] JUDGE_GPU_IDS has ${#judge_gpu_array[@]} GPUs; expected ${required_judge_gpus} for ${JUDGE_REPLICA_COUNT} replicas x TP ${JUDGE_TP_SIZE}." >&2
    exit 2
fi

judge_guard_overlap=0
for judge_gpu in "${judge_gpu_array[@]}"; do
    for guard_gpu in "${guard_gpu_array[@]}"; do
        if [ "${judge_gpu}" = "${guard_gpu}" ]; then
            judge_guard_overlap=1
        fi
    done
done

if [ "${judge_guard_overlap}" = "1" ] && [ "${REWARD_MODE_NEEDS_GUARD}" = "1" ]; then
    if [ "${JUDGE_TP_SIZE}" -ne 1 ]; then
        echo "[ERROR] Judge and Guard GPU assignments overlap, and automatic Guard GPU reservation is only supported when JUDGE_TP_SIZE=1." >&2
        exit 2
    fi

    filtered_judge_gpu_array=()
    for judge_gpu in "${judge_gpu_array[@]}"; do
        is_guard_gpu=0
        for guard_gpu in "${guard_gpu_array[@]}"; do
            if [ "${judge_gpu}" = "${guard_gpu}" ]; then
                is_guard_gpu=1
                break
            fi
        done
        if [ "${is_guard_gpu}" = "0" ]; then
            filtered_judge_gpu_array+=("${judge_gpu}")
        fi
    done

    if [ "${#filtered_judge_gpu_array[@]}" -eq 0 ]; then
        echo "[ERROR] No Judge GPU remains after reserving GUARD_GPU_IDS=${GUARD_GPU_IDS}." >&2
        exit 2
    fi

    judge_gpu_array=("${filtered_judge_gpu_array[@]}")
    JUDGE_GPU_IDS="$(IFS=,; echo "${judge_gpu_array[*]}")"
    JUDGE_REPLICA_COUNT="${#judge_gpu_array[@]}"
    echo "[INFO] Reserving GPU(s) ${GUARD_GPU_IDS} for Qwen3Guard; adjusted Judge layout to ${JUDGE_REPLICA_COUNT} replicas on GPUs ${JUDGE_GPU_IDS}."
fi
export JUDGE_GPU_IDS JUDGE_REPLICA_COUNT

judge_prefix_args=()
case "${JUDGE_ENABLE_PREFIX_CACHING}" in
    1|true|TRUE|yes|YES|on|ON) judge_prefix_args+=(--enable-prefix-caching) ;;
esac

if [ "${REWARD_MODE_NEEDS_GUARD}" = "1" ]; then
echo "[INFO] Starting Qwen3Guard: GPUs=${GUARD_GPU_IDS}, TP=${GUARD_TP_SIZE}, port=${GUARD_PORT}"
CUDA_VISIBLE_DEVICES="${GUARD_GPU_IDS}" nohup python -m vllm.entrypoints.openai.api_server \
    --model "${GUARD_MODEL_PATH}" \
    --served-model-name "${GUARD_SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${GUARD_TP_SIZE}" \
    --port "${GUARD_PORT}" \
    --dtype bfloat16 \
    --gpu-memory-utilization "${GUARD_GPU_MEMORY_UTILIZATION:-0.8}" \
    > "${API_LOG_DIR}/rubric_guard_vllm.log" 2>&1 &
guard_service_pid=$!
fi

echo "[INFO] Launching ${JUDGE_REPLICA_COUNT} vLLM Judge replicas on GPUs ${JUDGE_GPU_IDS}..."
judge_api_bases=()
judge_ports=()
for ((replica_idx=0; replica_idx<JUDGE_REPLICA_COUNT; replica_idx++)); do
    replica_gpu_ids=()
    for ((tp_idx=0; tp_idx<JUDGE_TP_SIZE; tp_idx++)); do
        gpu_idx=$((replica_idx * JUDGE_TP_SIZE + tp_idx))
        replica_gpu_ids+=("${judge_gpu_array[$gpu_idx]}")
    done
    replica_gpu_csv="$(IFS=,; echo "${replica_gpu_ids[*]}")"
    replica_port=$((JUDGE_BASE_PORT + replica_idx))
    if [ "${JUDGE_BASE_PORT}" -le "${GUARD_PORT}" ] && [ "${replica_port}" -ge "${GUARD_PORT}" ]; then
        replica_port=$((replica_port + 1))
    fi
    judge_ports+=("${replica_port}")
    judge_api_bases+=("http://127.0.0.1:${replica_port}/v1")
    echo "[INFO] Starting Judge replica ${replica_idx}: GPUs=${replica_gpu_csv}, TP=${JUDGE_TP_SIZE}, port=${replica_port}"
    if [ "${RUBRIC_GRADED_IMPACT_ENABLE:-0}" = "1" ]; then
    CUDA_VISIBLE_DEVICES="${replica_gpu_csv}" nohup python "${PROJECT_ROOT}/scripts/judge_replay_server.py" \
        --model "${JUDGE_MODEL_PATH}" --served qwen3_32B_instruct --port "${replica_port}" \
        --manifest "${API_LOG_DIR}/judge_profile_${replica_idx}.json" --training \
        > "${API_LOG_DIR}/rubric_judge_vllm_${replica_idx}.log" 2>&1 &
    else
    CUDA_VISIBLE_DEVICES="${replica_gpu_csv}" nohup python -m vllm.entrypoints.openai.api_server \
        --model "${JUDGE_MODEL_PATH}" \
        --served-model-name qwen3_32B_instruct \
        --tensor-parallel-size "${JUDGE_TP_SIZE}" \
        --max-model-len "${JUDGE_MAX_MODEL_LEN}" \
        --gpu-memory-utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
        --port "${replica_port}" \
        "${judge_prefix_args[@]}" \
        > "${API_LOG_DIR}/rubric_judge_vllm_${replica_idx}.log" 2>&1 &
    fi
    judge_service_pids+=("$!")
done
RUBRIC_JUDGE_LOCAL_API_BASES="$(IFS=,; echo "${judge_api_bases[*]}")"
export RUBRIC_JUDGE_LOCAL_API_BASES
export RUBRIC_JUDGE_SERVICE_PIDS="$(IFS=,; echo "${judge_service_pids[*]}")"
export RUBRIC_JUDGE_SERVICE_LOG_DIR="${API_LOG_DIR}"
export RUBRIC_JUDGE_TOKENIZER_DIR="${JUDGE_MODEL_PATH}"
export RUBRIC_GUARD_LOCAL_API_BASE="http://127.0.0.1:${GUARD_PORT}/v1"
export RUBRIC_GUARD_LOCAL_API_MODEL="${GUARD_SERVED_MODEL_NAME}"
fi

# ... (前面的环境变量设置和 nohup 启动命令) ...

# ==========================================
# 定义等待 API 就绪的函数
# ==========================================
wait_for_api() {
    local port=$1
    local service_name=$2
    local service_pid="${3:-}"
    local max_retries=3000  # 最大重试次数 (约 5-10 分钟，取决于加载速度)
    local retry_count=0
    
    echo "[INFO] Waiting for ${service_name} on port ${port} to start..."
    
    while [ $retry_count -lt $max_retries ]; do
        if [ -n "${service_pid}" ]; then
            local process_state
            process_state="$(ps -p "${service_pid}" -o stat= 2>/dev/null || true)"
            if [[ -z "${process_state//[[:space:]]/}" || "${process_state}" =~ ^[[:space:]]*[ZX] ]]; then
                echo "[ERROR] ${service_name} process ${service_pid} exited before readiness; original service log retained." >&2
                return 1
            fi
        fi
        # 使用 curl 检查端口是否连通且返回 HTTP 200 或正常响应
        # vLLM 的 /health 或 /v1/models 端点通常可用
        if curl --connect-timeout 2 --max-time 5 -s -o /dev/null -w "%{http_code}" http://localhost:${port}/v1/models | grep -q "200"; then
            echo "[SUCCESS] ${service_name} is ready on port ${port}!"
            return 0
        fi
        
        # 如果 curl 失败，等待 5 秒后重试
        sleep 5
        retry_count=$((retry_count + 1))
        echo "[WAIT] ${service_name} not ready yet... (${retry_count}/${max_retries})"
    done
    
    echo "[ERROR] ${service_name} failed to start on port ${port} after ${max_retries} attempts."
    exit 1
}

wait_for_model() {
    local port=$1
    local service_name=$2
    local expected_model=$3
    local max_retries=3000
    local retry_count=0

    echo "[INFO] Waiting for ${service_name} model=${expected_model} on port ${port}..."
    while [ "${retry_count}" -lt "${max_retries}" ]; do
        if curl -fsS "http://localhost:${port}/v1/models" 2>/dev/null | grep -Fq "\"${expected_model}\""; then
            echo "[SUCCESS] ${service_name} serves ${expected_model} on port ${port}."
            return 0
        fi
        sleep 5
        retry_count=$((retry_count + 1))
    done
    echo "[ERROR] ${service_name} did not expose model ${expected_model} on port ${port}." >&2
    return 1
}

wait_for_port_release() {
    local port=$1
    local max_retries=60
    local retry_count=0
    while [ "${retry_count}" -lt "${max_retries}" ]; do
        if ! curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        retry_count=$((retry_count + 1))
    done
    echo "[ERROR] Port ${port} is still occupied after stopping the Judge service." >&2
    return 1
}

# ==========================================
# 执行等待逻辑
# ==========================================

if [ "${RUBRIC_EVAL_ONLY}" != "1" ]; then
# 1. 等待 Guard 服务
if [ "${REWARD_MODE_NEEDS_GUARD}" = "1" ]; then
wait_for_api "${GUARD_PORT}" "Guard Service" "${guard_service_pid}"
wait_for_model "${GUARD_PORT}" "Guard Service" "${GUARD_SERVED_MODEL_NAME}"
fi

# 2. 等待所有 Judge 服务
for replica_idx in "${!judge_ports[@]}"; do
    wait_for_api "${judge_ports[$replica_idx]}" "Qwen3 Judge Service ${replica_idx}" "${judge_service_pids[$replica_idx]}"
    wait_for_model "${judge_ports[$replica_idx]}" "Qwen3 Judge Service ${replica_idx}" "qwen3_32B_instruct"
done

if [ "${RUBRIC_EVIDENCE_V5_PREFLIGHT:-0}" = "1" ]; then
    echo "[PREFLIGHT] Checking real Judge, unchanged v4 cases, writer diagnosis and v5 admission..."
    (cd "${PROJECT_ROOT}" && python3 scripts/smoke_evidence_v5.py --output "${RUBRIC_RUN_DIR}/preflight/v5_gate.json")
fi
if [ "${RUBRIC_TARGETED_REFUSAL_ENABLE:-0}" = "1" ]; then
    if [ "${RUBRIC_GROUPED_TUNING_ENABLE:-0}" = "1" ]; then
        (cd "${PROJECT_ROOT}" && python3 scripts/benchmark_grouped_dispatch.py --output "${RUBRIC_RUN_DIR}/grouped_tuning")
        selection="${RUBRIC_RUN_DIR}/grouped_tuning/selection.json"
        selection_values=$(cd "${PROJECT_ROOT}" && python3 scripts/benchmark_grouped_dispatch.py --verify-selection "${selection}")
        read -r RUBRIC_GROUPED_API_CONCURRENCY RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE RUBRIC_GROUPED_WIRE_FORMAT <<< "${selection_values}"
        export RUBRIC_GROUPED_API_CONCURRENCY RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE RUBRIC_GROUPED_WIRE_FORMAT
        echo "[GROUPED_TUNING] Selected grouped workers=${RUBRIC_GROUPED_API_CONCURRENCY}, fair=${RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE}, wire=${RUBRIC_GROUPED_WIRE_FORMAT}; original mandatory gates still follow."
    fi
    if [ "${RUBRIC_JUDGE_THROUGHPUT_CANARY_ENABLE:-0}" = "1" ]; then
        (cd "${PROJECT_ROOT}" && python3 scripts/benchmark_judge_dispatch.py --output "${RUBRIC_RUN_DIR}/throughput_canary")
        selection="${RUBRIC_RUN_DIR}/throughput_canary/selection.json"
        RUBRIC_JUDGE_API_CONCURRENCY=$(python3 -c 'import json,sys;x=json.load(open(sys.argv[1]));assert x["selected_worker_concurrency"] in (48,144);print(x["selected_worker_concurrency"])' "${selection}")
        RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE=$(python3 -c 'import json,sys;print(int(json.load(open(sys.argv[1]))["selected_fair_dispatch"]))' "${selection}")
        export RUBRIC_JUDGE_API_CONCURRENCY RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE
        echo "[THROUGHPUT] Real replay selected API workers=${RUBRIC_JUDGE_API_CONCURRENCY}, fair=${RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE}; mandatory gates still follow."
    fi
    echo "[PREFLIGHT] Validating targeted refusal against frozen development cases and historical answers..."
    if [ "${RUBRIC_GRADED_IMPACT_ENABLE:-0}" = "1" ]; then
    (cd "${PROJECT_ROOT}" && python3 scripts/graded_presentation_gate.py --output "${RUBRIC_RUN_DIR}/preflight")
    else
    (cd "${PROJECT_ROOT}" && python3 scripts/smoke_targeted_refusal.py --output "${RUBRIC_RUN_DIR}/preflight/targeted_refusal_gate.json")
    fi
fi
if [ "${RUBRIC_TARGETED_PREFLIGHT_ONLY:-0}" = "1" ]; then
    if [ "${RUBRIC_TARGETED_REFUSAL_ENABLE:-0}" != "1" ]; then
        echo "[ERROR] Preflight-only requires the complete real targeted Judge gate."
        exit 1
    fi
    echo "[PREFLIGHT] All training Judge replicas and pooled route passed; validation-only complete."
    exit 0
fi
if [ "${RUBRIC_FROZEN_REVIEW_ENABLE:-0}" = "1" ]; then
    echo "[NEW_REEVALUATION] 新复评: preserved frozen policies only; optimizer updates disabled."
else
    echo "[INFO] All API services are up and running. Starting training..."
fi

# 启动 GRPO 训练
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS:-4,5,6,7}"
bash "${PROJECT_ROOT}/examples/ppo_trainer/qwen3B-instruct-GRPO.sh"

if [ "${RUBRIC_TRAIN_BATCH_PREFLIGHT_ONLY:-0}" = "1" ]; then
    python3 - "${RUBRIC_RUN_DIR}/preflight/training_batch_gate.json" <<'PY'
import json, sys
with open(sys.argv[1]) as stream:
    report = json.load(stream)
assert report['passed'] is True and report['optimizer_updates'] == 0
assert report['response_count'] == 64 and report['prompt_count'] == 8
print('[PREFLIGHT] Actual training batch verified; validation-only complete.')
PY
    exit 0
fi

# Judge replicas are no longer needed after training. The Guard remains resident
# on its dedicated GPU for final safety evaluation.
stop_service_pids "${judge_service_pids[@]}"
judge_service_pids=()
for judge_port in "${judge_ports[@]}"; do
    wait_for_port_release "${judge_port}"
done

# A smoke run proves that online evolution really dispatched, called the
# writer, produced candidates, and persisted audit evidence. It intentionally
# exits before checkpoint merging and the expensive final evaluation. Normal
# training never enters this branch.
if [ "${RUBRIC_EVOLVE_SMOKE_TEST:-0}" = "1" ]; then
    smoke_save_dir="${RUBRIC_EVOLVE_SAVE_DIR:-${RUBRIC_RUN_DIR}/online_rubric_evolution}"
    smoke_status="${smoke_save_dir}/runtime_status.json"
    smoke_audit="${smoke_save_dir}/online_evolution_audit.jsonl"
    python3 - "${smoke_status}" "${smoke_audit}" <<'PY'
import json
import pathlib
import sys

status_path = pathlib.Path(sys.argv[1])
audit_path = pathlib.Path(sys.argv[2])
if not status_path.is_file():
    raise SystemExit(f"[SMOKE][ERROR] Missing runtime status: {status_path}")
status = json.loads(status_path.read_text(encoding="utf-8"))
required = {
    "dispatch_active": True,
}
for key, expected in required.items():
    if status.get(key) != expected:
        raise SystemExit(f"[SMOKE][ERROR] {key}={status.get(key)!r}, expected {expected!r}")
for key in ("generation_due_count", "generation_attempt_count", "candidate_rule_count"):
    if int(status.get(key, 0)) < 1:
        raise SystemExit(f"[SMOKE][ERROR] {key} did not advance: {status.get(key)!r}")
if not audit_path.is_file() or audit_path.stat().st_size == 0:
    raise SystemExit(f"[SMOKE][ERROR] Missing evolution audit: {audit_path}")
records = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not records:
    raise SystemExit("[SMOKE][ERROR] Evolution audit contains no records")
print("[SMOKE][PASS] Online evolution dispatched and persisted evidence.")
print(json.dumps(status, ensure_ascii=False, sort_keys=True))
PY
    exit 0
fi
else
    echo "[EVAL_ONLY] Skipping training and all training-time Judge services."
fi

if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ] && ! kill -0 "${guard_service_pid}" 2>/dev/null; then
    echo "[WARN] Training-time Guard exited; restarting it for final evaluation."
    CUDA_VISIBLE_DEVICES="${GUARD_GPU_IDS}" nohup python -m vllm.entrypoints.openai.api_server \
        --model "${GUARD_MODEL_PATH}" \
        --served-model-name "${GUARD_SERVED_MODEL_NAME}" \
        --tensor-parallel-size "${GUARD_TP_SIZE}" \
        --port "${GUARD_PORT}" \
        --dtype bfloat16 \
        --gpu-memory-utilization "${GUARD_GPU_MEMORY_UTILIZATION:-0.8}" \
        > "${API_LOG_DIR}/final_safety_guard_vllm.log" 2>&1 &
    guard_service_pid=$!
fi

if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
    wait_for_api "${GUARD_PORT}" "Final Qwen3Guard Service"
    wait_for_model "${GUARD_PORT}" "Final Qwen3Guard Service" "${GUARD_SERVED_MODEL_NAME}"
fi

export SAFETY_GUARD_API_BASE="http://127.0.0.1:${GUARD_PORT}/v1"
export SAFETY_GUARD_MODEL="${GUARD_SERVED_MODEL_NAME}"
# 合并模型 - 自动遍历所有 global_step_* checkpoints
# =========================
# 6. 合并所有 Checkpoints (使用指定的 model_merger.py)
# =========================
echo "[INFO] ============================================="
echo "[INFO] Merging All Checkpoints"
echo "[INFO] ============================================="

BASE_MODEL_NAME="${POLICY_MODEL_ID}"
MERGED_MODELS_DIR="${CKPT_PATH}/models"
FINAL_EVAL_MIN_STEP="${FINAL_EVAL_MIN_STEP:-0}"
FINAL_SAFETY_EVAL_STEPS="${FINAL_SAFETY_EVAL_STEPS:-600,900,last,best}"
if ! [[ "${FINAL_EVAL_MIN_STEP}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] FINAL_EVAL_MIN_STEP must be a non-negative integer." >&2
  exit 2
fi
mkdir -p "${MERGED_MODELS_DIR}"
MERGED_THIS_RUN=()
SAFETY_EVAL_MODELS=()
BEST_MERGED_MODEL=""
LAST_MERGED_MODEL=""
LAST_MERGED_STEP=-1
MERGE_STAGING_ROOT="${MERGE_STAGING_ROOT:-${RAY_TMPDIR:-/tmp}/model_merge_staging}"
mkdir -p "${MERGE_STAGING_ROOT}"

merge_rubricalign_checkpoint() {
  local actor_dir="$1"
  local target_dir="$2"
  local step_name="$3"
  local staging_dir merge_rc=0
  staging_dir="$(mktemp -d "${MERGE_STAGING_ROOT%/}/rubricalign-${step_name}.XXXXXX")"
  echo "[INFO] Node-local merge staging: ${staging_dir}"
  python3 "${PROJECT_ROOT}/scripts/model_merger.py" \
    --backend fsdp \
    --hf_model_path "${POLICY_MODEL_PATH}" \
    --local_dir "${actor_dir}" \
    --target_dir "${staging_dir}" || merge_rc=$?
  if [ "${merge_rc}" -eq 0 ]; then
    python3 "${PROJECT_ROOT}/scripts/move_tokenizer.py" \
      --source "${POLICY_MODEL_PATH}" \
      --destination "${staging_dir}" || merge_rc=$?
  fi
  if [ "${merge_rc}" -ne 0 ] || [ ! -f "${staging_dir}/config.json" ]; then
    rm -rf "${staging_dir}"
    return 1
  fi
  mkdir -p "${target_dir}"
  cp -a "${staging_dir}/." "${target_dir}/"
  rm -rf "${staging_dir}"
  [ -f "${target_dir}/config.json" ]
}

# 查找所有 global_step_* 目录
CHECKPOINT_SOURCE_DIR="${RUBRIC_EVAL_SOURCE_DIR:-${CKPT_PATH}}"
if [ "${RUBRIC_FROZEN_REVIEW_ENABLE:-0}" = "1" ]; then
  CHECKPOINT_SOURCE_DIR="${CKPT_PATH}/new_reevaluation/selected_checkpoints"
fi
if [ "${RUBRIC_EVAL_ONLY}" = "1" ]; then
  # A local symlink view keeps exported actor-only sources read-only and gives
  # the existing merger an unambiguous checkpoint name. Outputs stay on new OSS.
  if [ "${RUBRIC_EVAL_USE_RECOVERY_COPY:-0}" = "1" ]; then
    CHECKPOINT_SOURCE_DIR="${RUBRIC_EVAL_SOURCE_DIR}"
  else
    CHECKPOINT_SOURCE_DIR="$(mktemp -d "${MERGE_STAGING_ROOT%/}/eval-source.XXXXXX")"
    mkdir -p "${CHECKPOINT_SOURCE_DIR}/global_step_${RUBRIC_EVAL_STEP}"
    ln -s "${RUBRIC_EVAL_ACTOR_DIR}" "${CHECKPOINT_SOURCE_DIR}/global_step_${RUBRIC_EVAL_STEP}/actor"
  fi
fi
if [ "${RUBRIC_VALIDATION_REPLAY_ENABLE:-0}" = "1" ]; then
  CHECKPOINT_SOURCE_DIR="${CKPT_PATH}/original_validation_replay_selected"
fi
checkpoint_pattern='global_step_*'
if [ "${RUBRIC_EVAL_ONLY}" = "1" ]; then
  # Evaluate the requested numbered copy once, even if best points to that step.
  checkpoint_pattern="global_step_${RUBRIC_EVAL_STEP}"
fi
mapfile -t CKPT_DIRS < <(find "${CHECKPOINT_SOURCE_DIR}" -maxdepth 1 -mindepth 1 -type d -name "${checkpoint_pattern}" | sort -V)

if [ "${FINAL_EVAL_EXACT_TRIO:-0}" = "1" ]; then
  # Capture exit status before mapfile: process substitution would hide failures.
  selected_paths="$(python3 "${PROJECT_ROOT}/scripts/select_final_checkpoints.py" \
    --root "${CHECKPOINT_SOURCE_DIR}" \
    --requested "${FINAL_SAFETY_EVAL_STEPS}" \
    --expected-last-step "${FINAL_EVAL_EXPECTED_LAST_STEP:-0}" \
    --manifest "${EVAL_OUTPUT_DIR}/checkpoint_selection.json")"
  mapfile -t CKPT_DIRS <<< "${selected_paths}"
  SELECTED_BEST_CHECKPOINT_PATH="$(python3 - "${EVAL_OUTPUT_DIR}/checkpoint_selection.json" <<'PYROLE'
import json,sys
v=json.load(open(sys.argv[1]))
rows=[r['checkpoint'] for r in v['checkpoints'] if 'best' in r['roles']]
if len(rows)>1:raise ValueError('Ambiguous selected best role')
print(rows[0] if rows else '')
PYROLE
)"
fi

if [ "${#CKPT_DIRS[@]}" -eq 0 ]; then
  echo "[ERROR] No checkpoint directories found in ${CHECKPOINT_SOURCE_DIR}" >&2
  exit 1
fi

LAST_STANDARD_STEP=-1
# Determine last from all numbered checkpoints, even when the selected best
# artifact replaces its numbered copy in CKPT_DIRS.
for ckpt_dir in "${CHECKPOINT_SOURCE_DIR}"/global_step_[0-9]*; do
  step_name="$(basename "${ckpt_dir}")"
  if [[ "${step_name}" =~ ^global_step_([0-9]+)$ ]] && [ "${BASH_REMATCH[1]}" -gt "${LAST_STANDARD_STEP}" ]; then
    LAST_STANDARD_STEP="${BASH_REMATCH[1]}"
  fi
done
if [ "${LAST_STANDARD_STEP}" -lt 0 ]; then
  echo "[ERROR] No numbered global_step_<n> checkpoints found under ${CKPT_PATH}." >&2
  exit 1
fi
echo "[INFO] Final safety checkpoint selection: ${FINAL_SAFETY_EVAL_STEPS} (last=${LAST_STANDARD_STEP})"

for ckpt_dir in "${CKPT_DIRS[@]}"; do
  step_name="$(basename "${ckpt_dir}")"
  is_best_checkpoint=0
  if [[ "${step_name}" =~ ^global_step_([0-9]+)$ ]]; then
    checkpoint_step="${BASH_REMATCH[1]}"
  elif [[ "${step_name}" =~ ^global_step_best_([0-9]+)$ ]]; then
    checkpoint_step="${BASH_REMATCH[1]}"
    is_best_checkpoint=1
  else
    echo "[WARN] Cannot parse checkpoint step from ${step_name}; skipping final evaluation."
    continue
  fi
  is_selected=0
  if [[ ",${FINAL_SAFETY_EVAL_STEPS}," == *",${checkpoint_step},"* ]]; then
    is_selected=1
  fi
  if [ "${checkpoint_step}" = "${LAST_STANDARD_STEP}" ] && [[ ",${FINAL_SAFETY_EVAL_STEPS}," == *",last,"* ]]; then
    is_selected=1
  fi
  if [ "${is_best_checkpoint}" = "1" ] && [[ ",${FINAL_SAFETY_EVAL_STEPS}," == *",best,"* ]]; then
    is_selected=1
  fi
  # Exact selection already resolved roles from authoritative metadata. A
  # numbered best (e.g.1110) is still best, regardless of its directory name.
  if [ "${FINAL_EVAL_EXACT_TRIO:-0}" = "1" ]; then
    is_selected=1
    if [ "${ckpt_dir}" = "${SELECTED_BEST_CHECKPOINT_PATH}" ]; then
      is_best_checkpoint=1
    fi
  fi
  if [ "${is_selected}" != "1" ]; then
    echo "[SKIP] ${step_name} is outside FINAL_SAFETY_EVAL_STEPS=${FINAL_SAFETY_EVAL_STEPS}"
    continue
  fi
  if [ "${checkpoint_step}" -lt "${FINAL_EVAL_MIN_STEP}" ] && [ "${is_best_checkpoint}" != "1" ]; then
    echo "[SKIP] ${step_name} is below FINAL_EVAL_MIN_STEP=${FINAL_EVAL_MIN_STEP}"
    continue
  fi
  actor_dir="${ckpt_dir}/actor"
  
  # 检查 actor 目录是否存在
  if [ ! -d "${actor_dir}" ]; then
    echo "[WARN] Missing actor dir for ${ckpt_dir}, trying root dir..."
    actor_dir="${ckpt_dir}"
    if [ ! -f "${actor_dir}/pytorch_model.bin" ] && [ ! -f "${actor_dir}/model.safetensors" ]; then
         echo "[ERROR] Cannot find valid model files in ${ckpt_dir}. Skipping."
         continue
    fi
  fi

  target_dir="${MERGED_MODELS_DIR}/${BASE_MODEL_NAME}-${step_name}"
  
  # 跳过已合并的
  if [ -f "${target_dir}/config.json" ]; then
      echo "[SKIP] Already merged: ${step_name}"
      MERGED_THIS_RUN+=("${target_dir}")
      if [ "${is_selected}" = "1" ] && { [ "${checkpoint_step}" -ge "${FINAL_EVAL_MIN_STEP}" ] || [ "${is_best_checkpoint}" = "1" ]; }; then
          SAFETY_EVAL_MODELS+=("${target_dir}")
      fi
      if [ "${is_best_checkpoint}" = "1" ]; then
          BEST_MERGED_MODEL="${target_dir}"
      fi
      if [ "${is_best_checkpoint}" = "0" ] && [ "${checkpoint_step}" -gt "${LAST_MERGED_STEP}" ]; then
          LAST_MERGED_MODEL="${target_dir}"
          LAST_MERGED_STEP="${checkpoint_step}"
      fi
      continue
  fi

  echo "[INFO] Merging ${step_name} -> ${target_dir}"
  
  if merge_rubricalign_checkpoint "${actor_dir}" "${target_dir}" "${step_name}"; then
      echo "[DONE] Merged ${step_name}"
      MERGED_THIS_RUN+=("${target_dir}")
      if [ "${is_selected}" = "1" ] && { [ "${checkpoint_step}" -ge "${FINAL_EVAL_MIN_STEP}" ] || [ "${is_best_checkpoint}" = "1" ]; }; then
          SAFETY_EVAL_MODELS+=("${target_dir}")
      fi
      if [ "${is_best_checkpoint}" = "1" ]; then
          BEST_MERGED_MODEL="${target_dir}"
      fi
      if [ "${is_best_checkpoint}" = "0" ] && [ "${checkpoint_step}" -gt "${LAST_MERGED_STEP}" ]; then
          LAST_MERGED_MODEL="${target_dir}"
          LAST_MERGED_STEP="${checkpoint_step}"
      fi
  else
      echo "[ERROR] Failed to merge ${step_name}"
      if [ "${RUBRIC_EVAL_ONLY}" = "1" ] || [ "${FINAL_EVAL_EXACT_TRIO:-0}" = "1" ]; then
          exit 1
      fi
  fi
done

if [ "${FINAL_EVAL_EXACT_TRIO:-0}" = "1" ] && [ "${#MERGED_THIS_RUN[@]}" -ne "${#CKPT_DIRS[@]}" ]; then
  echo '[ERROR] At least one explicitly selected final checkpoint was not merged; refuse incomplete final evaluation.' >&2
  exit 1
fi

# A best copy is optional: it is only created when a validation metric is
# available at a save step.  Do not make general evaluation depend on that
# naming convention.  Recover the highest successfully merged numbered model
# (including models merged in an earlier invocation) as a deterministic
# fallback when the selected last checkpoint was unavailable.
if [ -z "${LAST_MERGED_MODEL}" ] || [ ! -f "${LAST_MERGED_MODEL}/config.json" ]; then
  while IFS= read -r config_file; do
    candidate_dir="$(dirname "${config_file}")"
    candidate_name="$(basename "${candidate_dir}")"
    candidate_step=""
    if [[ "${candidate_name}" =~ ^global_step_([0-9]+)$ ]]; then
      candidate_step="${BASH_REMATCH[1]}"
    elif [[ "${candidate_name}" =~ -global_step_([0-9]+)$ ]]; then
      candidate_step="${BASH_REMATCH[1]}"
    fi
    if [ -n "${candidate_step}" ] && [ "${candidate_step}" -gt "${LAST_MERGED_STEP}" ]; then
      LAST_MERGED_MODEL="${candidate_dir}"
      LAST_MERGED_STEP="${candidate_step}"
    fi
  done < <(find "${MERGED_MODELS_DIR}" -type f -name config.json | sort -V)
fi

# =========================
# 7. 批量安全评测
# =========================
if [ "${#MERGED_THIS_RUN[@]}" -eq 0 ]; then
  echo "[ERROR] No models were merged successfully." >&2
  exit 1
fi

if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
  if [ "${#SAFETY_EVAL_MODELS[@]}" -eq 0 ]; then
    echo "[ERROR] No successfully merged checkpoint satisfies FINAL_EVAL_MIN_STEP=${FINAL_EVAL_MIN_STEP}." >&2
    exit 1
  fi
  echo "[INFO] ============================================="
  echo "[INFO] Starting Safety Evaluation"
  echo "[INFO] ============================================="

  SAFETY_DIR="${PROJECT_ROOT}/Evaluation/safety"
  if [ ! -d "${SAFETY_DIR}" ]; then
    echo "[ERROR] Safety evaluation directory not found: ${SAFETY_DIR}" >&2
    exit 1
  fi

  cd "${SAFETY_DIR}"
  # Persist raw responses and every Guard attempt outside the temporary checkout.
  # A new invocation has its own evidence directory; previous results are retained.
  export SAFETY_RESULT_DIR="${EVAL_OUTPUT_DIR}/safety_results/${PRIMUS_JOB_SUBMIT_TIMESTAMP:-$(date +%s)}"
  export SAFETY_GUARD_EVIDENCE_DIR="${SAFETY_RESULT_DIR}/guard_attempts"
  safety_eval_started_at=$SECONDS
  for model_dir in "${SAFETY_EVAL_MODELS[@]}"; do
    model_name="$(basename "${model_dir}")"
    model_eval_started_at=$SECONDS
    echo "[EVAL] Processing: ${model_name}"

    log_file="${EVAL_OUTPUT_DIR}/${model_name}_eval.log"
    summary_file="${EVAL_OUTPUT_DIR}/${model_name}_summary.txt"
    mkdir -p "$(dirname "$log_file")"

    {
        echo "========================================"
        echo "Model: ${model_name}"
        echo "Time: $(date)"
        echo "========================================"
    } > "$summary_file"

    echo "[GEN] Running generate.py..."
    safety_eval_gpu_ids="${SAFETY_EVAL_GPU_IDS:-${TRAIN_GPU_IDS%%,*}}"
    if [[ "${safety_eval_gpu_ids}" == *,* ]]; then
      echo "[ERROR] SAFETY_EVAL_GPU_IDS must contain exactly one GPU; generate.py uses tensor_parallel_size=1." >&2
      exit 2
    fi
    CUDA_VISIBLE_DEVICES="${safety_eval_gpu_ids}" VLLM_WORKER_MULTIPROC_METHOD=spawn python3 generate.py --model_id "${model_dir}" 2>&1 | tee -a "$log_file"

    echo "[EVAL] Running eval_answer.py..."
    python3 eval_answer.py --model_id "${model_name}" --require-complete 2>&1 | tee -a "$log_file" >> "$summary_file"

    echo "[EVAL] Running XSTest over-refusal evaluation..."
    python3 overrefusal.py \
      --model_id "${model_name}" \
      --dataset_name xstest \
      --require-complete \
      2>&1 | tee -a "$log_file" >> "$summary_file"

    echo "[TIME] ${model_name} safety evaluation: $((SECONDS - model_eval_started_at))s" | tee -a "$log_file" >> "$summary_file"

    echo "[DONE] ${model_name}"
  done
  echo "[TIME] Total safety evaluation: $((SECONDS - safety_eval_started_at))s for ${#SAFETY_EVAL_MODELS[@]} model(s)."
else
  echo "[INFO] Final safety evaluation disabled."
fi

# Release the Guard before loading policy checkpoints for general evaluation.
if [ -n "${guard_service_pid}" ]; then
  stop_service_pids "${guard_service_pid}"
  guard_service_pid=""
fi

if [ "${FINAL_GENERAL_EVAL_ENABLE}" = "1" ]; then
  GENERAL_EVAL_SCRIPT="${PROJECT_ROOT}/Evaluation/lm_eval_last3_general.sh"
  if [ ! -f "${GENERAL_EVAL_SCRIPT}" ]; then
    echo "[ERROR] General evaluation script not found: ${GENERAL_EVAL_SCRIPT}" >&2
    exit 1
  fi

  if [ -z "${BEST_MERGED_MODEL}" ] || [ ! -f "${BEST_MERGED_MODEL}/config.json" ]; then
    if [ -n "${LAST_MERGED_MODEL}" ] && [ -f "${LAST_MERGED_MODEL}/config.json" ]; then
      echo "[WARN] No merged global_step_best_<step> was found under ${CKPT_PATH}." >&2
      echo "[WARN] Falling back to the highest successfully merged standard checkpoint (step ${LAST_MERGED_STEP}) for MMLU/GSM8K: ${LAST_MERGED_MODEL}" >&2
      BEST_MERGED_MODEL="${LAST_MERGED_MODEL}"
    else
      echo "[ERROR] Best checkpoint was not generated or merged successfully." >&2
      echo "[ERROR] Expected a checkpoint named global_step_best_<step> under ${CKPT_PATH}." >&2
      echo "[ERROR] No valid last merged checkpoint is available for the general evaluation fallback." >&2
      exit 1
    fi
  fi

  GENERAL_MODELS_DIR="$(dirname "${BEST_MERGED_MODEL}")"
  BEST_MODEL_NAME="$(basename "${BEST_MERGED_MODEL}")"

  if [ -n "${FINAL_GENERAL_EVAL_GPU_IDS:-}" ]; then
    GENERAL_EVAL_GPU_IDS="${FINAL_GENERAL_EVAL_GPU_IDS}"
  else
    general_gpu_candidates="${JUDGE_GPU_IDS},${GUARD_GPU_IDS},${TRAIN_GPU_IDS:-4,5,6,7}"
    GENERAL_EVAL_GPU_IDS=""
    IFS=',' read -r -a general_gpu_candidate_array <<< "${general_gpu_candidates}"
    for raw_gpu_id in "${general_gpu_candidate_array[@]}"; do
      gpu_id="$(echo "${raw_gpu_id}" | xargs)"
      if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Invalid GPU ID in general evaluation GPU list: ${raw_gpu_id}" >&2
        exit 2
      fi
      if [[ ",${GENERAL_EVAL_GPU_IDS}," != *",${gpu_id},"* ]]; then
        GENERAL_EVAL_GPU_IDS="${GENERAL_EVAL_GPU_IDS:+${GENERAL_EVAL_GPU_IDS},}${gpu_id}"
      fi
    done
  fi

  IFS=',' read -r -a general_eval_gpu_array <<< "${GENERAL_EVAL_GPU_IDS}"
  GENERAL_EVAL_GPU_COUNT="${#general_eval_gpu_array[@]}"
  GENERAL_EVAL_TP_SIZE="${FINAL_GENERAL_EVAL_TP_SIZE:-1}"
  if [ "${FINAL_GENERAL_EVAL_BACKEND}" = "vllm" ]; then
    GENERAL_EVAL_REPLICAS="${FINAL_GENERAL_EVAL_REPLICAS:-1}"
  else
    GENERAL_EVAL_REPLICAS=1
  fi
  if ! [[ "${GENERAL_EVAL_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${GENERAL_EVAL_REPLICAS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] General evaluation TP size and replica count must be positive integers." >&2
    exit 2
  fi
  if [ $((GENERAL_EVAL_TP_SIZE * GENERAL_EVAL_REPLICAS)) -gt "${GENERAL_EVAL_GPU_COUNT}" ]; then
    echo "[ERROR] General evaluation requests $((GENERAL_EVAL_TP_SIZE * GENERAL_EVAL_REPLICAS)) GPUs, but only ${GENERAL_EVAL_GPU_COUNT} were assigned." >&2
    exit 2
  fi

  echo "[INFO] ============================================="
  echo "[INFO] Starting MMLU and GSM8K Evaluation"
  echo "[INFO] Best model: ${BEST_MERGED_MODEL}"
  echo "[INFO] Backend: ${FINAL_GENERAL_EVAL_BACKEND}"
  echo "[INFO] GPUs: ${GENERAL_EVAL_GPU_IDS}; replicas=${GENERAL_EVAL_REPLICAS}; TP=${GENERAL_EVAL_TP_SIZE}"
  echo "[INFO] ============================================="

  if [ "${FINAL_GENERAL_EVAL_BACKEND}" = "vllm" ]; then
    ray stop --force 2>/dev/null || true
  fi

  GENERAL_EVAL_MODELS=("${BEST_MERGED_MODEL}")
  if [ "${FINAL_EVAL_EXACT_TRIO:-0}" = "1" ]; then
    GENERAL_EVAL_MODELS=("${MERGED_THIS_RUN[@]}")
  fi
  general_eval_prepared=0
  for general_model_dir in "${GENERAL_EVAL_MODELS[@]}"; do
  GENERAL_MODELS_DIR="$(dirname "${general_model_dir}")"
  BEST_MODEL_NAME="$(basename "${general_model_dir}")"
  echo "[GENERAL] Evaluating ${BEST_MODEL_NAME}"
  CUDA_VISIBLE_DEVICES="${GENERAL_EVAL_GPU_IDS}" \
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  LM_EVAL_DP_SIZE="${GENERAL_EVAL_REPLICAS}" \
  LM_EVAL_TP_SIZE="${GENERAL_EVAL_TP_SIZE}" \
  LM_EVAL_EXTRA_MODEL_ARGS="${LM_EVAL_EXTRA_MODEL_ARGS:-max_model_len=4096}" \
  SKIP_LM_EVAL_INSTALL="${SKIP_LM_EVAL_INSTALL:-${general_eval_prepared}}" \
  SKIP_LM_EVAL_DATA_PREPARE="${SKIP_LM_EVAL_DATA_PREPARE:-${general_eval_prepared}}" \
  bash "${GENERAL_EVAL_SCRIPT}" \
    --ckpt-path "${CKPT_PATH}" \
    --models-dir "${GENERAL_MODELS_DIR}" \
    --glob "${BEST_MODEL_NAME}" \
    --num-models 1 \
    --tasks "gsm8k_cot_modelscope,mmlu_modelscope" \
    --backend "${FINAL_GENERAL_EVAL_BACKEND}" \
    --device "${FINAL_GENERAL_EVAL_DEVICE}"
  # Environment and local benchmark preparation are shared by all checkpoints.
  general_eval_prepared=1
  done
else
  echo "[INFO] Final MMLU/GSM8K evaluation disabled."
fi

echo "[INFO] train_val.sh completed successfully."
