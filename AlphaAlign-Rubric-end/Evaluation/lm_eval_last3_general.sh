#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HARNESS_DIR="${SCRIPT_DIR}/lm-evaluation-harness"
DATA_PREP_SCRIPT="${SCRIPT_DIR}/prepare_lm_eval_datasets_modelscope.py"
SCORE_SUMMARY_SCRIPT="${SCRIPT_DIR}/summarize_mmlu_gsm8k.py"

usage() {
  cat <<'USAGE'
Usage:
  bash Evaluation/lm_eval_last3_general.sh [options]

Description:
  使用 lm-evaluation-harness（不依赖 sglang）评测合并后的最后 3 个 step 模型。
  默认任务（ModelScope 本地数据版）：
    - gsm8k_cot_modelscope
    - mmlu_modelscope

  默认会从训练脚本对应的合并模型目录中挑选：
    - 优先: ${CKPT_PATH}/models/Qwen/*-global_step_*
    - 否则: ${CKPT_PATH}/models/*-global_step_*

Options:
  --ckpt-path <dir>         checkpoint 根目录（默认: $CKPT_PATH 或 $VERL_HOME/checkpoints/qwen25_3B_instruct）
  --models-dir <dir>        合并后模型目录（默认自动探测 $CKPT_PATH/models/Qwen 或 $CKPT_PATH/models）
  --glob <pattern>          模型目录匹配模式（默认: *-global_step_*）
  --num-models <n>          选择最近 n 个模型（默认: 3）
  --tasks <csv>             逗号分隔任务（默认: gsm8k_cot_modelscope,mmlu_modelscope）
  --backend <hf|vllm>       lm-eval 后端（默认: hf）
  --device <device>         hf 后端设备（默认: cuda:0）
  --batch-size <value>      batch size（默认: auto）
  --limit <n>               仅跑前 n 条样本，便于测试脚本
  --data-dir <dir>          本地 benchmark 数据和任务配置目录
  --skip-data-prepare       跳过 ModelScope 数据准备
  --skip-install            跳过依赖安装
  -h, --help                显示帮助

Useful env vars:
  VERL_HOME
  CKPT_PATH
  EVAL_OUTPUT_DIR
  PYTHON_BIN
  LM_EVAL_DTYPE
  LM_EVAL_NUMPY_PIN
  LM_EVAL_EXTRA_MODEL_ARGS
  LM_EVAL_GPU_MEMORY_UTILIZATION
  LM_EVAL_TP_SIZE
  LM_EVAL_DP_SIZE
  LM_EVAL_RESUME                  Set 1 to verify and reuse completed tasks
  LM_EVAL_EXTRA_MODEL_ARGS

Examples:
  bash Evaluation/lm_eval_last3_general.sh
  bash Evaluation/lm_eval_last3_general.sh --limit 20
  bash Evaluation/lm_eval_last3_general.sh --backend vllm --device cuda:0
USAGE
}

export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export CKPT_PATH="${CKPT_PATH:-${VERL_HOME}/checkpoints/qwen25_3B_instruct}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-${VERL_HOME}}/eval_outputs_general}"

PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS_DIR=""
MODEL_GLOB="${LM_EVAL_MODEL_GLOB:-*-global_step_*}"
NUM_MODELS="${LM_EVAL_NUM_MODELS:-3}"
TASKS_CSV="${LM_EVAL_TASKS:-gsm8k_cot_modelscope,mmlu_modelscope}"
BACKEND="${LM_EVAL_BACKEND:-hf}"
DEVICE="${LM_EVAL_DEVICE:-cuda:0}"
BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LIMIT="${LM_EVAL_LIMIT:-}"
SKIP_INSTALL="${SKIP_LM_EVAL_INSTALL:-0}"
SKIP_DATA_PREPARE="${SKIP_LM_EVAL_DATA_PREPARE:-0}"
SKIP_TE_CLEANUP="${SKIP_TE_CLEANUP:-0}"
DTYPE="${LM_EVAL_DTYPE:-bfloat16}"
EXTRA_MODEL_ARGS="${LM_EVAL_EXTRA_MODEL_ARGS:-}"
GPU_MEMORY_UTILIZATION="${LM_EVAL_GPU_MEMORY_UTILIZATION:-0.85}"
TP_SIZE="${LM_EVAL_TP_SIZE:-1}"
DP_SIZE="${LM_EVAL_DP_SIZE:-1}"
APPLY_CHAT_TEMPLATE="${LM_EVAL_APPLY_CHAT_TEMPLATE:-1}"
LOCAL_DATA_DIR="${LM_EVAL_LOCAL_DATA_ROOT:-${SCRIPT_DIR}/lm_eval_local_data_modelscope}"
LOCAL_TASKS_DIR="${LOCAL_DATA_DIR}/tasks"
# transformers 4.57.x declares huggingface-hub>=0.34.0,<1.0. Installing the hub
# unpinned lets pip resolve the 1.x line, which makes `import transformers` raise
# at eval time. Keep this aligned with examples/ppo_trainer/requirement.txt.
HF_HUB_PIN="${LM_EVAL_HF_HUB_PIN:-0.36.2}"
# Set before importing vLLM or launching Ray workers. Forking a process that
# already owns Ray communication threads can stall engine initialization.
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
# Keep compiled extensions from the base image (notably soxr) on the NumPy 1.x
# ABI. ModelScope/datasets otherwise allow pip to upgrade NumPy while they are
# force-reinstalled, which only surfaces later when PEFT loads a model class.
NUMPY_PIN="${LM_EVAL_NUMPY_PIN:-1.26.4}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt-path)
      CKPT_PATH="${2:?missing value for --ckpt-path}"
      shift 2
      ;;
    --models-dir)
      MODELS_DIR="${2:?missing value for --models-dir}"
      shift 2
      ;;
    --glob)
      MODEL_GLOB="${2:?missing value for --glob}"
      shift 2
      ;;
    --num-models)
      NUM_MODELS="${2:?missing value for --num-models}"
      shift 2
      ;;
    --tasks)
      TASKS_CSV="${2:?missing value for --tasks}"
      shift 2
      ;;
    --backend)
      BACKEND="${2:?missing value for --backend}"
      shift 2
      ;;
    --device)
      DEVICE="${2:?missing value for --device}"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="${2:?missing value for --batch-size}"
      shift 2
      ;;
    --limit)
      LIMIT="${2:?missing value for --limit}"
      shift 2
      ;;
    --data-dir)
      LOCAL_DATA_DIR="${2:?missing value for --data-dir}"
      LOCAL_TASKS_DIR="${LOCAL_DATA_DIR}/tasks"
      shift 2
      ;;
    --skip-data-prepare)
      SKIP_DATA_PREPARE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
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

TASKS_CSV="${TASKS_CSV//[[:space:]]/}"

if [[ -z "${MODELS_DIR}" ]]; then
  if [[ -d "${CKPT_PATH}/models/Qwen" ]]; then
    MODELS_DIR="${CKPT_PATH}/models/Qwen"
  else
    MODELS_DIR="${CKPT_PATH}/models"
  fi
fi

if [[ ! -d "${HARNESS_DIR}" ]]; then
  echo "[ERROR] lm-evaluation-harness not found: ${HARNESS_DIR}" >&2
  exit 1
fi
if [[ ! -f "${DATA_PREP_SCRIPT}" ]]; then
  echo "[ERROR] dataset prepare script not found: ${DATA_PREP_SCRIPT}" >&2
  exit 1
fi
if [[ ! -f "${SCORE_SUMMARY_SCRIPT}" ]]; then
  echo "[ERROR] score summary script not found: ${SCORE_SUMMARY_SCRIPT}" >&2
  exit 1
fi

dataset_prepare_tasks=()
[[ ",${TASKS_CSV}," == *",gsm8k_cot_modelscope,"* ]] && dataset_prepare_tasks+=(gsm8k)
[[ ",${TASKS_CSV}," == *",mmlu_modelscope,"* ]] && dataset_prepare_tasks+=(mmlu)
[[ ",${TASKS_CSV}," == *",gpqa_main_cot_zeroshot_modelscope,"* ]] && dataset_prepare_tasks+=(gpqa)
if [[ "${#dataset_prepare_tasks[@]}" -eq 0 ]]; then
  echo "[ERROR] No supported local ModelScope task found in --tasks=${TASKS_CSV}" >&2
  exit 2
fi
DATASET_PREPARE_TASKS="$(IFS=,; echo "${dataset_prepare_tasks[*]}")"
WRITE_MMLU_GSM8K_SUMMARY=0
if [[ ",${TASKS_CSV}," == *",gsm8k_cot_modelscope,"* && ",${TASKS_CSV}," == *",mmlu_modelscope,"* ]]; then
  WRITE_MMLU_GSM8K_SUMMARY=1
fi

install_lm_eval_deps() {
  if [[ "${SKIP_INSTALL}" == "1" ]]; then
    echo "[INFO] Skip dependency installation."
    return 0
  fi

  echo "[INFO] Installing ModelScope/datasets compatibility set ..."
  # datasets>=2.21 exports LargeList; datasets<4 keeps dataset-script
  # trust_remote_code support and DatasetBuilder.as_dataset(verification_mode).
  "${PYTHON_BIN}" -m pip install --upgrade --force-reinstall \
    "numpy==${NUMPY_PIN}" \
    "datasets==3.6.0" \
    "modelscope[datasets]>=1.31.0"

  echo "[INFO] Installing lm-eval dependencies with backend=${BACKEND}"
  if [[ "${BACKEND}" == "vllm" ]]; then
    "${PYTHON_BIN}" -m pip install -e "${HARNESS_DIR}[vllm]"
  else
    "${PYTHON_BIN}" -m pip install -e "${HARNESS_DIR}"
  fi
  "${PYTHON_BIN}" -m pip install sentencepiece fire "huggingface_hub==${HF_HUB_PIN}"
  "${PYTHON_BIN}" -m pip install --upgrade --force-reinstall \
    "numpy==${NUMPY_PIN}" \
    "datasets==3.6.0" \
    "modelscope[datasets]>=1.31.0"
  # Restore both pins last so later dependency resolution cannot leave the
  # interpreter with a NumPy/soxr ABI mismatch or an unsupported hub release.
  "${PYTHON_BIN}" -m pip install --force-reinstall \
    "numpy==${NUMPY_PIN}" \
    "huggingface_hub==${HF_HUB_PIN}"
}

verify_eval_env() {
  # lm_eval imports transformers at module load. If that import is broken the run
  # fails once per (model, task) with the real cause buried far above the final
  # error, so check it up front and abort before any GPU work.
  echo "[INFO] Verifying eval environment imports ..."
  if "${PYTHON_BIN}" - <<'PY'
import sys

try:
    import huggingface_hub
    import numpy
    import peft
    import transformers
    # Import the lazy model symbol that previously triggered
    # transformers.audio_utils -> soxr.cysoxr at task launch.
    from transformers import BloomPreTrainedModel  # noqa: F401
except Exception as exc:
    print(f"[ERROR] import failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

print(f"[INFO] transformers={transformers.__version__}")
print(f"[INFO] huggingface_hub={huggingface_hub.__version__}")
print(f"[INFO] numpy={numpy.__version__}")
print(f"[INFO] peft={getattr(peft, '__version__', 'unknown')}")
PY
  then
    return 0
  fi

  {
    echo "[ERROR] The general-evaluation Python environment is not import-compatible."
    echo "[ERROR] NumPy must remain on ${NUMPY_PIN} so compiled extensions such as soxr can load."
    echo "[ERROR] transformers 4.57.x requires huggingface-hub>=0.34.0,<1.0."
    echo "[ERROR] Repair with: ${PYTHON_BIN} -m pip install --force-reinstall 'numpy==${NUMPY_PIN}' 'huggingface_hub==${HF_HUB_PIN}'"
    echo "[ERROR] Pins can be overridden with LM_EVAL_NUMPY_PIN and LM_EVAL_HF_HUB_PIN."
  } >&2
  exit 1
}

cleanup_transformer_engine() {
  if [[ "${SKIP_TE_CLEANUP}" == "1" ]]; then
    echo "[INFO] Skip transformer-engine cleanup."
    return 0
  fi
  echo "[INFO] Removing potentially broken transformer-engine packages ..."
  "${PYTHON_BIN}" -m pip uninstall -y \
    transformer-engine \
    transformer_engine \
    transformer-engine-cu12 \
    transformer-engine-cu13 || true
}

prepare_local_datasets() {
  if [[ "${SKIP_DATA_PREPARE}" == "1" ]]; then
    echo "[INFO] Skip ModelScope dataset preparation."
    return 0
  fi
  mkdir -p "${LOCAL_DATA_DIR}"
  if "${PYTHON_BIN}" "${DATA_PREP_SCRIPT}" \
    --output-dir "${LOCAL_DATA_DIR}" \
    --harness-dir "${HARNESS_DIR}" \
    --tasks "${DATASET_PREPARE_TASKS}" \
    --validate-only; then
    echo "[INFO] Reusing validated local benchmark data."
    return 0
  fi
  echo "[INFO] Local benchmark data is missing or incomplete; downloading it now."
  "${PYTHON_BIN}" "${DATA_PREP_SCRIPT}" \
    --output-dir "${LOCAL_DATA_DIR}" \
    --harness-dir "${HARNESS_DIR}" \
    --tasks "${DATASET_PREPARE_TASKS}"
}

build_model_args() {
  local model_dir="$1"
  local model_args=""
  if [[ "${BACKEND}" == "vllm" ]]; then
    model_args="pretrained=${model_dir},tensor_parallel_size=${TP_SIZE},data_parallel_size=${DP_SIZE},dtype=${DTYPE},gpu_memory_utilization=${GPU_MEMORY_UTILIZATION},trust_remote_code=True"
  else
    model_args="pretrained=${model_dir},trust_remote_code=True,dtype=${DTYPE}"
  fi
  if [[ -n "${EXTRA_MODEL_ARGS}" ]]; then
    model_args="${model_args},${EXTRA_MODEL_ARGS}"
  fi
  printf '%s' "${model_args}"
}

append_summary() {
  local results_dir="$1"
  local task_name="$2"
  local summary_file="$3"

  local json_file=""
  json_file="$(find "${results_dir}" -type f -name '*.json' 2>/dev/null | sort | tail -n 1 || true)"
  if [[ -z "${json_file}" ]]; then
    echo "[WARN] No JSON result found under ${results_dir}" >> "${summary_file}"
    return 0
  fi

  "${PYTHON_BIN}" - "${json_file}" "${task_name}" >> "${summary_file}" <<'PY'
import json
import sys

path, task = sys.argv[1], sys.argv[2]
with open(path, "r", encoding="utf-8") as f:
    obj = json.load(f)

print(f"[SUMMARY] task={task}")
target = None

for bucket_name in ("results", "groups", "group_subtasks"):
    bucket = obj.get(bucket_name, {})
    if task in bucket:
        target = bucket[task]
        break

if target is None:
    print(f"  result_file={path}")
    print("  parsed_metrics=NOT_FOUND")
    raise SystemExit(0)

printed = False
for key, value in target.items():
    if isinstance(value, (int, float, str)):
        print(f"  {key}={value}")
        printed = True

if not printed:
    print(f"  result_file={path}")
    print("  parsed_metrics=EMPTY")
PY
}

run_one_task() {
  local model_dir="$1"
  local model_name="$2"
  local task_name="$3"
  local model_output_dir="$4"
  local log_file="$5"
  local summary_file="$6"

  local task_output_dir="${model_output_dir}/${task_name}"
  mkdir -p "${task_output_dir}"

  local model_args
  model_args="$(build_model_args "${model_dir}")"

  local -a cmd
  cmd=("${PYTHON_BIN}" -m lm_eval
       --model "${BACKEND}"
       --model_args "${model_args}"
       --tasks "${task_name}"
       --include_path "${LOCAL_TASKS_DIR}"
       --batch_size "${BATCH_SIZE}"
       --output_path "${task_output_dir}")

  if [[ "${BACKEND}" == "hf" ]]; then
    cmd+=(--device "${DEVICE}")
  fi
  if [[ -n "${LIMIT}" ]]; then
    cmd+=(--limit "${LIMIT}")
  fi
  if [[ "${APPLY_CHAT_TEMPLATE}" == "1" ]]; then
    cmd+=(--apply_chat_template)
  fi

  if [[ "${LM_EVAL_RESUME:-0}" == "1" ]]; then
    cmd=("${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/resumable_general_eval.py"
         --model "${model_dir}" --data "${LOCAL_DATA_DIR}"
         --harness "${HARNESS_DIR}" --output "${task_output_dir}"
         --task "${task_name}" -- "${cmd[@]}")
  fi

  {
    echo ""
    echo "========================================"
    echo "[RUN] model=${model_name}"
    echo "[RUN] task=${task_name}"
    echo "[RUN] backend=${BACKEND}"
    echo "[RUN] output=${task_output_dir}"
    echo "========================================"
  } | tee -a "${log_file}" >> "${summary_file}"

  if "${cmd[@]}" 2>&1 | tee -a "${log_file}"; then
    echo "[OK] task=${task_name}" >> "${summary_file}"
    append_summary "${task_output_dir}" "${task_name}" "${summary_file}"
  else
    echo "[FAIL] task=${task_name}" >> "${summary_file}"
    return 1
  fi
}

cleanup_transformer_engine
install_lm_eval_deps
verify_eval_env
prepare_local_datasets

mkdir -p "${EVAL_OUTPUT_DIR}"

if [[ ! -d "${MODELS_DIR}" ]]; then
  echo "[ERROR] merged models dir not found: ${MODELS_DIR}" >&2
  exit 1
fi

mapfile -t ALL_MODELS < <(find "${MODELS_DIR}" -maxdepth 1 -mindepth 1 -type d -name "${MODEL_GLOB}" | sort -V)
if [[ "${#ALL_MODELS[@]}" -eq 0 ]]; then
  echo "[ERROR] No merged models found in ${MODELS_DIR} matching ${MODEL_GLOB}" >&2
  exit 1
fi

if ! [[ "${NUM_MODELS}" =~ ^[0-9]+$ ]] || [[ "${NUM_MODELS}" -le 0 ]]; then
  echo "[ERROR] --num-models must be a positive integer, got: ${NUM_MODELS}" >&2
  exit 1
fi

mapfile -t SELECTED_MODELS < <(printf '%s\n' "${ALL_MODELS[@]}" | tail -n "${NUM_MODELS}")

IFS=',' read -r -a TASK_ARRAY <<< "${TASKS_CSV}"
if [[ "${#TASK_ARRAY[@]}" -eq 0 ]]; then
  echo "[ERROR] No tasks specified." >&2
  exit 1
fi

echo "[INFO] VERL_HOME=${VERL_HOME}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] MODELS_DIR=${MODELS_DIR}"
echo "[INFO] EVAL_OUTPUT_DIR=${EVAL_OUTPUT_DIR}"
echo "[INFO] LOCAL_DATA_DIR=${LOCAL_DATA_DIR}"
echo "[INFO] LOCAL_TASKS_DIR=${LOCAL_TASKS_DIR}"
echo "[INFO] BACKEND=${BACKEND}"
echo "[INFO] TASKS=${TASKS_CSV}"
echo "[INFO] Selected models:"
printf '  - %s\n' "${SELECTED_MODELS[@]}"

cd "${HARNESS_DIR}"

FAILED_ITEMS=()

for model_dir in "${SELECTED_MODELS[@]}"; do
  model_name="$(basename "${model_dir}")"
  model_output_dir="${EVAL_OUTPUT_DIR}/${model_name}"
  log_file="${EVAL_OUTPUT_DIR}/${model_name}_lm_eval.log"
  summary_file="${EVAL_OUTPUT_DIR}/${model_name}_lm_eval_summary.txt"

  mkdir -p "${model_output_dir}"
  {
    echo "========================================"
    echo "lm-eval summary"
    echo "model_name=${model_name}"
    echo "model_path=${model_dir}"
    echo "backend=${BACKEND}"
    echo "tasks=${TASKS_CSV}"
    echo "timestamp=$(date)"
    echo "========================================"
  } > "${summary_file}"

  model_task_failures=0
  for raw_task in "${TASK_ARRAY[@]}"; do
    task_name="$(echo "${raw_task}" | xargs)"
    [[ -z "${task_name}" ]] && continue
    if ! run_one_task "${model_dir}" "${model_name}" "${task_name}" "${model_output_dir}" "${log_file}" "${summary_file}"; then
      FAILED_ITEMS+=("${model_name}:${task_name}")
      model_task_failures=$((model_task_failures + 1))
    fi
  done

  if [[ "${WRITE_MMLU_GSM8K_SUMMARY}" == "1" ]]; then
    if [[ "${model_task_failures}" -gt 0 ]]; then
      # A failed task leaves no result JSON, so the summarizer would raise
      # FileNotFoundError and bury the actual upstream failure.
      echo "[SKIP] score summary for ${model_name}: ${model_task_failures} task(s) failed" \
        | tee -a "${summary_file}"
    else
      score_file="${EVAL_OUTPUT_DIR}/${model_name}_general_scores.txt"
      if ! "${PYTHON_BIN}" "${SCORE_SUMMARY_SCRIPT}" \
        --mmlu-results "${model_output_dir}/mmlu_modelscope" \
        --gsm8k-results "${model_output_dir}/gsm8k_cot_modelscope" \
        --output "${score_file}" | tee -a "${summary_file}"; then
        FAILED_ITEMS+=("${model_name}:score_summary")
      fi
    fi
  fi
done

if [[ "${#FAILED_ITEMS[@]}" -gt 0 ]]; then
  echo "[WARN] Some evaluations failed:"
  printf '  - %s\n' "${FAILED_ITEMS[@]}"
  exit 1
fi

if [[ "${WRITE_MMLU_GSM8K_SUMMARY}" == "1" ]]; then
  combined_score_file="${EVAL_OUTPUT_DIR}/mmlu_gsm8k_scores.txt"
  : > "${combined_score_file}"
  for model_dir in "${SELECTED_MODELS[@]}"; do
    model_name="$(basename "${model_dir}")"
    score_file="${EVAL_OUTPUT_DIR}/${model_name}_general_scores.txt"
    if [[ ! -s "${score_file}" ]]; then
      echo "[ERROR] Missing concise score file: ${score_file}" >&2
      exit 1
    fi
    {
      echo "${model_name}"
      cat "${score_file}"
      echo ""
    } >> "${combined_score_file}"
  done
  echo "[INFO] Concise scores: ${combined_score_file}"
fi

echo "[INFO] All evaluations finished successfully."
echo "[INFO] Results saved under: ${EVAL_OUTPUT_DIR}"
