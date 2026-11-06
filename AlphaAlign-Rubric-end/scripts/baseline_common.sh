#!/usr/bin/env bash
# Shared Primus plumbing for the AlphaAlign and NSPO baseline launchers.
#
# Sourced by train_val_alphaalign.sh and train_val_nspo.sh. It provides the same
# service-startup, checkpoint-merge and evaluation stages that train_val.sh uses
# for RubricAlign, so the three methods are scored by identical code.
#
# The caller must set: RUBRIC_PROJECT_ROOT, CKPT_PATH, EVAL_OUTPUT_DIR,
# API_LOG_DIR, POLICY_MODEL_PATH, POLICY_MODEL_ID, METHOD_TAG.

normalize_bool() {
  local name="$1"
  local value="$2"
  case "${value}" in
    1|true|TRUE|yes|YES|on|ON) echo 1 ;;
    0|false|FALSE|no|NO|off|OFF) echo 0 ;;
    *)
      echo "[ERROR] ${name} must be a boolean value, got '${value}'." >&2
      exit 2
      ;;
  esac
}

require_positive_int() {
  local name="$1"
  local value="$2"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] ${name} must be a positive integer, got '${value}'." >&2
    exit 2
  fi
}

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

wait_for_api() {
  local port=$1
  local service_name=$2
  local max_retries=${3:-3000}
  local service_pid=${4:-}
  local service_log=${5:-}
  local retry_count=0

  echo "[INFO] Waiting for ${service_name} on port ${port} to start..."
  while [ "${retry_count}" -lt "${max_retries}" ]; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:${port}/v1/models" | grep -q "200"; then
      echo "[SUCCESS] ${service_name} is ready on port ${port}!"
      return 0
    fi
    if [ -n "${service_pid}" ] && ! kill -0 "${service_pid}" 2>/dev/null; then
      echo "[ERROR] ${service_name} process ${service_pid} exited before opening port ${port}." >&2
      if [ -n "${service_log}" ] && [ -f "${service_log}" ]; then
        echo "[ERROR] Last service log lines (${service_log}):" >&2
        tail -n 80 "${service_log}" >&2 || true
      fi
      return 1
    fi
    sleep 5
    retry_count=$((retry_count + 1))
    echo "[WAIT] ${service_name} not ready yet... (${retry_count}/${max_retries})"
  done
  echo "[ERROR] ${service_name} failed to start on port ${port} after ${max_retries} attempts." >&2
  return 1
}

wait_for_model() {
  local port=$1
  local service_name=$2
  local expected_model=$3
  local max_retries=${4:-3000}
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
  local max_retries=${2:-60}
  local retry_count=0
  while [ "${retry_count}" -lt "${max_retries}" ]; do
    if ! curl -fsS "http://localhost:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    retry_count=$((retry_count + 1))
  done
  echo "[ERROR] Port ${port} is still occupied." >&2
  return 1
}

# Validates that a Python interpreter can import the inference backend and read
# the model config before any replica is launched. Skipping this check turns a
# dependency problem into a silent crash after startup.
interpreter_ready() {
  local python_bin="$1"
  local model_path="$2"
  local label="$3"
  echo "[INFO] Checking ${label} interpreter readiness: ${python_bin}"
  if ! "${python_bin}" - "${model_path}" <<'PYCHECK'
import sys

model_path = sys.argv[1]
import vllm  # noqa: F401
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"[INFO] interpreter ready: vllm importable, config model_type={config.model_type}")
PYCHECK
  then
    echo "[ERROR] ${label} interpreter is not usable for ${model_path}." >&2
    return 1
  fi
  return 0
}

start_vllm_service() {
  # start_vllm_service <python_bin> <model_path> <served_name> <gpu_ids> <tp_size> <port> <log_file> [extra args...]
  local python_bin="$1"; shift
  local model_path="$1"; shift
  local served_name="$1"; shift
  local gpu_ids="$1"; shift
  local tp_size="$1"; shift
  local port="$1"; shift
  local log_file="$1"; shift

  # Only the PID goes to stdout: the caller captures it with $(...).
  echo "[INFO] Starting ${served_name}: GPUs=${gpu_ids}, TP=${tp_size}, port=${port}" >&2
  CUDA_VISIBLE_DEVICES="${gpu_ids}" nohup "${python_bin}" -m vllm.entrypoints.openai.api_server \
    --model "${model_path}" \
    --served-model-name "${served_name}" \
    --tensor-parallel-size "${tp_size}" \
    --port "${port}" \
    --dtype bfloat16 \
    "$@" \
    > "${log_file}" 2>&1 &
  echo $!
}

# =========================
# Checkpoint merge
# =========================
# merge_checkpoints <merge_backend: verl02|verl05>
# Populates MERGED_THIS_RUN, SAFETY_EVAL_MODELS and LAST_MERGED_MODEL.
# By default only the fixed mid-training, final, and training-selected best
# checkpoints are materialized/evaluated. This avoids evaluating every 30-step
# checkpoint after a threshold in a roughly 1,000-step job.
merge_checkpoints() {
  local merge_backend="$1"
  local merged_models_dir="${CKPT_PATH}/models"
  # safetensors uses filesystem operations that are unsupported by the Primus
  # OSS/FUSE mount (the observed failure is errno 95). Materialize the model on
  # node-local storage, then copy the complete result to persistent storage.
  local merge_staging_root="${MERGE_STAGING_ROOT:-${RAY_TMPDIR:-/tmp}/model_merge_staging}"
  local min_step="${FINAL_EVAL_MIN_STEP:-0}"
  local eval_steps_csv="${FINAL_SAFETY_EVAL_STEPS:-600,900,last,best}"

  if ! [[ "${min_step}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] FINAL_EVAL_MIN_STEP must be a non-negative integer." >&2
    exit 2
  fi

  mkdir -p "${merged_models_dir}" "${merge_staging_root}"
  MERGED_THIS_RUN=()
  SAFETY_EVAL_MODELS=()
  LAST_MERGED_MODEL=""
  local last_step=-1
  local last_standard_step=-1

  local ckpt_dirs=()
  # A read loop instead of mapfile keeps this usable on bash 3.x as well.
  local line
  while IFS= read -r line; do
    [ -n "${line}" ] && ckpt_dirs+=("${line}")
  done < <(find "${CKPT_PATH}" -maxdepth 1 -mindepth 1 -type d -name "global_step_*" | sort -V)
  if [ "${#ckpt_dirs[@]}" -eq 0 ]; then
    echo "[ERROR] No checkpoint directories found in ${CKPT_PATH}" >&2
    exit 1
  fi

  local line_name
  for ckpt_dir in "${ckpt_dirs[@]}"; do
    line_name="$(basename "${ckpt_dir}")"
    if [[ "${line_name}" =~ ^global_step_([0-9]+)$ ]] && [ "${BASH_REMATCH[1]}" -gt "${last_standard_step}" ]; then
      last_standard_step="${BASH_REMATCH[1]}"
    fi
  done
  if [ "${last_standard_step}" -lt 0 ]; then
    echo "[ERROR] No numbered global_step_<n> checkpoints found in ${CKPT_PATH}" >&2
    exit 1
  fi
  echo "[INFO] Final safety checkpoint selection: ${eval_steps_csv} (last=${last_standard_step})"

  local ckpt_dir step_name checkpoint_step actor_dir target_dir is_best_checkpoint is_selected
  local staging_dir merge_rc
  for ckpt_dir in "${ckpt_dirs[@]}"; do
    step_name="$(basename "${ckpt_dir}")"
    is_best_checkpoint=0
    if [[ "${step_name}" =~ ^global_step_([0-9]+)$ ]]; then
      checkpoint_step="${BASH_REMATCH[1]}"
    elif [[ "${step_name}" =~ ^global_step_best_([0-9]+)$ ]]; then
      checkpoint_step="${BASH_REMATCH[1]}"
      is_best_checkpoint=1
    else
      echo "[WARN] Cannot parse checkpoint step from ${step_name}; skipping."
      continue
    fi

    is_selected=0
    if [[ ",${eval_steps_csv}," == *",${checkpoint_step},"* ]]; then
      is_selected=1
    fi
    if [ "${checkpoint_step}" = "${last_standard_step}" ] && [[ ",${eval_steps_csv}," == *",last,"* ]]; then
      is_selected=1
    fi
    if [ "${is_best_checkpoint}" = "1" ] && [[ ",${eval_steps_csv}," == *",best,"* ]]; then
      is_selected=1
    fi
    if [ "${is_selected}" != "1" ]; then
      continue
    fi

    actor_dir="${ckpt_dir}/actor"
    if [ ! -d "${actor_dir}" ]; then
      echo "[WARN] Missing actor dir for ${ckpt_dir}; skipping."
      continue
    fi

    target_dir="${merged_models_dir}/${POLICY_MODEL_ID}-${METHOD_TAG}-${step_name}"

    if [ -f "${target_dir}/config.json" ]; then
      echo "[SKIP] Already merged: ${step_name}"
    else
      echo "[INFO] Merging ${step_name} -> ${target_dir}"
      staging_dir="$(mktemp -d "${merge_staging_root%/}/${METHOD_TAG}-${step_name}.XXXXXX")"
      echo "[INFO] Node-local merge staging: ${staging_dir}"
      merge_rc=0
      case "${merge_backend}" in
        verl02)
          python3 "${RUBRIC_PROJECT_ROOT}/scripts/model_merger.py" \
            --backend fsdp \
            --hf_model_path "${POLICY_MODEL_PATH}" \
            --local_dir "${actor_dir}" \
            --target_dir "${staging_dir}" || merge_rc=$?
          ;;
        verl05)
          ( cd "${NSPO_VERL_ROOT}" && "${NSPO_PYTHON_BIN}" "${NSPO_ROOT}/script/merge_verl_ckpt.py" \
              --local_dir "${actor_dir}" \
              --target_dir "${staging_dir}" \
              --hf_model_config_path "${POLICY_MODEL_PATH}" ) || merge_rc=$?
          ;;
        *)
          echo "[ERROR] Unknown merge backend: ${merge_backend}" >&2
          exit 2
          ;;
      esac
      if [ "${merge_rc}" -eq 0 ]; then
        python3 "${RUBRIC_PROJECT_ROOT}/scripts/move_tokenizer.py" \
          --source "${POLICY_MODEL_PATH}" \
          --destination "${staging_dir}" || merge_rc=$?
      fi
      if [ "${merge_rc}" -ne 0 ] || [ ! -f "${staging_dir}/config.json" ]; then
        rm -rf "${staging_dir}"
        echo "[ERROR] Merge produced no config.json for ${step_name}" >&2
        return 1
      fi
      mkdir -p "${target_dir}"
      cp -a "${staging_dir}/." "${target_dir}/"
      rm -rf "${staging_dir}"
      if [ ! -f "${target_dir}/config.json" ]; then
        echo "[ERROR] Publishing the merged model failed for ${step_name}: ${target_dir}/config.json is missing." >&2
        return 1
      fi
      echo "[DONE] Merged ${step_name}"
    fi

    MERGED_THIS_RUN+=("${target_dir}")
    if [ "${checkpoint_step}" -ge "${min_step}" ] || [ "${is_best_checkpoint}" = "1" ]; then
      SAFETY_EVAL_MODELS+=("${target_dir}")
    fi
    # Keep the general-evaluation fallback pointed at a regular checkpoint.
    # A best copy may have the same step and must not replace the standard
    # checkpoint when best-checkpoint materialization was incomplete.
    if [ "${is_best_checkpoint}" = "0" ] && [ "${checkpoint_step}" -gt "${last_step}" ]; then
      last_step="${checkpoint_step}"
      LAST_MERGED_MODEL="${target_dir}"
    fi
  done

  # Recover from merged models created by an earlier invocation, and from
  # layouts where POLICY_MODEL_ID contains a slash.  The parent directory of
  # config.json is either global_step_<n> or ...-global_step_<n>.
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
      if [ -n "${candidate_step}" ] && [ "${candidate_step}" -gt "${last_step}" ]; then
        LAST_MERGED_MODEL="${candidate_dir}"
        last_step="${candidate_step}"
      fi
    done < <(find "${merged_models_dir}" -type f -name config.json | sort -V)
  fi

  if [ "${#MERGED_THIS_RUN[@]}" -eq 0 ]; then
    echo "[ERROR] No models were merged successfully." >&2
    exit 1
  fi
  if [ -z "${LAST_MERGED_MODEL}" ] || [ ! -f "${LAST_MERGED_MODEL}/config.json" ]; then
    echo "[ERROR] No valid standard merged checkpoint is available for general evaluation." >&2
    exit 1
  fi
  validate_baseline_final_coverage || return $?
  echo "[INFO] Merged ${#MERGED_THIS_RUN[@]} checkpoint(s); latest standard checkpoint is ${LAST_MERGED_MODEL} (step ${last_step})."
}

# =========================
# Safety evaluation
# =========================
# run_safety_eval <think_flag: True|False>
run_safety_eval() {
  local think_flag="$1"
  local safety_dir="${RUBRIC_PROJECT_ROOT}/Evaluation/safety"

  if [ "${#SAFETY_EVAL_MODELS[@]}" -eq 0 ]; then
    echo "[ERROR] No merged checkpoint satisfies FINAL_EVAL_MIN_STEP=${FINAL_EVAL_MIN_STEP:-0}." >&2
    exit 1
  fi
  if [ ! -d "${safety_dir}" ]; then
    echo "[ERROR] Safety evaluation directory not found: ${safety_dir}" >&2
    exit 1
  fi

  echo "[INFO] ============================================="
  echo "[INFO] Starting Safety Evaluation (Think=${think_flag})"
  echo "[INFO] ============================================="

  local pushed_dir
  pushed_dir="$(pwd)"
  cd "${safety_dir}"

  local eval_gpu_ids="${SAFETY_EVAL_GPU_IDS:-${TRAIN_GPU_IDS%%,*}}"
  if [[ "${eval_gpu_ids}" == *,* ]]; then
    echo "[ERROR] SAFETY_EVAL_GPU_IDS must name exactly one GPU; generate.py has TP=1." >&2
    return 2
  fi
  local total_started_at=$SECONDS
  echo "[INFO] Safety generation GPU: ${eval_gpu_ids} (one vLLM replica, TP=1)"

  local model_dir model_name log_file summary_file model_started_at
  for model_dir in "${SAFETY_EVAL_MODELS[@]}"; do
    model_name="$(basename "${model_dir}")"
    model_started_at=$SECONDS
    echo "[EVAL] Processing: ${model_name}"

    log_file="${EVAL_OUTPUT_DIR}/${model_name}_eval.log"
    summary_file="${EVAL_OUTPUT_DIR}/${model_name}_summary.txt"
    mkdir -p "$(dirname "${log_file}")"

    {
      echo "========================================"
      echo "Model: ${model_name}"
      echo "Method: ${METHOD_TAG}"
      echo "Think: ${think_flag}"
      echo "Time: $(date)"
      echo "========================================"
    } > "${summary_file}"

    echo "[GEN] Running generate.py..."
    CUDA_VISIBLE_DEVICES="${eval_gpu_ids}" VLLM_WORKER_MULTIPROC_METHOD=spawn python3 generate.py \
      --model_id "${model_dir}" \
      --Think "${think_flag}" 2>&1 | tee -a "${log_file}"

    echo "[EVAL] Running eval_answer.py..."
    if ! PYTHONUNBUFFERED=1 python3 eval_answer.py --model_id "${model_name}" 2>&1 \
      | tee -a "${log_file}" \
      | tee -a "${summary_file}"; then
      echo "[ERROR] eval_answer.py failed for ${model_name}." >&2
      echo "[ERROR] Tail of ${log_file}:" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi

    echo "[EVAL] Running XSTest over-refusal evaluation..."
    if ! PYTHONUNBUFFERED=1 python3 overrefusal.py \
      --model_id "${model_name}" \
      --dataset_name xstest \
      --require-complete \
      2>&1 | tee -a "${log_file}" | tee -a "${summary_file}"; then
      echo "[ERROR] overrefusal.py failed for ${model_name}." >&2
      echo "[ERROR] Tail of ${log_file}:" >&2
      tail -n 80 "${log_file}" >&2 || true
      return 1
    fi

    echo "[TIME] ${model_name} safety evaluation: $((SECONDS - model_started_at))s" | tee -a "${log_file}" -a "${summary_file}"
    echo "[DONE] ${model_name}"
  done

  echo "[TIME] Total safety evaluation: $((SECONDS - total_started_at))s for ${#SAFETY_EVAL_MODELS[@]} model(s)."

  cd "${pushed_dir}"
}

# =========================
# MMLU / GSM8K evaluation
# =========================
# run_general_eval <model_dir>
run_general_eval() {
  local model_dir="$1"
  local general_eval_script="${RUBRIC_PROJECT_ROOT}/Evaluation/lm_eval_last3_general.sh"

  if [ ! -f "${general_eval_script}" ]; then
    echo "[ERROR] General evaluation script not found: ${general_eval_script}" >&2
    exit 1
  fi
  if [ -z "${model_dir}" ] || [ ! -f "${model_dir}/config.json" ]; then
    echo "[ERROR] General evaluation target is not a merged model: ${model_dir}" >&2
    exit 1
  fi

  local models_dir model_name
  models_dir="$(dirname "${model_dir}")"
  model_name="$(basename "${model_dir}")"

  local gpu_ids
  if [ -n "${FINAL_GENERAL_EVAL_GPU_IDS:-}" ]; then
    gpu_ids="${FINAL_GENERAL_EVAL_GPU_IDS}"
  else
    gpu_ids="${ALL_JOB_GPU_IDS}"
  fi

  local gpu_array=()
  IFS=',' read -r -a gpu_array <<< "${gpu_ids}"
  local gpu_count="${#gpu_array[@]}"
  local tp_size="${FINAL_GENERAL_EVAL_TP_SIZE:-1}"
  local replicas
  if [ "${FINAL_GENERAL_EVAL_BACKEND}" = "vllm" ]; then
    replicas="${FINAL_GENERAL_EVAL_REPLICAS:-1}"
  else
    replicas=1
  fi
  require_positive_int FINAL_GENERAL_EVAL_TP_SIZE "${tp_size}"
  require_positive_int FINAL_GENERAL_EVAL_REPLICAS "${replicas}"
  if [ $((tp_size * replicas)) -gt "${gpu_count}" ]; then
    echo "[ERROR] General evaluation requests $((tp_size * replicas)) GPUs, but only ${gpu_count} were assigned." >&2
    exit 2
  fi

  echo "[INFO] ============================================="
  echo "[INFO] Starting MMLU and GSM8K Evaluation"
  echo "[INFO] Model: ${model_dir}"
  echo "[INFO] Backend: ${FINAL_GENERAL_EVAL_BACKEND}"
  echo "[INFO] GPUs: ${gpu_ids}; replicas=${replicas}; TP=${tp_size}"
  echo "[INFO] ============================================="
  local general_eval_started_at=$SECONDS

  if [ "${FINAL_GENERAL_EVAL_BACKEND}" = "vllm" ]; then
    ray stop --force 2>/dev/null || true
  fi

  CUDA_VISIBLE_DEVICES="${gpu_ids}" \
  EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR}" \
  LM_EVAL_DP_SIZE="${replicas}" \
  LM_EVAL_TP_SIZE="${tp_size}" \
  LM_EVAL_EXTRA_MODEL_ARGS="${LM_EVAL_EXTRA_MODEL_ARGS:-max_model_len=4096}" \
  bash "${general_eval_script}" \
    --ckpt-path "${CKPT_PATH}" \
    --models-dir "${models_dir}" \
    --glob "${model_name}" \
    --num-models 1 \
    --tasks "${FINAL_GENERAL_EVAL_TASKS:-gsm8k_cot_modelscope,mmlu_modelscope}" \
    --backend "${FINAL_GENERAL_EVAL_BACKEND}" \
    --device "${FINAL_GENERAL_EVAL_DEVICE}"
  echo "[TIME] Total general evaluation: $((SECONDS - general_eval_started_at))s"
}

# Match shared safety/general coverage without introducing a synthetic best.
# Validate all models before scoring any of them. Only a regular numbered model
# can satisfy a required numbered checkpoint; a separate best copy is distinct.
validate_baseline_final_coverage() {
  [ -n "${BASELINE_REQUIRED_FINAL_STEPS:-}" ] || return 0
  if [ "${BASELINE_GENERAL_EVAL_SCOPE:-last}" != "selected" ]; then
    echo "[ERROR] Required matched coverage needs BASELINE_GENERAL_EVAL_SCOPE=selected." >&2
    return 2
  fi
  if ! [[ "${BASELINE_REQUIRED_FINAL_STEPS}" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]]; then
    echo "[ERROR] BASELINE_REQUIRED_FINAL_STEPS must be comma-separated positive integers." >&2
    return 2
  fi
  local required_steps=() required model found expected
  IFS=',' read -r -a required_steps <<< "${BASELINE_REQUIRED_FINAL_STEPS}"
  for required in "${required_steps[@]}"; do
    found=0
    expected="${CKPT_PATH}/models/${POLICY_MODEL_ID}-${METHOD_TAG}-global_step_${required}"
    for model in "${SAFETY_EVAL_MODELS[@]}"; do
      if [ "${model}" = "${expected}" ] && [ -f "${model}/config.json" ]; then
        found=1
      fi
    done
    if [ "${found}" != "1" ]; then
      echo "[ERROR] Required final checkpoint ${required} is missing from selected safety/general coverage." >&2
      return 1
    fi
  done
}

# Preserve historical last-only behavior by default. Matched comparisons opt
# into selected coverage, including a native best only if it actually exists.
run_baseline_general_eval() {
  local scope="${BASELINE_GENERAL_EVAL_SCOPE:-last}" model
  local targets=()
  case "${scope}" in
    last)
      model="${LAST_MERGED_MODEL:-}"
      if [ -n "${BASELINE_GENERAL_EVAL_STEP:-}" ]; then
        require_positive_int BASELINE_GENERAL_EVAL_STEP "${BASELINE_GENERAL_EVAL_STEP}"
        model="${CKPT_PATH}/models/${POLICY_MODEL_ID}-${METHOD_TAG}-global_step_${BASELINE_GENERAL_EVAL_STEP}"
      fi
      # Required matched coverage must not silently become last-only general eval.
      if [ -n "${BASELINE_REQUIRED_FINAL_STEPS:-}" ]; then
        echo "[ERROR] Required matched coverage needs BASELINE_GENERAL_EVAL_SCOPE=selected." >&2
        return 2
      fi
      targets+=("${model}")
      ;;
    selected)
      if [ -n "${BASELINE_GENERAL_EVAL_STEP:-}" ]; then
        echo "[ERROR] BASELINE_GENERAL_EVAL_STEP conflicts with selected coverage." >&2
        return 2
      fi
      if [ "${#SAFETY_EVAL_MODELS[@]}" -eq 0 ]; then
        echo "[ERROR] No selected models are available for general evaluation." >&2
        return 1
      fi
      validate_baseline_final_coverage || return $?
      targets=("${SAFETY_EVAL_MODELS[@]}")
      ;;
    *)
      echo "[ERROR] BASELINE_GENERAL_EVAL_SCOPE must be last or selected." >&2
      return 2
      ;;
  esac
  for model in "${targets[@]}"; do
    if [ -z "${model}" ] || [ ! -f "${model}/config.json" ]; then
      echo "[ERROR] General evaluation target is not a merged model: ${model}" >&2
      return 1
    fi
  done
  for model in "${targets[@]}"; do
    run_general_eval "${model}" || return $?
  done
}
