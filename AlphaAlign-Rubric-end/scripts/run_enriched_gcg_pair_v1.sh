#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REQUIREMENTS_PATH="${SCRIPT_DIR}/requirements-safety-data-pipeline.txt"

DATASET_NAME="updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1"
PIPELINE_WORK="${PIPELINE_WORK:-$(dirname "${PROJECT_ROOT}")/safety_pipeline_work}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_ROOT}/Evaluation/updated_training_rubric_dataset_independent_v3}"
RUBRIC_BUILD_OUTPUT="${RUBRIC_BUILD_OUTPUT:-${PROJECT_ROOT}/Evaluation/${DATASET_NAME}}"
LOG_PATH="${PIPELINE_LOG_PATH:-${PIPELINE_WORK}/${DATASET_NAME}.log}"
PID_PATH="${PIPELINE_PID_PATH:-${PIPELINE_WORK}/${DATASET_NAME}.pid}"

usage() {
  cat <<EOF
Usage: bash scripts/run_enriched_gcg_pair_v1.sh [start|foreground|status|logs|stop]

Commands:
  start       Validate inputs and run the full pipeline in the background (default)
  foreground  Run the full pipeline in the current terminal
  status      Print one status snapshot
  logs        Follow the pipeline log; Ctrl-C stops viewing only
  stop        Send SIGTERM to the recorded pipeline process

Optional environment variables:
  SAFETY_PIPELINE_ENV_FILE  Absolute path to the external API env.yaml
  RUBRIC_PILOT_CONCURRENCY API concurrency, default 4
  RUBRIC_GENERATION_VALIDATION_ATTEMPTS Generation JSON/schema retries, default 3
  RUBRIC_VERIFICATION_VALIDATION_ATTEMPTS Verification/factuality retries, default 3
  RUBRIC_ADAPTIVE_REGENERATION_ATTEMPTS Targeted GCG/PAIR repairs, default 2
  RUBRIC_RUBRIC_REPAIR_ATTEMPTS Targeted source-rubric repairs, default 2
  RUBRIC_PIPELINE_MAX_ROUNDS Resume rounds before stopping, default 8
  RUBRIC_AUTO_INSTALL_DEPS Install missing Python dependencies, default 0
  PIPELINE_WORK             Plan/result/log directory
  SOURCE_DATA               Source independent_v3 dataset directory
  RUBRIC_BUILD_OUTPUT       New dataset output directory
EOF
}

read_pid() {
  if [ -f "${PID_PATH}" ]; then
    tr -d '[:space:]' < "${PID_PATH}"
  fi
}

is_running() {
  local pid
  pid="$(read_pid)"
  [ -n "${pid}" ] && kill -0 "${pid}" 2>/dev/null
}

find_env_file() {
  local candidates=()
  if [ -n "${SAFETY_PIPELINE_ENV_FILE:-}" ]; then
    candidates+=("${SAFETY_PIPELINE_ENV_FILE}")
  fi
  candidates+=(
    "$(dirname "${PROJECT_ROOT}")/agentic_reward/reward_system/rewards/utils/env.yaml"
    "$(dirname "$(dirname "${PROJECT_ROOT}")")/agentic_reward/reward_system/rewards/utils/env.yaml"
    "$(dirname "$(dirname "${PROJECT_ROOT}")")/code/rl_benchmark/agentic_bench_ruip_v2/agentic_reward/reward_system/rewards/utils/env.yaml"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -f "${candidate}" ]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

preflight() {
  if [ ! -d "${SOURCE_DATA}" ]; then
    echo "[ERROR] Source dataset not found: ${SOURCE_DATA}" >&2
    exit 1
  fi
  local split filename
  for split in train val; do
    for filename in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
      if [ ! -f "${SOURCE_DATA}/${split}/${filename}" ]; then
        echo "[ERROR] Source dataset file missing: ${SOURCE_DATA}/${split}/${filename}" >&2
        exit 1
      fi
    done
  done
  if [ -e "${RUBRIC_BUILD_OUTPUT}" ]; then
    echo "[ERROR] Output already exists: ${RUBRIC_BUILD_OUTPUT}" >&2
    echo "[ERROR] Move it aside or choose a new RUBRIC_BUILD_OUTPUT; it will not be overwritten." >&2
    exit 1
  fi
  if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] python3 is not available." >&2
    exit 1
  fi
  local missing_modules
  missing_modules="$(python3 -c 'import importlib.util; modules={"pandas":"pandas","pyarrow":"pyarrow","yaml":"PyYAML","requests":"requests"}; print(" ".join(package for module, package in modules.items() if importlib.util.find_spec(module) is None))')"
  if [ -n "${missing_modules}" ]; then
    if [ "${RUBRIC_AUTO_INSTALL_DEPS:-0}" = "1" ]; then
      echo "[INFO] Installing missing pipeline dependencies: ${missing_modules}"
      python3 -m pip install --user -r "${REQUIREMENTS_PATH}"
    else
      echo "[ERROR] Missing Python dependencies: ${missing_modules}" >&2
      echo "[ERROR] Install with: python3 -m pip install --user -r ${REQUIREMENTS_PATH}" >&2
      echo "[ERROR] Or rerun with RUBRIC_AUTO_INSTALL_DEPS=1." >&2
      exit 1
    fi
  fi
  PYTHONWARNINGS=ignore python3 -c 'import pandas, pyarrow, yaml, requests' >/dev/null
  local env_file
  if ! env_file="$(find_env_file)"; then
    echo "[ERROR] API env.yaml was not found." >&2
    echo "[ERROR] Run again with SAFETY_PIPELINE_ENV_FILE=/absolute/path/env.yaml" >&2
    exit 1
  fi
  export SAFETY_PIPELINE_ENV_FILE="${env_file}"
  mkdir -p "${PIPELINE_WORK}"
  PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/safety_pipeline_pycache" \
    python3 -m py_compile \
      "${PROJECT_ROOT}/scripts/safety_data_pipeline.py"
}

run_pipeline() {
  cd "${PROJECT_ROOT}"
  export PIPELINE_WORK SOURCE_DATA RUBRIC_BUILD_OUTPUT
  export RUBRIC_BUILD_DATASET=1
  export RUBRIC_PILOT_SCOPE=all
  export RUBRIC_PILOT_LIMIT=0
  export RUBRIC_REVIEW_FRACTION=1.0
  export RUBRIC_ADAPTIVE_SEEDS=120
  exec bash "${PROJECT_ROOT}/scripts/run_rubric_pilot.sh"
}

command="${1:-start}"
case "${command}" in
  start)
    if is_running; then
      echo "[INFO] Pipeline is already running: PID=$(read_pid)"
      echo "[INFO] Log: ${LOG_PATH}"
      exit 0
    fi
    preflight
    nohup bash "${SCRIPT_PATH}" foreground > "${LOG_PATH}" 2>&1 &
    pipeline_pid=$!
    printf '%s\n' "${pipeline_pid}" > "${PID_PATH}"
    echo "[INFO] Pipeline started: PID=${pipeline_pid}"
    echo "[INFO] Log: ${LOG_PATH}"
    echo "[INFO] Output: ${RUBRIC_BUILD_OUTPUT}"
    echo "[INFO] Status: bash scripts/run_enriched_gcg_pair_v1.sh status"
    ;;
  foreground)
    preflight
    run_pipeline
    ;;
  status)
    if is_running; then
      echo "[INFO] RUNNING: PID=$(read_pid)"
      echo "[INFO] Log: ${LOG_PATH}"
    else
      echo "[INFO] NOT RUNNING"
      if [ -f "${LOG_PATH}" ]; then
        tail -n 30 "${LOG_PATH}"
      fi
    fi
    ;;
  logs)
    if [ ! -f "${LOG_PATH}" ]; then
      echo "[ERROR] Log does not exist: ${LOG_PATH}" >&2
      exit 1
    fi
    tail -f "${LOG_PATH}"
    ;;
  stop)
    if ! is_running; then
      echo "[INFO] Pipeline is not running."
      exit 0
    fi
    pipeline_pid="$(read_pid)"
    kill "${pipeline_pid}"
    echo "[INFO] Sent SIGTERM to PID=${pipeline_pid}"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
