#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Optional base env
if [ -f "${PROJECT_ROOT}/verl/base.sh" ]; then
  # shellcheck disable=SC1091
  source "${PROJECT_ROOT}/verl/base.sh"
fi

source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"

# =========================
# Train/val 基础设置（保留 train_val.sh 关键设置）
# =========================
export VERL_HOME="${VERL_HOME:-${PROJECT_ROOT}}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
# The outer Primus launcher may select a fresh run directory when the mounted
# checkpoint root already contains an older run. Respect that explicit value;
# only fall back to PRIMUS_SAVE_CHECKPOINT_DIR when CKPT_PATH is unset.
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/qwen25_3B_instruct}}"
export API_LOG_DIR="${API_LOG_DIR:-/root/logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/eval_outputs}"
export RUBRIC_EVOLVE_DATASET_DIR="${RUBRIC_EVOLVE_DATASET_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-${VERL_HOME}}/data}"
export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-${RUBRIC_RUN_DIR}/online_rubric_evolution}"

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
# REWARD_MODE (reward_multiple is retained as an alias for existing launch commands):
#   legacy                     -> original configurable Guard reward
#   guard_multiplicative       -> Guard pass/fail multiplier
#   guard_safety_alignment     -> Guard-supervised safety alignment (default)
#   rubric_safety_gated        -> rubric-only hard safety gate, no Guard API
#   guard_soft_calibrated      -> Guard-supervised soft safety calibration
#   rubric_refusal_calibrated  -> rubric-only refusal calibration, no Guard API
#   rubric_safe_helpful_balanced -> profile-aware balanced rubric reward, no Guard API
# RUBRIC_MULTIPLE_FAIL_VALUE:
#   only effective with guard_multiplicative
#   pass multiplier fixed to +1; fail multiplier is this value (e.g., -1 or 0)
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
    echo "[ERROR] Use legacy, guard_multiplicative, guard_safety_alignment, rubric_safety_gated, guard_soft_calibrated, rubric_refusal_calibrated, or rubric_safe_helpful_balanced." >&2
    exit 2
    ;;
esac
export REWARD_MODE
export reward_multiple="${REWARD_MODE}"
export RUBRIC_MULTIPLE_FAIL_VALUE="${RUBRIC_MULTIPLE_FAIL_VALUE:--1}"
export RUBRIC_SAFE_PRED_REWARD_MODE="${RUBRIC_SAFE_PRED_REWARD_MODE:-0}"
export RUBRIC_FORMAT_REQUIRE_THINK="${RUBRIC_FORMAT_REQUIRE_THINK:-1}"
export RUBRIC_LENGTH_PENALTY_WEIGHT="${RUBRIC_LENGTH_PENALTY_WEIGHT:-0.1}"
export RUBRIC_LENGTH_TOKENIZER_PATH="${RUBRIC_LENGTH_TOKENIZER_PATH:-${JUDGE_MODEL_PATH}}"
export RUBRIC_BALANCED_HARM_BOUNDARY_GAMMA="${RUBRIC_BALANCED_HARM_BOUNDARY_GAMMA:-2.0}"
export RUBRIC_BALANCED_HARM_BOUNDARY_GAP_PENALTY="${RUBRIC_BALANCED_HARM_BOUNDARY_GAP_PENALTY:-0.50}"
export RUBRIC_BALANCED_ADAPTIVE_BOUNDARY_GAP_PENALTY="${RUBRIC_BALANCED_ADAPTIVE_BOUNDARY_GAP_PENALTY:-0.70}"
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

# Online rubric evolution. It remains off for existing launch commands. When
# enabled, already-promoted rules score every batch, while new proposals are
# generated only on the configured step schedule and activate on the next batch.
export RUBRIC_EVOLVE="${RUBRIC_EVOLVE:-0}"
export RUBRIC_EVOLVE_START_EPOCH="${RUBRIC_EVOLVE_START_EPOCH:-2}"
export RUBRIC_EVOLVE_START_STEP="${RUBRIC_EVOLVE_START_STEP:-0}"
export RUBRIC_EVOLVE_INTERVAL_STEPS="${RUBRIC_EVOLVE_INTERVAL_STEPS:-120}"
export RUBRIC_EVOLVE_STEPS="${RUBRIC_EVOLVE_STEPS:-}"
export RUBRIC_EVOLVE_SCOPE="${RUBRIC_EVOLVE_SCOPE:-profile}"
export RUBRIC_EVOLVE_PROVIDER="${RUBRIC_EVOLVE_PROVIDER:-auto}"
export RUBRIC_EVOLVE_API_MODEL="${RUBRIC_EVOLVE_API_MODEL:-${RUBRIC_JUDGE_LOCAL_API_MODEL:-qwen3_32B_instruct}}"
export RUBRIC_EVOLVE_API_AUTH_FILE="${RUBRIC_EVOLVE_API_AUTH_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}"
export RUBRIC_EVOLVE_MAX_TRAJECTORIES="${RUBRIC_EVOLVE_MAX_TRAJECTORIES:-16}"
export RUBRIC_EVOLVE_MIN_LABEL_COUNT="${RUBRIC_EVOLVE_MIN_LABEL_COUNT:-3}"
export RUBRIC_EVOLVE_MIN_GAP="${RUBRIC_EVOLVE_MIN_GAP:-0.33}"
export RUBRIC_EVOLVE_BANK_MAX_RULES="${RUBRIC_EVOLVE_BANK_MAX_RULES:-4}"
export RUBRIC_EVOLVE_MAX_PROFILES_PER_STEP="${RUBRIC_EVOLVE_MAX_PROFILES_PER_STEP:-4}"
export RUBRIC_EVOLVE_MIN_RULES="${RUBRIC_EVOLVE_MIN_RULES:-0}"
export RUBRIC_EVOLVE_MAX_RULES="${RUBRIC_EVOLVE_MAX_RULES:-2}"
export RUBRIC_EVOLVE_TOTAL_MAX_RULES="${RUBRIC_EVOLVE_TOTAL_MAX_RULES:-4}"
export RUBRIC_EVOLVE_TWO_STAGE="${RUBRIC_EVOLVE_TWO_STAGE:-1}"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE:-0}"
export RUBRIC_EVOLVE_VARIANCE_START_STEP="${RUBRIC_EVOLVE_VARIANCE_START_STEP:-${RUBRIC_EVOLVE_START_STEP}}"
export RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS="${RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS:-8}"
export RUBRIC_EVOLVE_VARIANCE_ALPHA="${RUBRIC_EVOLVE_VARIANCE_ALPHA:-0.4}"
export RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA="${RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA:-0.3}"
export RUBRIC_EVOLVE_VARIANCE_EMA_DECAY="${RUBRIC_EVOLVE_VARIANCE_EMA_DECAY:-0.9}"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN:-0.5}"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX:-1.5}"
export RUBRIC_EVOLVE_DIRECT="${RUBRIC_EVOLVE_DIRECT:-1}"
export RUBRIC_EVOLVE_PRINT="${RUBRIC_EVOLVE_PRINT:-1}"
export RUBRIC_EVOLVE_TRAJ_MAX_TOKENS="${RUBRIC_EVOLVE_TRAJ_MAX_TOKENS:-400}"
export RUBRIC_EVOLVE_TOKENIZER_PATH="${RUBRIC_EVOLVE_TOKENIZER_PATH:-${JUDGE_MODEL_PATH}}"

# =========================
# Judge / Guard API 设置（默认适配 Qwen/Qwen3Guard-Gen-8B）
# =========================
export RUBRIC_JUDGE_BACKEND="${RUBRIC_JUDGE_BACKEND:-local_api}"
export RUBRIC_JUDGE_LOCAL_API_BASE="${RUBRIC_JUDGE_LOCAL_API_BASE:-http://127.0.0.1:8000/v1}"
export RUBRIC_JUDGE_LOCAL_API_BASES="${RUBRIC_JUDGE_LOCAL_API_BASES:-}"
export RUBRIC_JUDGE_LOAD_BALANCE="${RUBRIC_JUDGE_LOAD_BALANCE:-sticky}"
export RUBRIC_JUDGE_LOCAL_API_KEY="${RUBRIC_JUDGE_LOCAL_API_KEY:-EMPTY}"
export RUBRIC_JUDGE_LOCAL_API_MODEL="${RUBRIC_JUDGE_LOCAL_API_MODEL:-qwen3_32B_instruct}"
export RUBRIC_JUDGE_GRANULARITY="${RUBRIC_JUDGE_GRANULARITY:-all_rules}"
export RUBRIC_JUDGE_VLLM_MAX_MODEL_LEN="${RUBRIC_JUDGE_VLLM_MAX_MODEL_LEN:-4096}"
export RUBRIC_JUDGE_MAX_NEW_TOKENS="${RUBRIC_JUDGE_MAX_NEW_TOKENS:-512}"
export RUBRIC_JUDGE_API_CONCURRENCY="${RUBRIC_JUDGE_API_CONCURRENCY:-64}"
export RUBRIC_JUDGE_API_LOG_ENABLE="${RUBRIC_JUDGE_API_LOG_ENABLE:-1}"
export RUBRIC_GUARD_API_CONCURRENCY="${RUBRIC_GUARD_API_CONCURRENCY:-64}"
export RUBRIC_JUDGE_DEBUG="${RUBRIC_JUDGE_DEBUG:-0}"
export RUBRIC_JUDGE_DEBUG_EVERY="${RUBRIC_JUDGE_DEBUG_EVERY:-50}"
export RUBRIC_JUDGE_API_MAX_RETRIES="${RUBRIC_JUDGE_API_MAX_RETRIES:-3}"
export RUBRIC_JUDGE_API_RETRY_BACKOFF="${RUBRIC_JUDGE_API_RETRY_BACKOFF:-1.0}"
export RUBRIC_JUDGE_API_RETRY_BACKOFF_MAX="${RUBRIC_JUDGE_API_RETRY_BACKOFF_MAX:-10.0}"

export RUBRIC_GUARD_ENABLE="${RUBRIC_GUARD_ENABLE:-${RUBRIC_LLAMAGUARD_ENABLE:-1}}"
export RUBRIC_GUARD_BATCH_SIZE="${RUBRIC_GUARD_BATCH_SIZE:-${RUBRIC_LLAMAGUARD_BATCH_SIZE:-8}}"
export RUBRIC_GUARD_MAX_NEW_TOKENS="${RUBRIC_GUARD_MAX_NEW_TOKENS:-${RUBRIC_LLAMAGUARD_MAX_NEW_TOKENS:-256}}"
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
export ROLLOUT_TRACE_ENABLE="${ROLLOUT_TRACE_ENABLE:-1}"
EVAL_OUTPUT_PARENT="$(dirname "${EVAL_OUTPUT_DIR}")"
export ROLLOUT_TRACE_DIR="${ROLLOUT_TRACE_DIR:-${EVAL_OUTPUT_PARENT}/rollout_traces}"
export ROLLOUT_TRACE_MAX_PER_STEP="${ROLLOUT_TRACE_MAX_PER_STEP:-0}"
mkdir -p "${ROLLOUT_TRACE_DIR}"

# =========================
# 数据集按 mode 自动切换
# =========================
training_data_version_raw="${TRAINING_DATA_VERSION:-origin}"
training_data_version_raw="$(tr '[:upper:]-' '[:lower:]_' <<< "${training_data_version_raw}")"
case "${training_data_version_raw}" in
  origin|legacy|old) TRAINING_DATA_VERSION="origin" ;;
  new|balanced) TRAINING_DATA_VERSION="new" ;;
  new_v2|verified|verified_v2) TRAINING_DATA_VERSION="new_v2" ;;
  new_v3|independent|independent_v3) TRAINING_DATA_VERSION="new_v3" ;;
  *)
    echo "[ERROR] Unsupported TRAINING_DATA_VERSION: ${training_data_version_raw}" >&2
    echo "[ERROR] Use origin, new, new_v2, or new_v3." >&2
    exit 2
    ;;
esac
export TRAINING_DATA_VERSION
if [ -z "${TRAIN_MAX_PROMPT_LENGTH:-}" ]; then
  if [ "${TRAINING_DATA_VERSION}" = "new_v3" ]; then
    TRAIN_MAX_PROMPT_LENGTH=1536
  else
    TRAIN_MAX_PROMPT_LENGTH=1024
  fi
fi
export TRAIN_MAX_PROMPT_LENGTH
if [[ "${TRAINING_DATA_VERSION}" =~ ^new(_v[23])?$ ]] && [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" = "1" ]; then
  echo "[ERROR] TRAINING_DATA_VERSION=${TRAINING_DATA_VERSION} is not compatible with RUBRIC_SAFE_PRED_REWARD_MODE=1." >&2
  exit 2
fi

if [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" = "1" ]; then
  dataset_dir_name="Rubric_pipline_with_prediction_new"
elif [ "${TRAINING_DATA_VERSION}" = "new_v3" ]; then
  # The v3 comparison runs use the completed corpus: the original independent
  # split plus the 120 GCG and 120 PAIR examples. An older corpus can still be
  # selected explicitly with RUBRIC_DATASET_DIR_NAME or FILE_PATH.
  dataset_dir_name="${RUBRIC_DATASET_DIR_NAME:-updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1}"
elif [ "${TRAINING_DATA_VERSION}" = "new_v2" ]; then
  dataset_dir_name="updated_training_rubric_dataset_verified_v2"
elif [ "${TRAINING_DATA_VERSION}" = "new" ]; then
  dataset_dir_name="updated_training_rubric_dataset_balanced"
else
  dataset_dir_name="updated_training_rubric_dataset_v3"
fi

if [ -z "${FILE_PATH:-}" ]; then
  dataset_candidates=()
  if [ -n "${RUBRIC_DATASET_ROOT:-}" ]; then
    dataset_candidates+=("${RUBRIC_DATASET_ROOT}/${dataset_dir_name}")
  fi
  dataset_candidates+=("${PROJECT_ROOT}/Evaluation/${dataset_dir_name}")
  if [ -n "${PRIMUS_OSS_WORKSPACE_DIR:-}" ]; then
    dataset_candidates+=(
      "${PRIMUS_OSS_WORKSPACE_DIR}/datasets/${dataset_dir_name}"
      "${PRIMUS_OSS_WORKSPACE_DIR}/data/${dataset_dir_name}"
      "${PRIMUS_OSS_WORKSPACE_DIR}/${dataset_dir_name}"
    )
  fi

  for dataset_candidate in "${dataset_candidates[@]}"; do
    if [[ "${TRAINING_DATA_VERSION}" =~ ^new(_v[23])?$ ]] && [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" != "1" ]; then
      dataset_candidate_is_usable=1
      for dataset_split in train val; do
        for dataset_file in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
          if [ ! -f "${dataset_candidate}/${dataset_split}/${dataset_file}" ]; then
            dataset_candidate_is_usable=0
            break 2
          fi
        done
      done
    elif [ -d "${dataset_candidate}" ]; then
      dataset_candidate_is_usable=1
    else
      dataset_candidate_is_usable=0
    fi
    if [ "${dataset_candidate_is_usable}" = "1" ]; then
      FILE_PATH="${dataset_candidate}"
      break
    fi
  done
  FILE_PATH="${FILE_PATH:-${dataset_candidates[0]}}"
fi

if [ ! -d "${FILE_PATH}" ]; then
  echo "[ERROR] FILE_PATH not found: ${FILE_PATH}" >&2
  echo "[ERROR] On Primus, set RUBRIC_DATASET_ROOT to the mounted dataset parent, or FILE_PATH to the dataset directory." >&2
  exit 1
fi

if [[ "${TRAINING_DATA_VERSION}" =~ ^new(_v[23])?$ ]] && [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" != "1" ]; then
  TRAIN_DATA_PATH="${TRAIN_FILE_PATH:-${FILE_PATH}/train}"
  VAL_DATA_PATH="${VAL_FILE_PATH:-${FILE_PATH}/val}"
else
  TRAIN_DATA_PATH="${TRAIN_FILE_PATH:-${FILE_PATH}}"
  VAL_DATA_PATH="${VAL_FILE_PATH:-${FILE_PATH}}"
fi
if [ ! -d "${TRAIN_DATA_PATH}" ] || [ ! -d "${VAL_DATA_PATH}" ]; then
  echo "[ERROR] Dataset split missing: TRAIN_DATA_PATH=${TRAIN_DATA_PATH}, VAL_DATA_PATH=${VAL_DATA_PATH}" >&2
  echo "[ERROR] On Primus, upload the complete dataset (including train/ and val/) and set RUBRIC_DATASET_ROOT or FILE_PATH." >&2
  exit 1
fi
if [[ "${TRAINING_DATA_VERSION}" =~ ^new(_v[23])?$ ]] && [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" != "1" ]; then
  for dataset_file in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
    if [ ! -f "${TRAIN_DATA_PATH}/${dataset_file}" ] || [ ! -f "${VAL_DATA_PATH}/${dataset_file}" ]; then
      echo "[ERROR] Dataset file missing from train or val split: ${dataset_file}" >&2
      echo "[ERROR] TRAIN_DATA_PATH=${TRAIN_DATA_PATH}, VAL_DATA_PATH=${VAL_DATA_PATH}" >&2
      exit 1
    fi
  done
fi
SOURCE_TRAIN_DATA_PATH="${TRAIN_DATA_PATH}"
SOURCE_VAL_DATA_PATH="${VAL_DATA_PATH}"

train_prompt_version_was_set="${TRAIN_PROMPT_VERSION+x}"
if [ "${RUBRIC_SAFE_PRED_REWARD_MODE}" = "1" ]; then
  if [ -n "${train_prompt_version_was_set}" ]; then
    echo "[ERROR] TRAIN_PROMPT_VERSION is only supported when RUBRIC_SAFE_PRED_REWARD_MODE=0." >&2
    echo "[ERROR] Prediction-mode datasets require a separate <prediction> prompt template." >&2
    exit 2
  fi
  export TRAIN_PROMPT_VERSION="dataset_embedded"
else
  prompt_version_raw="${TRAIN_PROMPT_VERSION:-structured_steps}"
  prompt_version_raw="$(tr '[:upper:]-' '[:lower:]_' <<< "${prompt_version_raw}")"
  case "${prompt_version_raw}" in
    legacy|legacy_compact|old) TRAIN_PROMPT_VERSION="legacy_compact" ;;
    structured|structured_steps|new) TRAIN_PROMPT_VERSION="structured_steps" ;;
    *)
      echo "[ERROR] Unsupported TRAIN_PROMPT_VERSION: ${prompt_version_raw}" >&2
      echo "[ERROR] Use legacy_compact or structured_steps." >&2
      exit 2
      ;;
  esac
  export TRAIN_PROMPT_VERSION

  prompt_cache_root="${RUBRIC_PROMPT_CACHE_DIR:-${RAY_TMPDIR}/rubric_prompt_datasets}"
  prompt_dataset_dir="${prompt_cache_root}/${TRAINING_DATA_VERSION}/${TRAIN_PROMPT_VERSION}"
  python3 "${PROJECT_ROOT}/scripts/prepare_rubric_prompt_dataset.py" \
    --input-dir "${TRAIN_DATA_PATH}" \
    --output-dir "${prompt_dataset_dir}/train" \
    --version "${TRAIN_PROMPT_VERSION}"
  if [ "${TRAIN_DATA_PATH}" = "${VAL_DATA_PATH}" ]; then
    TRAIN_DATA_PATH="${prompt_dataset_dir}/train"
    VAL_DATA_PATH="${TRAIN_DATA_PATH}"
  else
    python3 "${PROJECT_ROOT}/scripts/prepare_rubric_prompt_dataset.py" \
      --input-dir "${VAL_DATA_PATH}" \
      --output-dir "${prompt_dataset_dir}/val" \
      --version "${TRAIN_PROMPT_VERSION}"
    TRAIN_DATA_PATH="${prompt_dataset_dir}/train"
    VAL_DATA_PATH="${prompt_dataset_dir}/val"
  fi
fi

export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-$TRAIN_DATA_PATH}"

python3 "${PROJECT_ROOT}/scripts/run_evidence.py" snapshot \
  --run-dir "${RUBRIC_RUN_DIR}" --repo "${PROJECT_ROOT}" --source "${FILE_PATH}" \
  --train "${TRAIN_DATA_PATH}" --val "${VAL_DATA_PATH}"

echo "[INFO] DATASET_SOURCE: TRAIN=${SOURCE_TRAIN_DATA_PATH}, VAL=${SOURCE_VAL_DATA_PATH}"
echo "[INFO] DATASET_RUNTIME: TRAIN=${TRAIN_DATA_PATH}, VAL=${VAL_DATA_PATH}"
echo "[INFO] REWARD_MODE=${REWARD_MODE}, RUBRIC_JUDGE_GRANULARITY=${RUBRIC_JUDGE_GRANULARITY}, TRAINING_DATA_VERSION=${TRAINING_DATA_VERSION}, TRAIN_PROMPT_VERSION=${TRAIN_PROMPT_VERSION:-structured_steps}, TRAIN_MAX_PROMPT_LENGTH=${TRAIN_MAX_PROMPT_LENGTH}, RUBRIC_MULTIPLE_FAIL_VALUE=${RUBRIC_MULTIPLE_FAIL_VALUE}, RUBRIC_SAFE_PRED_REWARD_MODE=${RUBRIC_SAFE_PRED_REWARD_MODE}, RUBRIC_LENGTH_PENALTY_WEIGHT=${RUBRIC_LENGTH_PENALTY_WEIGHT}, RUBRIC_EVOLVE=${RUBRIC_EVOLVE}"
if [ "${RUBRIC_EVOLVE}" = "1" ]; then
  echo "[INFO] ONLINE_EVOLUTION: start_step=${RUBRIC_EVOLVE_START_STEP}, interval=${RUBRIC_EVOLVE_INTERVAL_STEPS}, explicit_steps=${RUBRIC_EVOLVE_STEPS}, scope=${RUBRIC_EVOLVE_SCOPE}, provider=${RUBRIC_EVOLVE_PROVIDER}, model=${RUBRIC_EVOLVE_API_MODEL}, two_stage=${RUBRIC_EVOLVE_TWO_STAGE}, min_gap=${RUBRIC_EVOLVE_MIN_GAP}"
  echo "[INFO] VARIANCE_WEIGHT: enabled=${RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE}, start_step=${RUBRIC_EVOLVE_VARIANCE_START_STEP}, min_rollouts=${RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS}, range=[${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN},${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX}]"
fi
echo "[INFO] POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "[INFO] JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH}"
echo "[INFO] GUARD_MODEL_PATH=${GUARD_MODEL_PATH}"
echo "[INFO] CKPT_PATH=${CKPT_PATH}"
echo "[INFO] ROLLOUT_TRACE_ENABLE=${ROLLOUT_TRACE_ENABLE}, ROLLOUT_TRACE_DIR=${ROLLOUT_TRACE_DIR}, ROLLOUT_TRACE_MAX_PER_STEP=${ROLLOUT_TRACE_MAX_PER_STEP}"

if [[ "${TRAINING_DATA_VERSION}" =~ ^new(_v[23])?$ ]]; then
  train_files="['${TRAIN_DATA_PATH}/benign_sampled.parquet','${TRAIN_DATA_PATH}/sensitive_sampled.parquet','${TRAIN_DATA_PATH}/harm_sampled.parquet','${TRAIN_DATA_PATH}/adaptive_harm.parquet']"
  test_files="['${VAL_DATA_PATH}/benign_sampled.parquet','${VAL_DATA_PATH}/sensitive_sampled.parquet','${VAL_DATA_PATH}/harm_sampled.parquet','${VAL_DATA_PATH}/adaptive_harm.parquet']"
else
  benign_train_path="${TRAIN_DATA_PATH}/benign_sampled.parquet"
  harm_train_path="${TRAIN_DATA_PATH}/harm_sampled.parquet"
  enhanced_harm_train_path="${TRAIN_DATA_PATH}/enhanced_harm_sampled.parquet"
  sensitive_benign_train_path="${TRAIN_DATA_PATH}/sensitive_sampled.parquet"
  xstest_benign_train_path="${TRAIN_DATA_PATH}/xstest_benign.parquet"
  xtest_harm_train_path="${TRAIN_DATA_PATH}/xtest_harm.parquet"
  train_files="['$benign_train_path','$harm_train_path','$enhanced_harm_train_path','$sensitive_benign_train_path','$xstest_benign_train_path','$xtest_harm_train_path']"
  test_files="['${VAL_DATA_PATH}/xstest_benign.parquet','${VAL_DATA_PATH}/xtest_harm.parquet','${VAL_DATA_PATH}/enhanced_harm_sampled.parquet','${VAL_DATA_PATH}/sensitive_sampled.parquet']"
fi

unset RAY_ADDRESS
ray stop --force 2>/dev/null || true

if [ -z "${TRAIN_BEST_METRIC_KEY:-}" ]; then
  if [ "${REWARD_MODE}" = "rubric_safe_helpful_balanced" ]; then
    TRAIN_BEST_METRIC_KEY="val-core/balanced/pareto_score"
  else
    TRAIN_BEST_METRIC_KEY="val-core/harm/rubric_plus_unsafe"
  fi
fi
export TRAIN_BEST_METRIC_KEY

if [ -z "${TRAIN_N_GPUS_PER_NODE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    TRAIN_N_GPUS_PER_NODE="$(awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
  else
    TRAIN_N_GPUS_PER_NODE=1
  fi
fi
ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-${TRAIN_N_GPUS_PER_NODE}}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-true}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-true}"
TRAIN_TOTAL_EPOCHS="${TRAIN_TOTAL_EPOCHS:-3}"
if ! [[ "${TRAIN_TOTAL_EPOCHS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TRAIN_TOTAL_EPOCHS must be a positive integer." >&2
  exit 2
fi
TRAIN_RESUME_MODE="${TRAIN_RESUME_MODE:-auto}"
case "${TRAIN_RESUME_MODE}" in
  auto|disable) ;;
  *)
    echo "[ERROR] TRAIN_RESUME_MODE must be auto or disable." >&2
    exit 2
    ;;
esac
checkpoint_recovery_args=()
if [ "${RUBRIC_CHECKPOINT_RECOVERY_ENABLE:-0}" = "1" ]; then
  checkpoint_recovery_args+=(trainer.del_local_ckpt_after_load=False)
fi
TRAIN_TOTAL_TRAINING_STEPS="${TRAIN_TOTAL_TRAINING_STEPS:-}"
trainer_total_training_steps_args=()
if [ -n "${TRAIN_TOTAL_TRAINING_STEPS}" ]; then
  if ! [[ "${TRAIN_TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] TRAIN_TOTAL_TRAINING_STEPS must be a positive integer when set." >&2
    exit 2
  fi
  trainer_total_training_steps_args=("trainer.total_training_steps=${TRAIN_TOTAL_TRAINING_STEPS}")
fi
TRAIN_SAVE_FREQ="${TRAIN_SAVE_FREQ:-40}"
if ! [[ "${TRAIN_SAVE_FREQ}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[ERROR] TRAIN_SAVE_FREQ must be a positive integer." >&2
  exit 2
fi

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${PROJECT_ROOT}/verl/utils/reward_score/rubric_judge.py" \
    custom_reward_function.name=compute_score_batch \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.rollout.temperature=1 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.max_prompt_length="${TRAIN_MAX_PROMPT_LENGTH}" \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP_SIZE}" \
    data.return_raw_chat=True \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.path="${POLICY_MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${TRAIN_LEARNING_RATE:-1e-6}" \
    actor_rollout_ref.actor.use_kl_loss="${TRAIN_USE_KL_LOSS:-False}" \
    actor_rollout_ref.actor.kl_loss_coef="${TRAIN_KL_LOSS_COEF:-0.001}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    trainer.logger="${TRAIN_LOGGER_BACKENDS:-[wandb]}" \
    trainer.project_name="${TRAIN_PROJECT_NAME:-rubric_safety_2}" \
    trainer.experiment_name="${TRAIN_EXPERIMENT_NAME:-qwen3B_instruct_GRPO}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node="${TRAIN_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq="${TRAIN_SAVE_FREQ}" \
    trainer.test_freq="${TRAIN_TEST_FREQ:-40}" \
    trainer.total_epochs="${TRAIN_TOTAL_EPOCHS}" \
    trainer.resume_mode="${TRAIN_RESUME_MODE}" \
    ${checkpoint_recovery_args[@]+"${checkpoint_recovery_args[@]}"} \
    ${trainer_total_training_steps_args[@]+"${trainer_total_training_steps_args[@]}"} \
    trainer.log_val_generations=10 \
    trainer.rollout_trace.enabled="${ROLLOUT_TRACE_ENABLE}" \
    trainer.rollout_trace.dir="${ROLLOUT_TRACE_DIR}" \
    trainer.rollout_trace.max_samples_per_step="${ROLLOUT_TRACE_MAX_PER_STEP}" \
    trainer.max_actor_ckpt_to_keep=-1 \
    trainer.default_local_dir="${CKPT_PATH}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    reward_model.enable=False \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.fsdp_config.param_offload=True \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.use_remove_padding=True \
    trainer.save_best_ckpt="${TRAIN_SAVE_BEST_CKPT:-True}" \
    trainer.best_metric_key="${TRAIN_BEST_METRIC_KEY}" \
    trainer.best_metric_mode="${TRAIN_BEST_METRIC_MODE:-max}"
