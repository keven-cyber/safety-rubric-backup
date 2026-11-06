#!/usr/bin/env bash
# One-click Primus launcher for the NSPO baseline on the RubricAlign v3 dataset.
#
# Mirrors train_val.sh stage for stage: download models, start the reward service,
# train, merge the selected checkpoints, run the shared safety evaluation, then run
# MMLU/GSM8K. The scoring code is the parent repository's Evaluation pipeline, so
# NSPO and RubricAlign numbers are produced by identical judges.
#
# Two things are specific to NSPO:
#   * the reward is a Qwen3Guard verdict served over an OpenAI-compatible
#     endpoint, so a Guard replica stays resident for the whole training run;
#   * every actor update is projected into the null space of the MLP input
#     activations measured on a general instruction corpus, which this script
#     downloads from a Hugging Face mirror before training starts.
#
# NSPO ships verl 0.5, which needs tensordict >=0.8.0 while the parent repository
# pins <=0.6.2. NSPO therefore uses a dedicated environment by default; it does
# not inherit the parent image's Python packages or its optional transformer-engine
# meta package. Set NSPO_USE_SYSTEM_PACKAGES=1 only for a verified compatible image.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_PROJECT_ROOT="${SCRIPT_DIR}"
export NSPO_ROOT="${NSPO_ROOT:-${SCRIPT_DIR}/NSPO}"
export NSPO_VERL_ROOT="${NSPO_VERL_ROOT:-${NSPO_ROOT}/verl}"
export METHOD_TAG="${METHOD_TAG:-nspo}"
# Dataset fallback is intentionally disabled: a missing enriched corpus must
# fail the job instead of silently changing the comparison set.
export RUBRIC_DATASET_FALLBACK_DIR_NAME=""

# shellcheck source=scripts/baseline_common.sh
source "${SCRIPT_DIR}/scripts/baseline_common.sh"

POLICY_MODEL_ID="${POLICY_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
POLICY_MODEL_DIR="${POLICY_MODEL_DIR:-}"
SKIP_POLICY_DOWNLOAD="${SKIP_POLICY_DOWNLOAD:-0}"

# Reward Guard (training). Use the same Qwen3Guard model as final evaluation;
# the previous Llama-Guard mirror repository is gated/unavailable on Primus nodes.
NSPO_GUARD_MODEL_ID="${NSPO_GUARD_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
export NSPO_GUARD_MODEL="${NSPO_GUARD_MODEL:-qwen3_guard_gen_8b}"
NSPO_GUARD_MODEL_DIR="${NSPO_GUARD_MODEL_DIR:-}"
SKIP_NSPO_GUARD_DOWNLOAD="${SKIP_NSPO_GUARD_DOWNLOAD:-0}"
NSPO_GUARD_GPU_IDS="${NSPO_GUARD_GPU_IDS:-0}"
NSPO_GUARD_TP_SIZE="${NSPO_GUARD_TP_SIZE:-1}"
NSPO_GUARD_PORT="${NSPO_GUARD_PORT:-52001}"
NSPO_GUARD_MAX_MODEL_LEN="${NSPO_GUARD_MAX_MODEL_LEN:-4096}"

# Evaluation Guard (final safety scoring) - identical to train_val.sh.
GUARD_MODEL_ID="${GUARD_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
GUARD_SERVED_MODEL_NAME="${GUARD_SERVED_MODEL_NAME:-qwen3_guard_gen_8b}"
GUARD_GPU_IDS="${GUARD_GPU_IDS:-3}"
GUARD_TP_SIZE="${GUARD_TP_SIZE:-1}"
GUARD_PORT="${GUARD_PORT:-8001}"
# The evaluation Guard is the same 8B snapshot the reward Guard already uses, so a
# complete local copy is reused instead of re-fetched on every run.
FORCE_GUARD_DOWNLOAD="${FORCE_GUARD_DOWNLOAD:-0}"

export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-4,5,6,7}"
export ALL_JOB_GPU_IDS="${ALL_JOB_GPU_IDS:-0,1,2,3,4,5,6,7}"

# Null-space corpus.  Keep this transient corpus on the node-local filesystem
# alongside downloaded models; Primus OSS mounts may reject link operations and
# are unnecessary for data that is rebuilt per job. Override the directory when
# a persistent corpus is explicitly desired.
NSPO_NULLSPACE_DATASET_ID="${NSPO_NULLSPACE_DATASET_ID:-tatsu-lab/alpaca}"
NSPO_NULLSPACE_ALLOW_FALLBACK="${NSPO_NULLSPACE_ALLOW_FALLBACK:-0}"
NSPO_NULLSPACE_STAGING_ROOT="${NSPO_NULLSPACE_STAGING_ROOT:-/root/nspo_nullspace_staging}"
# Rebuild stale corpora automatically.  Older Primus jobs created a
# ModelScope/fallback manifest at the same OSS path; reusing it would make the
# NSPO comparison depend on a different null-space estimate.
NSPO_NULLSPACE_REBUILD="${NSPO_NULLSPACE_REBUILD:-0}"
export NSPO_NULLSPACE_ENABLE="${NSPO_NULLSPACE_ENABLE:-1}"

# Interpreter for verl 0.5.
NSPO_PYTHON_BIN="${NSPO_PYTHON_BIN:-}"
# The reward Guard is an OpenAI-compatible HTTP service, but it still imports
# vLLM. Reuse the interpreter that passes the NSPO dependency probe by default;
# callers can explicitly set NSPO_GUARD_PYTHON_BIN for a separate environment.
NSPO_GUARD_PYTHON_BIN="${NSPO_GUARD_PYTHON_BIN:-}"
NSPO_VENV_DIR="${NSPO_VENV_DIR:-/root/venvs/nspo}"
NSPO_BUILD_VENV="${NSPO_BUILD_VENV:-1}"
NSPO_VLLM_VERSION="${NSPO_VLLM_VERSION:-0.8.5}"
# vLLM 0.8.5 requires the Transformers 4 tokenizer API. Keep this stack
# private to NSPO; RubricAlign and AlphaAlign continue using the image env.
NSPO_TRANSFORMERS_VERSION="${NSPO_TRANSFORMERS_VERSION:-4.51.3}"
NSPO_RAY_VERSION="${NSPO_RAY_VERSION:-2.43.0}"
# With --system-site-packages, pip can leave the image's vLLM ahead of the
# venv package on some Primus images.  Keep an explicit overlay so the
# interpreter used by training and both Guard services is deterministic.
NSPO_VLLM_OVERLAY_DIR="${NSPO_VLLM_OVERLAY_DIR:-${NSPO_VENV_DIR}/nspo_vllm_overlay}"
NSPO_USE_SYSTEM_PACKAGES="$(normalize_bool NSPO_USE_SYSTEM_PACKAGES "${NSPO_USE_SYSTEM_PACKAGES:-0}")"
NSPO_AUTO_ISOLATE_ON_BROKEN_ENV="$(normalize_bool NSPO_AUTO_ISOLATE_ON_BROKEN_ENV "${NSPO_AUTO_ISOLATE_ON_BROKEN_ENV:-1}")"
NSPO_ATTN_IMPL="${NSPO_ATTN_IMPL:-sdpa}"
NSPO_USE_REMOVE_PADDING="$(normalize_bool NSPO_USE_REMOVE_PADDING "${NSPO_USE_REMOVE_PADDING:-0}")"
NSPO_ORIGINAL_PYTHONPATH="${PYTHONPATH:-}"
RESUME_TRAINING="$(normalize_bool RESUME_TRAINING "${RESUME_TRAINING:-0}")"
export NSPO_VLLM_VERSION
export NSPO_TRANSFORMERS_VERSION NSPO_RAY_VERSION
export NSPO_VLLM_OVERLAY_DIR
export NSPO_AUTO_ISOLATE_ON_BROKEN_ENV
export NSPO_ATTN_IMPL NSPO_USE_REMOVE_PADDING

export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_LOGGING_LEVEL=WARNING

FINAL_SAFETY_EVAL_ENABLE="$(normalize_bool FINAL_SAFETY_EVAL_ENABLE "${FINAL_SAFETY_EVAL_ENABLE:-1}")"
FINAL_GENERAL_EVAL_ENABLE="$(normalize_bool FINAL_GENERAL_EVAL_ENABLE "${FINAL_GENERAL_EVAL_ENABLE:-1}")"
# Stage gate. The pipeline is linear, so an evaluation that dies used to force a
# full retrain on relaunch. RUN_TRAIN=0 (or --eval-only) keeps CKPT_PATH pointed
# at the existing checkpoints and resumes from the merge stage onward.
RUN_TRAIN="$(normalize_bool RUN_TRAIN "${RUN_TRAIN:-1}")"
FORCE_GUARD_DOWNLOAD="$(normalize_bool FORCE_GUARD_DOWNLOAD "${FORCE_GUARD_DOWNLOAD}")"
NSPO_BUILD_VENV="$(normalize_bool NSPO_BUILD_VENV "${NSPO_BUILD_VENV}")"
NSPO_NULLSPACE_ALLOW_FALLBACK="$(normalize_bool NSPO_NULLSPACE_ALLOW_FALLBACK "${NSPO_NULLSPACE_ALLOW_FALLBACK}")"
NSPO_NULLSPACE_REBUILD="$(normalize_bool NSPO_NULLSPACE_REBUILD "${NSPO_NULLSPACE_REBUILD}")"
if [ "${NSPO_NULLSPACE_ALLOW_FALLBACK}" = "1" ]; then
  echo "[ERROR] NSPO null-space fallback is disabled; provide a reachable HF corpus instead." >&2
  exit 2
fi
export FINAL_GENERAL_EVAL_BACKEND="${FINAL_GENERAL_EVAL_BACKEND:-vllm}"
export FINAL_GENERAL_EVAL_DEVICE="${FINAL_GENERAL_EVAL_DEVICE:-cuda:0}"
export FINAL_EVAL_MIN_STEP="${FINAL_EVAL_MIN_STEP:-0}"
case "${FINAL_GENERAL_EVAL_BACKEND}" in
  hf|vllm) ;;
  *)
    echo "[ERROR] FINAL_GENERAL_EVAL_BACKEND must be hf or vllm." >&2
    exit 2
    ;;
esac

usage() {
  cat <<'USAGE'
Usage: bash train_val_nspo.sh [options]

Options:
  -p, --policy-model <model_id>   Policy model id for the Hugging Face mirror download
      --policy-dir <dir>          Policy download directory (default: $MODEL_BASE_PATH/policy)
      --policy-path <dir>         Existing local policy path used for training
      --skip-policy-download      Skip the policy download
      --eval-only                 Skip training and merge/evaluate the checkpoints
                                  already present in CKPT_PATH (same as RUN_TRAIN=0)
      --python <bin>              Interpreter that provides verl 0.5 (skips the venv build)
  -h, --help                      Show this help

Environment:
  BASELINE_PROMPT_SOURCE          raw_question (default) | instruction
  TRAIN_GPU_IDS                   Policy GPU ids (default: 4,5,6,7)
  NSPO_GUARD_GPU_IDS              Reward Guard GPU ids (default: 0)
  NSPO_GUARD_PORT                 Reward Guard port (default: 52001)
  NSPO_GUARD_MODEL_ID             Reward Guard model id (default: Qwen/Qwen3Guard-Gen-8B)
  GUARD_GPU_IDS                   Qwen3Guard GPU ids for the final safety eval (default: 3)
  NSPO_NULLSPACE_DATASET_ID       Hugging Face dataset for the null-space corpus (default: tatsu-lab/alpaca)
  NSPO_NULLSPACE_ALLOW_FALLBACK   Must remain 0; null-space fallback is disabled
  NSPO_NULLSPACE_ENABLE           0 disables the projection, reducing NSPO to GRPO+Guard (default: 1)
  NSPO_NULLSPACE_SAMPLE_SIZE      Rank-0 sample count for the projection (default: 1024)
  NSPO_AMEND_INTERVAL             Projection amend interval in steps (default: 5)
  NSPO_PYTHON_BIN                 Interpreter that already has verl 0.5 installed
  NSPO_GUARD_PYTHON_BIN           Interpreter used for Guard APIs (default: NSPO interpreter)
  NSPO_VENV_DIR                   Clean venv location (default: /root/venvs/nspo)
  NSPO_TRANSFORMERS_VERSION        Transformers version for vLLM 0.8.5 (default: 4.51.3)
  NSPO_RAY_VERSION                 Ray version for isolated NSPO runtime (default: 2.43.0)
  NSPO_VLLM_OVERLAY_DIR           vLLM overlay used when system vLLM is mismatched
  NSPO_AUTO_ISOLATE_ON_BROKEN_ENV Rebuild without system packages after an environment probe failure (default: 1)
  NSPO_ATTN_IMPL                 HF attention implementation (default: sdpa; flash_attention_2 if wheel exists)
  NSPO_USE_REMOVE_PADDING        Enable the flash-attn remove-padding fast path (default: 0)
  NSPO_BUILD_VENV                 Build the venv when the probe fails (default: 1)
  NSPO_USE_SYSTEM_PACKAGES       Reuse image Python packages in the venv (default: 0)
  RESUME_TRAINING                 Resume existing checkpoints instead of creating a fresh run (default: 0)
  TRAIN_BATCH_SIZE                Default 8, aligned with the RubricAlign launcher
  ROLLOUT_N                       Default 8
  ACTOR_LR                        Default 1e-6
  TRAIN_MAX_PROMPT_LENGTH         Default 1536
  TRAIN_TOTAL_EPOCHS              Default 3
  TRAIN_TOTAL_STEPS               Optional hard cap on optimizer steps
  TRAIN_SAVE_FREQ                 Default 30
  FINAL_EVAL_MIN_STEP             Only merge/evaluate checkpoints at or above this step (default: 0)
  FINAL_SAFETY_EVAL_STEPS         Exact steps plus last,best (default: 600,900,last,best)
  SAFETY_EVAL_GPU_IDS             One GPU for final safety generation (default: first TRAIN_GPU_IDS GPU)
  BASELINE_GENERAL_EVAL_STEP      Evaluate this step (last scope only)
  BASELINE_GENERAL_EVAL_SCOPE     last (default) | selected (same models as safety)
  BASELINE_REQUIRED_FINAL_STEPS   Required numbered coverage, e.g. 900,990,1080,1170
  RUN_TRAIN                       Run the training stage (default: 1). With 0 the
                                  launcher requires an existing checkpoint in
                                  CKPT_PATH, skips the reward Guard service, and
                                  starts at the merge stage; this is how an
                                  interrupted evaluation is resumed.
  FORCE_GUARD_DOWNLOAD            Re-download the Guard even when a complete local
                                  copy exists (default: 0)
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
    --python)
      NSPO_PYTHON_BIN="${2:?missing value for --python}"
      shift 2
      ;;
    --eval-only)
      RUN_TRAIN=0
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

export POLICY_MODEL_ID

if [ ! -d "${NSPO_VERL_ROOT}/verl" ]; then
  echo "[ERROR] NSPO checkout not found at ${NSPO_VERL_ROOT}" >&2
  exit 1
fi

for package in fire tqdm; do
  if ! python3 -c "import ${package}" >/dev/null 2>&1; then
    python3 -m pip install "${package}"
  fi
done
# transformers 4.57.x requires huggingface-hub>=0.34.0,<1.0. Check the version,
# not merely whether the package imports.
if ! python3 - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

parts = tuple(int(part) for part in version("huggingface-hub").split(".")[:2])
raise SystemExit(0 if (0, 34) <= parts < (1, 0) else 1)
PY
then
  python3 -m pip install "huggingface_hub==${HF_HUB_PIN:-0.36.2}"
fi
ray stop --force

export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
if [ -z "${POLICY_MODEL_DIR}" ]; then
  POLICY_MODEL_DIR="${MODEL_BASE_PATH}/policy"
fi
if [ -z "${NSPO_GUARD_MODEL_DIR}" ]; then
  # Training and final evaluation use the same weights; avoid downloading the
  # 8B Guard twice into separate directories.
  NSPO_GUARD_MODEL_DIR="${MODEL_BASE_PATH}/guard"
fi
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/${METHOD_TAG}}}"
# Persist service logs in the Primus OSS workspace so they survive the task and
# can be inspected alongside eval_outputs. Override API_LOG_DIR for a custom
# mounted location.
export API_LOG_DIR="${API_LOG_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/api_logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/eval_outputs}"
RAY_TMPDIR_ROOT="${RAY_TMPDIR_ROOT:-/tmp/ray_temp}"
# Do not recursively clear the shared /tmp/ray_temp after ray stop: old Ray
# sessions can make that silent deletion take a long time on Primus. A unique
# directory also prevents concurrently scheduled jobs from deleting each other.
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMPDIR_ROOT}/${METHOD_TAG}_${PRIMUS_TASK_ID:-local}_$$}"
# The inner NSPO launcher honors this flag. The directory above is new for this
# job, so recursively clearing it again is unnecessary and can block when a
# caller explicitly supplied a shared RAY_TMPDIR.
export CLEAR_RAY_TMPDIR="${CLEAR_RAY_TMPDIR:-0}"

mkdir -p "${MODEL_BASE_PATH}/guard" "${POLICY_MODEL_DIR}" "${NSPO_GUARD_MODEL_DIR}" \
         "${CKPT_PATH}" "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${RAY_TMPDIR}"
if [ "${RUN_TRAIN}" = "1" ] && [ "${RESUME_TRAINING}" = "0" ] && find "${CKPT_PATH}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print -quit | grep -q .; then
  previous_ckpt_path="${CKPT_PATH}"
  fresh_run_id="$(date +%Y%m%d_%H%M%S)_$$"
  CKPT_PATH="${previous_ckpt_path}_fresh_${fresh_run_id}"
  while [ -e "${CKPT_PATH}" ]; do
    fresh_run_id="$(date +%Y%m%d_%H%M%S)_$((RANDOM % 100000))"
    CKPT_PATH="${previous_ckpt_path}_fresh_${fresh_run_id}"
  done
  export CKPT_PATH
  mkdir -p "${CKPT_PATH}"
  echo "[WARN] Existing checkpoints found in ${previous_ckpt_path}." >&2
  echo "[INFO] Starting a fresh run in ${CKPT_PATH}; old checkpoints are preserved." >&2
fi
if [ "${RUN_TRAIN}" = "0" ]; then
  # Resuming an interrupted evaluation only makes sense against the directory the
  # original run wrote to, so the fresh-directory branch above is bypassed and a
  # missing checkpoint is a hard error rather than an empty merge.
  if ! find "${CKPT_PATH}" -maxdepth 1 -mindepth 1 -type d -name 'global_step_*' -print -quit | grep -q .; then
    echo "[ERROR] RUN_TRAIN=0 but ${CKPT_PATH} holds no global_step_* checkpoint." >&2
    echo "[ERROR] Set CKPT_PATH to the directory the interrupted run used." >&2
    exit 1
  fi
  echo "[INFO] Training stage disabled; reusing checkpoints in ${CKPT_PATH}."
fi
echo "[INFO] Ray temporary directory: ${RAY_TMPDIR}"
echo "[INFO] CLEAR_RAY_TMPDIR=${CLEAR_RAY_TMPDIR}"

unset PYTORCH_CUDA_ALLOC_CONF
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export RAY_raylet_start_wait_time_s=300
export RAY_gcs_server_request_timeout_seconds=300

echo "[INFO] ============================================="
echo "[INFO] NSPO baseline, one-click Primus run"
echo "[INFO] ============================================="
echo "[INFO] Model Base Path: ${MODEL_BASE_PATH}"
echo "[INFO] Checkpoint Path: ${CKPT_PATH}"
echo "[INFO] Stages: train=${RUN_TRAIN} safety_eval=${FINAL_SAFETY_EVAL_ENABLE} general_eval=${FINAL_GENERAL_EVAL_ENABLE}"
echo "[INFO] Eval Output Dir: ${EVAL_OUTPUT_DIR}"
echo "[INFO] API Log Dir: ${API_LOG_DIR}"
echo "[INFO] ============================================="

# =========================
# Downloads
# =========================
download_model_from_hf_mirror() {
  local model_id="$1"
  local model_dir="$2"
  local model_label="$3"
  local max_attempts="${MODEL_DOWNLOAD_RETRIES:-5}"
  local attempt
  require_positive_int MODEL_DOWNLOAD_RETRIES "${max_attempts}"
  for ((attempt = 1; attempt <= max_attempts; attempt++)); do
    echo "[INFO] Downloading ${model_label} from ${HF_MIRROR_ENDPOINT}: ${model_id} (attempt ${attempt}/${max_attempts}) ..."
    if HF_ENDPOINT="${HF_MIRROR_ENDPOINT}" python3 - "${model_id}" "${model_dir}" "${HF_MIRROR_ENDPOINT}" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo_id, local_dir, endpoint = sys.argv[1:]
revision = None
if __import__('os').environ.get('CROSS_MODEL_EXPERIMENT_KEY'):
    import json
    from pathlib import Path
    contracts = json.loads((Path(__import__('os').environ['RUBRIC_PROJECT_ROOT']) / 'configs/cross_model_twelve_dependency_models.json').read_text())
    revision = contracts[repo_id]['HF_revision']
snapshot_download(repo_id=repo_id, revision=revision, local_dir=local_dir, endpoint=endpoint, max_workers=8)
PY
    then
      return 0
    fi
    if [ "${attempt}" -lt "${max_attempts}" ]; then
      echo "[WARN] ${model_label} download was interrupted; snapshot_download will resume partial files." >&2
      sleep $((attempt * 5))
    fi
  done
  echo "[ERROR] Failed to download ${model_label} after ${max_attempts} attempts: ${model_id}" >&2
  return 1
}

model_snapshot_complete() {
  # snapshot_download leaves *.incomplete markers behind when it is interrupted, so
  # a directory only counts as reusable when the config and at least one non-empty
  # weight shard are present and no partial file remains.
  local model_dir="$1"
  python3 - "${model_dir}" <<'PY'
import glob
import json
import os
import sys

root = os.path.abspath(sys.argv[1])
if not os.path.isfile(os.path.join(root, "config.json")):
    raise SystemExit(1)
if glob.glob(os.path.join(root, "**", "*.incomplete"), recursive=True):
    raise SystemExit(1)
indexes = glob.glob(os.path.join(root, "*.safetensors.index.json"))
indexes += glob.glob(os.path.join(root, "pytorch_model.bin.index.json"))
if indexes:
    required = set()
    for index_path in indexes:
        with open(index_path, encoding="utf-8") as handle:
            required.update(json.load(handle).get("weight_map", {}).values())
    ok = bool(required) and all(
        os.path.isfile(os.path.join(root, name)) and os.path.getsize(os.path.join(root, name)) > 0
        for name in required
    )
else:
    weights = glob.glob(os.path.join(root, "*.safetensors"))
    weights += glob.glob(os.path.join(root, "pytorch_model*.bin"))
    ok = bool(weights) and all(os.path.getsize(path) > 0 for path in weights)
raise SystemExit(0 if ok else 1)
PY
}

MODEL_DOWNLOAD_SOURCE="${MODEL_DOWNLOAD_SOURCE:-hf_mirror}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
if [ "${MODEL_DOWNLOAD_SOURCE}" != "hf_mirror" ]; then
  echo "[ERROR] MODEL_DOWNLOAD_SOURCE must be hf_mirror; ModelScope is not supported." >&2
  exit 2
fi
if [ "${SKIP_POLICY_DOWNLOAD}" = "1" ]; then
  echo "[INFO] Skip policy download (SKIP_POLICY_DOWNLOAD=1)"
elif [ "${RUN_TRAIN}" = "0" ] && model_snapshot_complete "${POLICY_MODEL_DIR}"; then
  # Without training the base model is only the tokenizer/config source for a
  # checkpoint that still has to be merged, so a complete local copy is enough.
  echo "[INFO] Reusing the policy model at ${POLICY_MODEL_DIR}; skipping download."
else
  download_model_from_hf_mirror "${POLICY_MODEL_ID}" "${POLICY_MODEL_DIR}" "Policy model"
fi

if [ "${RUN_TRAIN}" = "0" ]; then
  # The reward Guard only scores rollouts during training. When the evaluation
  # Guard shares this directory the final-eval branch below reuses it anyway.
  echo "[INFO] Skip reward Guard download (RUN_TRAIN=0; training-only dependency)"
elif [ "${SKIP_NSPO_GUARD_DOWNLOAD}" = "1" ]; then
  echo "[INFO] Skip reward Guard download (SKIP_NSPO_GUARD_DOWNLOAD=1)"
elif [ "${FORCE_GUARD_DOWNLOAD}" = "0" ] && model_snapshot_complete "${NSPO_GUARD_MODEL_DIR}"; then
  echo "[INFO] Reusing the reward Guard at ${NSPO_GUARD_MODEL_DIR}; skipping download."
else
  download_model_from_hf_mirror "${NSPO_GUARD_MODEL_ID}" "${NSPO_GUARD_MODEL_DIR}" "reward Guard model"
fi

if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
  if [ "${FORCE_GUARD_DOWNLOAD}" = "0" ] && model_snapshot_complete "${MODEL_BASE_PATH}/guard"; then
    echo "[INFO] Reusing the Guard at ${MODEL_BASE_PATH}/guard; skipping download."
    echo "[INFO] Set FORCE_GUARD_DOWNLOAD=1 to re-fetch ${GUARD_MODEL_ID}."
  else
    download_model_from_hf_mirror "${GUARD_MODEL_ID}" "${MODEL_BASE_PATH}/guard" "final-evaluation Guard"
  fi
fi

export POLICY_MODEL_PATH="${POLICY_MODEL_PATH:-${POLICY_MODEL_DIR}}"
export NSPO_GUARD_MODEL_PATH="${NSPO_GUARD_MODEL_PATH:-${NSPO_GUARD_MODEL_DIR}}"
export GUARD_MODEL_PATH="${GUARD_MODEL_PATH:-${MODEL_BASE_PATH}/guard}"

if [ ! -f "${POLICY_MODEL_PATH}/config.json" ]; then
  if [ "${RUN_TRAIN}" = "1" ]; then
    echo "[ERROR] Policy model is incomplete: ${POLICY_MODEL_PATH}/config.json is missing." >&2
    exit 1
  fi
  # Merging is skipped for checkpoints that already produced a model directory, so
  # an absent base model is only fatal when a new one still has to be merged.
  echo "[WARN] Policy model is absent: ${POLICY_MODEL_PATH}/config.json is missing." >&2
  echo "[WARN] Already-merged checkpoints still evaluate; merging a new step will fail." >&2
fi
if [ "${RUN_TRAIN}" = "1" ] && [ ! -f "${NSPO_GUARD_MODEL_PATH}/config.json" ]; then
  echo "[ERROR] Reward Guard model is incomplete: ${NSPO_GUARD_MODEL_PATH}/config.json is missing." >&2
  echo "[ERROR] Set NSPO_GUARD_MODEL_ID to a reachable mirror, or NSPO_GUARD_MODEL_PATH to a local copy." >&2
  exit 1
fi

# =========================
# Null-space corpus
# =========================
export NSPO_NULLSPACE_DATASET_DIR="${NSPO_NULLSPACE_DATASET_DIR:-/root/nspo_nullspace}"
# Primus jobs can inherit the old OSS path from a task template.  OSS does not
# support the link operations used by huggingface_hub, so never allow that
# stale value to select the null-space download location implicitly.
if [[ "${NSPO_NULLSPACE_DATASET_DIR}" == /primus_oss/* ]]; then
  echo "[WARN] Ignoring inherited OSS null-space path: ${NSPO_NULLSPACE_DATASET_DIR}" >&2
  export NSPO_NULLSPACE_DATASET_DIR="/root/nspo_nullspace"
fi
NSPO_NULLSPACE_RAW_DIR="${NSPO_NULLSPACE_RAW_DIR:-/root/nspo_nullspace_raw_hf}"
if [[ "${NSPO_NULLSPACE_RAW_DIR}" == /primus_oss/* ]]; then
  echo "[WARN] Ignoring inherited OSS null-space raw path: ${NSPO_NULLSPACE_RAW_DIR}" >&2
  NSPO_NULLSPACE_RAW_DIR="/root/nspo_nullspace_raw_hf"
fi
if [[ "${NSPO_NULLSPACE_STAGING_ROOT}" == /primus_oss/* ]]; then
  echo "[WARN] Ignoring inherited OSS null-space staging path: ${NSPO_NULLSPACE_STAGING_ROOT}" >&2
  NSPO_NULLSPACE_STAGING_ROOT="/root/nspo_nullspace_staging"
fi
export NSPO_NULLSPACE_STAGING_ROOT
if [ "${RUN_TRAIN}" = "0" ]; then
  # The null-space corpus is a training input only; building it during an
  # evaluation-only rerun would download and stage rows nothing reads.
  echo "[INFO] Skip the null-space corpus (RUN_TRAIN=0; training-only input)."
elif [ "${NSPO_NULLSPACE_ENABLE}" != "0" ]; then
  echo "[INFO] ============================================="
  echo "[INFO] Building the null-space corpus"
  echo "[INFO] ============================================="
  echo "[INFO] Null-space output: ${NSPO_NULLSPACE_DATASET_DIR}"
  echo "[INFO] Null-space raw download: ${NSPO_NULLSPACE_RAW_DIR}"
  echo "[INFO] Null-space staging: ${NSPO_NULLSPACE_STAGING_ROOT}"
  nullspace_args=(
    --dataset-id "${NSPO_NULLSPACE_DATASET_ID}"
    --output-dir "${NSPO_NULLSPACE_DATASET_DIR}"
    # Keep the HF download cache separate from the pre-existing ModelScope
    # cache.  The builder scans this directory, so mixing the two can silently
    # select an old shard even after a rebuild request.
    --raw-dir "${NSPO_NULLSPACE_RAW_DIR}"
    --max-rows "${NSPO_NULLSPACE_MAX_ROWS:-4096}"
  )
  if [ "${NSPO_NULLSPACE_ALLOW_FALLBACK}" = "1" ]; then
    nullspace_args+=(--allow-fallback)
  fi
  nullspace_manifest="${NSPO_NULLSPACE_DATASET_DIR}/manifest.json"
  if [ -f "${NSPO_NULLSPACE_DATASET_DIR}/train/nullspace.parquet" ] && [ -f "${nullspace_manifest}" ] && [ "${NSPO_NULLSPACE_REBUILD}" != "1" ]; then
    if ! python3 - "${nullspace_manifest}" "${NSPO_NULLSPACE_DATASET_ID}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    manifest = json.load(handle)
expected_dataset_id = sys.argv[2]
if manifest.get("source_kind") != "hf_mirror":
    raise SystemExit(f"source_kind={manifest.get('source_kind')!r}, expected 'hf_mirror'")
if manifest.get("dataset_id") != expected_dataset_id:
    raise SystemExit(
        f"dataset_id={manifest.get('dataset_id')!r}, expected {expected_dataset_id!r}"
    )
PY
    then
      echo "[WARN] Existing null-space corpus is stale or was built from a non-HF source: ${nullspace_manifest}" >&2
      echo "[INFO] Rebuilding it from ${NSPO_NULLSPACE_DATASET_ID} via ${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}." >&2
      NSPO_NULLSPACE_REBUILD=1
    fi
    if [ "${NSPO_NULLSPACE_REBUILD}" != "1" ]; then
      echo "[INFO] Reusing verified HF-mirror corpus: ${NSPO_NULLSPACE_DATASET_DIR}"
    fi
  fi
  if [ "${NSPO_NULLSPACE_REBUILD}" = "1" ] || [ ! -f "${NSPO_NULLSPACE_DATASET_DIR}/train/nullspace.parquet" ] || [ ! -f "${nullspace_manifest}" ]; then
    python3 "${SCRIPT_DIR}/scripts/prepare_nspo_nullspace_dataset.py" "${nullspace_args[@]}"
  fi
else
  echo "[WARN] NSPO_NULLSPACE_ENABLE=0: this run is GRPO+Guard, not NSPO."
fi

# =========================
# Interpreter for verl 0.5
# =========================
probe_nspo_interpreter() {
  local python_bin="$1"
  [ -x "${python_bin}" ] || command -v "${python_bin}" >/dev/null 2>&1 || return 1
  ( cd "${NSPO_VERL_ROOT}" && "${python_bin}" - <<'PYPROBE'
import os
import sys
from importlib.metadata import version as package_version

try:
    from packaging.version import Version
    import accelerate  # noqa: F401 - NSPO's FSDP worker dependency
    import datasets  # noqa: F401 - NSPO dataset worker dependency
    import hydra  # noqa: F401 - trainer entrypoint dependency
    import peft  # noqa: F401 - imported by NSPO's FSDP worker on Ray actors
    import pyvers  # noqa: F401 - required by tensordict >=0.8
    import ray
    import openai  # noqa: F401 - safe_reward.py needs it and requirements.txt omits it
    import tensordict
    import torchdata  # noqa: F401 - NSPO data worker dependency
    import transformers  # noqa: F401 - model and tokenizer dependency
    import vllm
    import verl

    version = Version(tensordict.__version__)
    if version < Version("0.8.0"):
        print(f"[PROBE] tensordict {tensordict.__version__} is too old for verl 0.5 (needs >=0.8.0)")
        sys.exit(1)
    expected_vllm = os.environ.get("NSPO_VLLM_VERSION", "0.8.5")
    expected_transformers = os.environ.get("NSPO_TRANSFORMERS_VERSION", "4.51.3")
    expected_ray = os.environ.get("NSPO_RAY_VERSION", "2.43.0")
    if Version(vllm.__version__) != Version(expected_vllm):
        print(
            f"[PROBE] vllm {vllm.__version__} does not match requested "
            f"NSPO_VLLM_VERSION={expected_vllm} (loaded from {vllm.__file__})"
        )
        sys.exit(1)
    if Version(ray.__version__) != Version(expected_ray):
        print(f"[PROBE] ray {ray.__version__} does not match NSPO_RAY_VERSION={expected_ray}")
        sys.exit(1)
    for package in ("opentelemetry-api", "opentelemetry-sdk", "opentelemetry-exporter-otlp"):
        installed = package_version(package)
        if not (Version("1.26.0") <= Version(installed) < Version("1.27.0")):
            print(f"[PROBE] {package} {installed} is outside vLLM 0.8.5's supported range [1.26, 1.27)")
            sys.exit(1)
    if Version(transformers.__version__) != Version(expected_transformers):
        print(
            f"[PROBE] transformers {transformers.__version__} does not match "
            f"NSPO_TRANSFORMERS_VERSION={expected_transformers}; vLLM {expected_vllm} "
            "requires the Transformers 4 tokenizer API"
        )
        sys.exit(1)
    print(
        f"[PROBE] python={sys.executable}, verl={verl.__version__}, "
        f"tensordict={tensordict.__version__}, transformers={transformers.__version__}, ray={ray.__version__}, vllm={vllm.__version__}, "
        f"vllm_path={vllm.__file__}, openai={openai.__version__}"
    )
except Exception as exc:  # noqa: BLE001
    print(f"[PROBE] import failed: {exc}")
    sys.exit(1)
PYPROBE
  )
}

build_nspo_venv() {
  local venv_dir="$1"
  echo "[INFO] ============================================="
  echo "[INFO] Building a clean environment for NSPO's verl 0.5 + vllm ${NSPO_VLLM_VERSION}: ${venv_dir}"
  echo "[INFO] NSPO_USE_SYSTEM_PACKAGES=${NSPO_USE_SYSTEM_PACKAGES}"
  echo "[INFO] ============================================="
  rm -rf "${venv_dir}"
  local venv_args=()
  if [ "${NSPO_USE_SYSTEM_PACKAGES}" = "1" ]; then
    venv_args+=(--system-site-packages)
    echo "[INFO] Reusing system-site-packages; only NSPO-specific packages will be overlaid."
  fi
  if python3 -m venv "${venv_args[@]}" "${venv_dir}"; then
    echo "[INFO] Created NSPO environment with stdlib venv."
  else
    echo "[WARN] python3 -m venv is unavailable (missing ensurepip); using virtualenv."
    if ! python3 -c 'import virtualenv' >/dev/null 2>&1; then
      python3 -m pip install virtualenv
    fi
    python3 -m virtualenv --no-download "${venv_args[@]}" "${venv_dir}"
  fi
  local pip_bin="${venv_dir}/bin/pip"
  # These packages share an API boundary in vLLM 0.8.5.  Without constraints,
  # `ray[default]` and verl's unpinned `transformers` requirement resolve to
  # their latest releases, which breaks the Guard tokenizer at startup.
  local constraints_file="${venv_dir}/nspo-runtime-constraints.txt"
  printf '%s\n' \
    "transformers==${NSPO_TRANSFORMERS_VERSION}" \
    "ray==${NSPO_RAY_VERSION}" \
    "opentelemetry-api==1.26.0" \
    "opentelemetry-sdk==1.26.0" \
    "opentelemetry-exporter-otlp==1.26.0" \
    "opentelemetry-proto==1.26.0" \
    "opentelemetry-semantic-conventions==0.47b0" \
    > "${constraints_file}"
  if [ "${NSPO_USE_SYSTEM_PACKAGES}" = "1" ]; then
    # The Primus image already contains the CUDA stack and vLLM used by the
    # working RubricAlign/AlphaAlign jobs. Keep those packages unless the
    # image's vLLM is older than the NSPO-pinned version.
    system_vllm_version="$(${venv_dir}/bin/python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || true)"
    if [ "${system_vllm_version}" != "${NSPO_VLLM_VERSION}" ]; then
      echo "[INFO] System vLLM=${system_vllm_version:-missing}; installing vLLM=${NSPO_VLLM_VERSION} into ${NSPO_VLLM_OVERLAY_DIR}."
      # pip's normal install/uninstall logic may regard the system package as
      # satisfying the requirement when system-site-packages is enabled.  A
      # clean --target overlay plus PYTHONPATH makes the selected version
      # unambiguous while retaining the image's CUDA/Torch dependencies.
      rm -rf "${NSPO_VLLM_OVERLAY_DIR}"
      mkdir -p "${NSPO_VLLM_OVERLAY_DIR}"
      "${pip_bin}" install --ignore-installed --no-deps --target "${NSPO_VLLM_OVERLAY_DIR}" "vllm==${NSPO_VLLM_VERSION}"
      export PYTHONPATH="${NSPO_VLLM_OVERLAY_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
      echo "[INFO] vLLM overlay: ${NSPO_VLLM_OVERLAY_DIR}"
      "${venv_dir}/bin/python" -c 'import vllm; print(f"[INFO] selected vLLM={vllm.__version__} ({vllm.__file__})")'
    fi
    if ! "${venv_dir}/bin/python" -c 'import pyvers' >/dev/null 2>&1; then
      "${pip_bin}" install --no-deps pyvers
    fi
    "${pip_bin}" install --no-deps "tensordict>=0.8.0,<=0.9.1,!=0.9.0"
    "${pip_bin}" install -c "${constraints_file}" "transformers==${NSPO_TRANSFORMERS_VERSION}" "ray[default]==${NSPO_RAY_VERSION}"
    "${pip_bin}" install --upgrade -c "${constraints_file}" "opentelemetry-api==1.26.0" "opentelemetry-sdk==1.26.0" "opentelemetry-exporter-otlp==1.26.0"
    if ! "${venv_dir}/bin/python" -c 'import openai' >/dev/null 2>&1; then
      "${pip_bin}" install --no-deps openai
    fi
    # An editable install with --no-deps keeps the image's CUDA/Torch stack,
    # but it also omits NSPO-only runtime requirements such as peft. Ray actor
    # processes import those modules after startup. Install the regular runtime
    # requirements, while excluding CUDA builds and optional dev/GPU extras.
    grep -v -E '^\s*(flash-attn|vllm|tensordict|torch|openai|ray|transformers|liger-kernel|pre-commit|latex2sympy2_extended|math_verify)\s*($|[<=>])' "${NSPO_VERL_ROOT}/requirements.txt" \
      > "${venv_dir}/requirements-nspo-direct.txt"
    "${pip_bin}" install -c "${constraints_file}" --no-deps -r "${venv_dir}/requirements-nspo-direct.txt"
    "${pip_bin}" install -c "${constraints_file}" --no-deps --no-build-isolation -e "${NSPO_VERL_ROOT}"
    return 0
  fi
  "${pip_bin}" install --upgrade pip wheel setuptools packaging
  # vllm pins a compatible torch, so it is installed first.
  "${pip_bin}" install -c "${constraints_file}" "vllm==${NSPO_VLLM_VERSION}"
  "${pip_bin}" install pyvers
  "${pip_bin}" install "tensordict>=0.8.0,<=0.9.1,!=0.9.0"
  "${pip_bin}" install -c "${constraints_file}" "ray[default]==${NSPO_RAY_VERSION}"
  # safe_reward.py talks to the Guard over the OpenAI client, which NSPO's
  # requirements.txt does not list.
  "${pip_bin}" install openai
  # flash-attn is an optional CUDA optimization. Never invoke its setup.py:
  # on Primus that path downloads a GitHub wheel and fails when GitHub is
  # unreachable. The launcher falls back to torch SDPA when no binary wheel is
  # available, preserving a functional NSPO run.
  if ! "${pip_bin}" install --only-binary=:all: --no-deps flash-attn >/dev/null 2>&1; then
    echo "[WARN] No flash-attn binary wheel is available from the configured index; using torch SDPA." >&2
    NSPO_ATTN_IMPL="sdpa"
    NSPO_USE_REMOVE_PADDING=0
    export NSPO_ATTN_IMPL NSPO_USE_REMOVE_PADDING
  fi
  grep -v -E '^\s*(flash-attn|vllm|tensordict|openai|ray|transformers)' "${NSPO_VERL_ROOT}/requirements.txt" \
    > "${venv_dir}/requirements-nspo.txt"
  "${pip_bin}" install -c "${constraints_file}" -r "${venv_dir}/requirements-nspo.txt"
  "${pip_bin}" install -c "${constraints_file}" -e "${NSPO_VERL_ROOT}"
}

if [ -n "${NSPO_PYTHON_BIN}" ]; then
  if ! probe_nspo_interpreter "${NSPO_PYTHON_BIN}"; then
    echo "[ERROR] The interpreter given by --python/NSPO_PYTHON_BIN cannot run verl 0.5 + vllm ${NSPO_VLLM_VERSION}." >&2
    exit 1
  fi
else
  if probe_nspo_interpreter python3; then
    echo "[INFO] The default interpreter can run verl 0.5 + vllm ${NSPO_VLLM_VERSION}; no venv is needed."
    NSPO_PYTHON_BIN="$(command -v python3)"
  elif [ -x "${NSPO_VENV_DIR}/bin/python" ] && probe_nspo_interpreter "${NSPO_VENV_DIR}/bin/python"; then
    echo "[INFO] Reusing the existing NSPO venv with vllm ${NSPO_VLLM_VERSION}: ${NSPO_VENV_DIR}"
    NSPO_PYTHON_BIN="${NSPO_VENV_DIR}/bin/python"
  elif [ "${NSPO_BUILD_VENV}" = "1" ]; then
    build_nspo_venv "${NSPO_VENV_DIR}"
    NSPO_PYTHON_BIN="${NSPO_VENV_DIR}/bin/python"
    nspo_probe_output=""
    if ! nspo_probe_output="$(probe_nspo_interpreter "${NSPO_PYTHON_BIN}" 2>&1)"; then
      echo "${nspo_probe_output}" >&2
      if [ "${NSPO_USE_SYSTEM_PACKAGES}" = "1" ] && [ "${NSPO_AUTO_ISOLATE_ON_BROKEN_ENV}" = "1" ]; then
        echo "[WARN] NSPO system-package environment is incompatible; rebuilding an isolated venv." >&2
        echo "[WARN] This avoids the image's empty transformer-engine package and old tensordict." >&2
        NSPO_USE_SYSTEM_PACKAGES=0
        export NSPO_USE_SYSTEM_PACKAGES
        if [ -n "${NSPO_ORIGINAL_PYTHONPATH}" ]; then
          export PYTHONPATH="${NSPO_ORIGINAL_PYTHONPATH}"
        else
          unset PYTHONPATH
        fi
        build_nspo_venv "${NSPO_VENV_DIR}"
        if ! probe_nspo_interpreter "${NSPO_PYTHON_BIN}"; then
          echo "[ERROR] The isolated NSPO venv still cannot run verl 0.5 + vllm ${NSPO_VLLM_VERSION}." >&2
          echo "[ERROR] The failure is now inside ${NSPO_VENV_DIR}; inspect its install log and package versions." >&2
          exit 1
        fi
      else
        echo "[ERROR] The NSPO venv cannot run verl 0.5 + vllm ${NSPO_VLLM_VERSION}." >&2
        echo "[ERROR] Set NSPO_USE_SYSTEM_PACKAGES=0 for a clean isolated environment." >&2
        exit 1
      fi
    fi
  else
    echo "[ERROR] No interpreter can run NSPO's verl 0.5 + vllm ${NSPO_VLLM_VERSION}." >&2
    echo "[ERROR] NSPO needs tensordict>=0.8.0 and vllm==${NSPO_VLLM_VERSION}; the parent repository pins tensordict<=0.6.2." >&2
    echo "[ERROR] Either set NSPO_BUILD_VENV=1 to build a clean venv, or prepare one manually:" >&2
    echo "[ERROR]   python3 -m venv ${NSPO_VENV_DIR}   # no --system-site-packages" >&2
    echo "[ERROR]   ${NSPO_VENV_DIR}/bin/pip install vllm==${NSPO_VLLM_VERSION}" >&2
    echo "[ERROR]   ${NSPO_VENV_DIR}/bin/pip install --only-binary=:all: --no-deps flash-attn  # optional; SDPA is the default" >&2
    echo "[ERROR]   ${NSPO_VENV_DIR}/bin/pip install -e ${NSPO_VERL_ROOT}" >&2
    echo "[ERROR] then rerun with --python ${NSPO_VENV_DIR}/bin/python" >&2
    exit 1
  fi
fi
export NSPO_PYTHON_BIN
if [[ "${TRAIN_LOGGER_BACKENDS:-[wandb]}" == '["console","tensorboard"]' ]]; then
  "${NSPO_PYTHON_BIN}" -m pip install tensorboard==2.20.0 numpy==1.26.4
fi
echo "[INFO] NSPO interpreter: ${NSPO_PYTHON_BIN}"

if [ -z "${NSPO_GUARD_PYTHON_BIN}" ]; then
  NSPO_GUARD_PYTHON_BIN="${NSPO_PYTHON_BIN}"
fi
export NSPO_GUARD_PYTHON_BIN
echo "[INFO] NSPO Guard interpreter: ${NSPO_GUARD_PYTHON_BIN}"

if ! interpreter_ready "${NSPO_PYTHON_BIN}" "${POLICY_MODEL_PATH}" "NSPO training"; then
  exit 1
fi

# =========================
# Reward Guard service
# =========================
nspo_guard_pid=""
guard_service_pid=""
cleanup_api_services() {
  if [ -n "${nspo_guard_pid}" ]; then
    stop_service_pids "${nspo_guard_pid}"
  fi
  if [ -n "${guard_service_pid}" ]; then
    stop_service_pids "${guard_service_pid}"
  fi
}
trap cleanup_api_services EXIT

if [ "${RUN_TRAIN}" = "1" ]; then
  require_positive_int NSPO_GUARD_TP_SIZE "${NSPO_GUARD_TP_SIZE}"
  require_positive_int NSPO_GUARD_PORT "${NSPO_GUARD_PORT}"
  nspo_guard_gpu_array=()
  IFS=',' read -r -a nspo_guard_gpu_array <<< "${NSPO_GUARD_GPU_IDS}"
  if [ "${#nspo_guard_gpu_array[@]}" -ne "${NSPO_GUARD_TP_SIZE}" ]; then
    echo "[ERROR] NSPO_GUARD_GPU_IDS has ${#nspo_guard_gpu_array[@]} GPUs; expected ${NSPO_GUARD_TP_SIZE}." >&2
    exit 2
  fi

  if ! interpreter_ready "${NSPO_GUARD_PYTHON_BIN}" "${NSPO_GUARD_MODEL_PATH}" "reward Guard"; then
    exit 1
  fi

  nspo_guard_pid="$(start_vllm_service "${NSPO_GUARD_PYTHON_BIN}" \
    "${NSPO_GUARD_MODEL_PATH}" \
    "${NSPO_GUARD_MODEL}" \
    "${NSPO_GUARD_GPU_IDS}" \
    "${NSPO_GUARD_TP_SIZE}" \
    "${NSPO_GUARD_PORT}" \
    "${API_LOG_DIR}/${METHOD_TAG}_reward_guard_vllm.log" \
    --max-model-len "${NSPO_GUARD_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${NSPO_GUARD_GPU_MEMORY_UTILIZATION:-0.6}")"

  wait_for_api "${NSPO_GUARD_PORT}" "NSPO Reward Guard" 3000 "${nspo_guard_pid}" "${API_LOG_DIR}/${METHOD_TAG}_reward_guard_vllm.log"
  wait_for_model "${NSPO_GUARD_PORT}" "NSPO Reward Guard" "${NSPO_GUARD_MODEL}"

  export NSPO_GUARD_API_BASE="http://127.0.0.1:${NSPO_GUARD_PORT}/v1"

  # =========================
  # Training
  # =========================
  echo "[INFO] ============================================="
  echo "[INFO] Starting NSPO training"
  echo "[INFO] ============================================="
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" \
  NSPO_PYTHON_BIN="${NSPO_PYTHON_BIN}" \
  bash "${NSPO_ROOT}/script/nspo_grpo_v3.sh"

  # The reward Guard is no longer needed once training ends.
  if [ -n "${nspo_guard_pid}" ]; then
    stop_service_pids "${nspo_guard_pid}"
    nspo_guard_pid=""
    wait_for_port_release "${NSPO_GUARD_PORT}" || true
  fi
else
  # No rollouts are generated, so the reward Guard service is never started and
  # its GPU stays free for the evaluation stages below.
  echo "[INFO] ============================================="
  echo "[INFO] Skipping training (RUN_TRAIN=0)"
  echo "[INFO] Resuming from the merge stage in ${CKPT_PATH}"
  echo "[INFO] ============================================="
fi

# =========================
# Merge
# =========================
echo "[INFO] ============================================="
echo "[INFO] Merging All Checkpoints"
echo "[INFO] ============================================="
merge_checkpoints verl05

# =========================
# Safety evaluation
# =========================
if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
  require_positive_int GUARD_TP_SIZE "${GUARD_TP_SIZE}"
  require_positive_int GUARD_PORT "${GUARD_PORT}"
  guard_gpu_array=()
  IFS=',' read -r -a guard_gpu_array <<< "${GUARD_GPU_IDS}"
  if [ "${#guard_gpu_array[@]}" -ne "${GUARD_TP_SIZE}" ]; then
    echo "[ERROR] GUARD_GPU_IDS has ${#guard_gpu_array[@]} GPUs; expected ${GUARD_TP_SIZE}." >&2
    exit 2
  fi

  if ! interpreter_ready "${NSPO_GUARD_PYTHON_BIN}" "${GUARD_MODEL_PATH}" "Qwen3Guard"; then
    exit 1
  fi

  guard_service_pid="$(start_vllm_service "${NSPO_GUARD_PYTHON_BIN}" \
    "${GUARD_MODEL_PATH}" \
    "${GUARD_SERVED_MODEL_NAME}" \
    "${GUARD_GPU_IDS}" \
    "${GUARD_TP_SIZE}" \
    "${GUARD_PORT}" \
    "${API_LOG_DIR}/${METHOD_TAG}_final_safety_guard_vllm.log" \
    --gpu-memory-utilization "${GUARD_GPU_MEMORY_UTILIZATION:-0.8}")"

  wait_for_api "${GUARD_PORT}" "Final Qwen3Guard Service" 3000 "${guard_service_pid}" "${API_LOG_DIR}/${METHOD_TAG}_final_safety_guard_vllm.log"
  wait_for_model "${GUARD_PORT}" "Final Qwen3Guard Service" "${GUARD_SERVED_MODEL_NAME}"

  export SAFETY_GUARD_API_BASE="http://127.0.0.1:${GUARD_PORT}/v1"
  export SAFETY_GUARD_MODEL="${GUARD_SERVED_MODEL_NAME}"

  if [ "${BASELINE_PROMPT_SOURCE:-raw_question}" = "instruction" ]; then
    SAFETY_EVAL_THINK=True
  else
    SAFETY_EVAL_THINK=False
  fi
  run_safety_eval "${SAFETY_EVAL_THINK}"
else
  echo "[INFO] Final safety evaluation disabled."
fi

if [ -n "${guard_service_pid}" ]; then
  stop_service_pids "${guard_service_pid}"
  guard_service_pid=""
  wait_for_port_release "${GUARD_PORT}" || true
fi

# =========================
# MMLU / GSM8K
# =========================
if [ "${FINAL_GENERAL_EVAL_ENABLE}" = "1" ]; then
  run_baseline_general_eval
else
  echo "[INFO] Final MMLU/GSM8K evaluation disabled."
fi

echo "[INFO] train_val_nspo.sh completed successfully."
