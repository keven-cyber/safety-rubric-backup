#!/usr/bin/env bash
# One-click Primus launcher for the AlphaAlign baseline on the RubricAlign v3 dataset.
#
# Mirrors train_val.sh stage for stage: download models, train, merge the selected
# checkpoints, run the shared safety evaluation, then run MMLU/GSM8K. The scoring
# code is the parent repository's Evaluation pipeline, so AlphaAlign and
# RubricAlign numbers are produced by identical judges.
#
# Differences from train_val.sh, all inherent to the method:
#   * reward is the FsfairX preference model held in-process by verl, so no Judge
#     replicas are started and no Guard is needed during training;
#   * prompts are the plain user requests (extra_info.raw_question), so the safety
#     evaluation generates with Think=False;
#   * AlphaAlign's verl has no in-training best-checkpoint selection, so MMLU and
#     GSM8K use the explicit baseline general-evaluation coverage.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RUBRIC_PROJECT_ROOT="${SCRIPT_DIR}"
ALPHAALIGN_ROOT="${ALPHAALIGN_ROOT:-${SCRIPT_DIR}/AlphaAlign}"
export METHOD_TAG="${METHOD_TAG:-alphaalign}"
# Dataset fallback is intentionally disabled: a missing enriched corpus must
# fail the job instead of silently changing the comparison set.
export RUBRIC_DATASET_FALLBACK_DIR_NAME=""

# shellcheck source=scripts/baseline_common.sh
source "${SCRIPT_DIR}/scripts/baseline_common.sh"

POLICY_MODEL_ID="${POLICY_MODEL_ID:-Qwen/Qwen2.5-3B-Instruct}"
POLICY_MODEL_DIR="${POLICY_MODEL_DIR:-}"
SKIP_POLICY_DOWNLOAD="${SKIP_POLICY_DOWNLOAD:-0}"
REWARD_MODEL_ID="${REWARD_MODEL_ID:-sfairXC/FsfairX-LLaMA3-RM-v0.1}"
REWARD_MODEL_DIR="${REWARD_MODEL_DIR:-}"
SKIP_REWARD_MODEL_DOWNLOAD="${SKIP_REWARD_MODEL_DOWNLOAD:-0}"
MODEL_DOWNLOAD_SOURCE="${MODEL_DOWNLOAD_SOURCE:-hf_mirror}"
HF_MIRROR_ENDPOINT="${HF_MIRROR_ENDPOINT:-https://hf-mirror.com}"
GUARD_MODEL_ID="${GUARD_MODEL_ID:-Qwen/Qwen3Guard-Gen-8B}"
GUARD_SERVED_MODEL_NAME="${GUARD_SERVED_MODEL_NAME:-qwen3_guard_gen_8b}"
GUARD_GPU_IDS="${GUARD_GPU_IDS:-3}"
GUARD_TP_SIZE="${GUARD_TP_SIZE:-1}"
GUARD_PORT="${GUARD_PORT:-8001}"
# The evaluation Guard is the same 8B snapshot the pre-training step downloads,
# so a complete local copy is reused instead of re-fetched on every run.
FORCE_GUARD_DOWNLOAD="${FORCE_GUARD_DOWNLOAD:-0}"

export TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-4,5,6,7}"
export ALL_JOB_GPU_IDS="${ALL_JOB_GPU_IDS:-0,1,2,3,4,5,6,7}"

export VLLM_USE_CUSTOM_ALLREDUCE=0
export VLLM_LOGGING_LEVEL=WARNING

FINAL_SAFETY_EVAL_ENABLE="$(normalize_bool FINAL_SAFETY_EVAL_ENABLE "${FINAL_SAFETY_EVAL_ENABLE:-1}")"
FINAL_GENERAL_EVAL_ENABLE="$(normalize_bool FINAL_GENERAL_EVAL_ENABLE "${FINAL_GENERAL_EVAL_ENABLE:-1}")"
# Stage gate. The pipeline is linear, so an evaluation that dies used to force a
# full retrain on relaunch. RUN_TRAIN=0 (or --eval-only) keeps CKPT_PATH pointed
# at the existing checkpoints and resumes from the merge stage onward.
RUN_TRAIN="$(normalize_bool RUN_TRAIN "${RUN_TRAIN:-1}")"
FORCE_GUARD_DOWNLOAD="$(normalize_bool FORCE_GUARD_DOWNLOAD "${FORCE_GUARD_DOWNLOAD}")"
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
Usage: bash train_val_alphaalign.sh [options]

Options:
  -p, --policy-model <model_id>   Policy model id for the Hugging Face mirror download
      --policy-dir <dir>          Policy download directory (default: $MODEL_BASE_PATH/policy)
      --policy-path <dir>         Existing local policy path used for training
      --skip-policy-download      Skip the policy download
      --reward-model <model_id>   Preference model id (default: sfairXC/FsfairX-LLaMA3-RM-v0.1)
      --reward-model-path <dir>   Existing local preference model path
      --eval-only                 Skip training and merge/evaluate the checkpoints
                                  already present in CKPT_PATH (same as RUN_TRAIN=0)
  -h, --help                      Show this help

Environment:
  BASELINE_PROMPT_SOURCE          raw_question (default) | instruction
  MODEL_DOWNLOAD_SOURCE           hf_mirror (default); all models use this source
  HF_MIRROR_ENDPOINT              Hugging Face mirror endpoint (default: https://hf-mirror.com)
  TRAIN_GPU_IDS                   Policy GPU ids (default: 4,5,6,7)
  GUARD_GPU_IDS                   Qwen3Guard GPU ids for the final safety eval (default: 3)
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
  FINAL_SAFETY_EVAL_ENABLE        Run the final Qwen3Guard safety evaluation (default: 1)
  FINAL_GENERAL_EVAL_ENABLE       Run MMLU and GSM8K (default: 1)
  BASELINE_GENERAL_EVAL_STEP      Evaluate this step (last scope only)
  BASELINE_GENERAL_EVAL_SCOPE     last (default) | selected (same models as safety)
  BASELINE_REQUIRED_FINAL_STEPS   Required numbered coverage, e.g. 900,990,1080,1170
  RUN_TRAIN                       Run the training stage (default: 1). With 0 the
                                  launcher requires an existing checkpoint in
                                  CKPT_PATH and starts at the merge stage, which is
                                  how an interrupted evaluation is resumed.
  RESUME_TRAINING                 Continue training inside an existing CKPT_PATH
                                  instead of branching to a fresh directory (default: 0)
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
    --reward-model)
      REWARD_MODEL_ID="${2:?missing value for --reward-model}"
      shift 2
      ;;
    --reward-model-path)
      export REWARD_MODEL_PATH="${2:?missing value for --reward-model-path}"
      SKIP_REWARD_MODEL_DOWNLOAD=1
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

# This job has already completed training and merging. Its startup command is
# immutable, so recover the exact evaluated artifact before any training setup.
if [ "${PRIMUS_JOB_UUID:-}" = "primus7703244e7ab3f1a9a8d69026cz" ]; then
  exec bash "${SCRIPT_DIR}/Evaluation/recover_alphaalign_primus770.sh"
fi

export POLICY_MODEL_ID

if [ ! -d "${ALPHAALIGN_ROOT}/verl" ]; then
  echo "[ERROR] AlphaAlign checkout not found at ${ALPHAALIGN_ROOT}" >&2
  exit 1
fi

if ! python3 -c 'import fire' >/dev/null 2>&1; then
  pip install fire
fi
if ! python3 -c 'import tqdm' >/dev/null 2>&1; then
  pip install tqdm
fi
# AlphaAlign's reward scorer imports mathruler.grader, which its requirements.txt
# does not list. Without it the first reward call crashes mid-training.
if ! python3 -c 'import mathruler' >/dev/null 2>&1; then
  pip install mathruler
fi
# transformers 4.57.x requires huggingface-hub>=0.34.0,<1.0. Checking only
# whether the module imports is insufficient because an installed 1.x does
# import successfully but makes transformers reject the environment.
if ! python3 - <<'PY' >/dev/null 2>&1
from importlib.metadata import version

parts = tuple(int(part) for part in version("huggingface-hub").split(".")[:2])
raise SystemExit(0 if (0, 34) <= parts < (1, 0) else 1)
PY
then
  python3 -m pip install "huggingface_hub==${HF_HUB_PIN:-0.36.2}"
fi
ray stop --force

export VERL_HOME="${VERL_HOME:-/root/code}"
export MODEL_BASE_PATH="${MODEL_BASE_PATH:-/root/models}"
if [ -z "${POLICY_MODEL_DIR}" ]; then
  POLICY_MODEL_DIR="${MODEL_BASE_PATH}/policy"
fi
if [ -z "${REWARD_MODEL_DIR}" ]; then
  REWARD_MODEL_DIR="${MODEL_BASE_PATH}/reward_model"
fi
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${SCRIPT_DIR}/checkpoints/${METHOD_TAG}}}"
# A persisted Primus checkpoint directory is never resumed implicitly. When a
# prior run is present, this launcher creates a fresh sibling directory; set
# RESUME_TRAINING=1 explicitly when continuation is intended.
RESUME_TRAINING="$(normalize_bool RESUME_TRAINING "${RESUME_TRAINING:-0}")"
# Persist service logs in the Primus OSS workspace alongside eval_outputs.
export API_LOG_DIR="${API_LOG_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/api_logs}"
export EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${PRIMUS_OSS_WORKSPACE_DIR:-/tmp}/eval_outputs}"
RAY_TMPDIR_ROOT="${RAY_TMPDIR_ROOT:-/tmp/ray_temp}"
# Use an isolated directory per job. Deleting the shared Ray directory can
# block for a long time after a previous Primus run and has no visible output.
export RAY_TMPDIR="${RAY_TMPDIR:-${RAY_TMPDIR_ROOT}/${METHOD_TAG}_${PRIMUS_TASK_ID:-local}_$$}"
# The inner AlphaAlign launcher honors this flag. The directory above is new
# for this job, so recursively clearing it again is unnecessary and can block
# when a caller explicitly supplied a shared RAY_TMPDIR.
export CLEAR_RAY_TMPDIR="${CLEAR_RAY_TMPDIR:-0}"

mkdir -p "${MODEL_BASE_PATH}/guard" "${POLICY_MODEL_DIR}" "${REWARD_MODEL_DIR}" \
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
echo "[INFO] AlphaAlign baseline, one-click Primus run"
echo "[INFO] ============================================="
echo "[INFO] Model Base Path: ${MODEL_BASE_PATH}"
echo "[INFO] Checkpoint Path: ${CKPT_PATH}"
echo "[INFO] Stages: train=${RUN_TRAIN} safety_eval=${FINAL_SAFETY_EVAL_ENABLE} general_eval=${FINAL_GENERAL_EVAL_ENABLE}"
echo "[INFO] Eval Output Dir: ${EVAL_OUTPUT_DIR}"
echo "[INFO] API Log Dir: ${API_LOG_DIR}"
echo "[INFO] ============================================="

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

if [ "${MODEL_DOWNLOAD_SOURCE}" != "hf_mirror" ]; then
  echo "[ERROR] MODEL_DOWNLOAD_SOURCE must be hf_mirror; ModelScope is unavailable on some Primus nodes." >&2
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

download_reward_model_from_hf_mirror() {
  download_model_from_hf_mirror "${REWARD_MODEL_ID}" "${REWARD_MODEL_DIR}" "Preference model"
}

verify_reward_model() {
  python3 - "${REWARD_MODEL_PATH}" <<'PY'
import glob
import json
import os
import sys

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

model_path = os.path.abspath(sys.argv[1])
config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)

weight_files = []
index_files = glob.glob(os.path.join(model_path, "*.safetensors.index.json"))
index_files += glob.glob(os.path.join(model_path, "pytorch_model.bin.index.json"))
for index_path in index_files:
    with open(index_path, encoding="utf-8") as handle:
        weight_map = json.load(handle).get("weight_map", {})
    for filename in set(weight_map.values()):
        shard = os.path.join(model_path, filename)
        if not os.path.isfile(shard) or os.path.getsize(shard) == 0:
            raise RuntimeError(f"missing or empty weight shard: {shard}")
        weight_files.append(shard)

if not weight_files:
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        weight_files.extend(glob.glob(os.path.join(model_path, pattern)))
if not weight_files or any(os.path.getsize(path) == 0 for path in weight_files):
    raise RuntimeError(f"no complete model weights found under {model_path}")

try:
    model_class = AutoModelForSequenceClassification._model_mapping[type(config)]
except KeyError as exc:
    raise RuntimeError(
        f"Transformers does not support sequence classification for model_type={config.model_type}"
    ) from exc

print(
    f"[INFO] Preference model verified: model_type={config.model_type}, "
    f"loader={model_class.__name__}, weight_files={len(set(weight_files))}"
)
PY
}

if [ "${RUN_TRAIN}" = "0" ]; then
  # The preference model is the in-process training reward; no evaluation stage
  # loads it, so downloading 8B of weights here would be pure waste.
  echo "[INFO] Skip preference model download (RUN_TRAIN=0; training-only dependency)"
elif [ "${SKIP_REWARD_MODEL_DOWNLOAD}" = "1" ]; then
  echo "[INFO] Skip preference model download (SKIP_REWARD_MODEL_DOWNLOAD=1)"
else
  download_reward_model_from_hf_mirror
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
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${REWARD_MODEL_DIR}}"
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
if [ "${RUN_TRAIN}" = "1" ]; then
  if [ ! -f "${REWARD_MODEL_PATH}/config.json" ]; then
    echo "[ERROR] Preference model is incomplete: ${REWARD_MODEL_PATH}/config.json is missing." >&2
    echo "[ERROR] ${REWARD_MODEL_ID} could not be downloaded completely from ${HF_MIRROR_ENDPOINT}." >&2
    exit 1
  fi
  if ! verify_reward_model; then
    echo "[ERROR] Preference model download is incomplete or incompatible: ${REWARD_MODEL_PATH}" >&2
    exit 1
  fi
fi
if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ] && [ ! -f "${GUARD_MODEL_PATH}/config.json" ]; then
  echo "[ERROR] Final-evaluation Guard is incomplete: ${GUARD_MODEL_PATH}/config.json is missing." >&2
  exit 1
fi

echo "[INFO] POLICY_MODEL_PATH=${POLICY_MODEL_PATH}"
echo "[INFO] REWARD_MODEL_PATH=${REWARD_MODEL_PATH}"
echo "[INFO] GUARD_MODEL_PATH=${GUARD_MODEL_PATH}"

guard_service_pid=""
cleanup_api_services() {
  if [ -n "${guard_service_pid}" ]; then
    stop_service_pids "${guard_service_pid}"
  fi
}
trap cleanup_api_services EXIT

# =========================
# Training
# =========================
if [ "${RUN_TRAIN}" = "1" ]; then
  echo "[INFO] ============================================="
  echo "[INFO] Starting AlphaAlign GRPO training"
  echo "[INFO] ============================================="
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU_IDS}" \
  bash "${ALPHAALIGN_ROOT}/examples/ppo_trainer/alphaalign-grpo-v3.sh"
else
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
merge_checkpoints verl02

# =========================
# Safety evaluation
# =========================
if [ "${FINAL_SAFETY_EVAL_ENABLE}" = "1" ]; then
  if ! [[ "${GUARD_TP_SIZE}" =~ ^[1-9][0-9]*$ ]] || ! [[ "${GUARD_PORT}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] GUARD_TP_SIZE and GUARD_PORT must be positive integers." >&2
    exit 2
  fi
  guard_gpu_array=()
  IFS=',' read -r -a guard_gpu_array <<< "${GUARD_GPU_IDS}"
  if [ "${#guard_gpu_array[@]}" -ne "${GUARD_TP_SIZE}" ]; then
    echo "[ERROR] GUARD_GPU_IDS has ${#guard_gpu_array[@]} GPUs; expected ${GUARD_TP_SIZE}." >&2
    exit 2
  fi

  if ! interpreter_ready python3 "${GUARD_MODEL_PATH}" "Qwen3Guard"; then
    exit 1
  fi

  guard_service_pid="$(start_vllm_service python3 \
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

  # AlphaAlign is trained to answer inside <safety_reasoning>/<answer>/\boxed{},
  # so it is generated under its own training-time instruction. eval_answer.py
  # already extracts <answer>...</answer>, which both methods produce.
  export SAFETY_EVAL_PROMPT_TEMPLATE="$(python3 -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}/scripts')
from prepare_baseline_prompt_dataset import ALPHAALIGN_INSTRUCTION
print(ALPHAALIGN_INSTRUCTION + ' {prompt}', end='')
")"
  if [ -z "${SAFETY_EVAL_PROMPT_TEMPLATE}" ]; then
    echo "[ERROR] Could not build the AlphaAlign evaluation prompt template." >&2
    exit 1
  fi

  run_safety_eval False
else
  echo "[INFO] Final safety evaluation disabled."
fi

# Release the Guard before loading policy checkpoints for the general evaluation.
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

echo "[INFO] train_val_alphaalign.sh completed successfully."
