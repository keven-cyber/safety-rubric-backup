#!/usr/bin/env bash
# API-assisted rubric revision with unchanged questions, followed by training.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo 'Usage: bash scripts/train_api_rubric.sh (optional SAFETY_PIPELINE_ENV_FILE override)'
  echo 'Uses qwen3.7-plus and the configured quota; archives revisions on Primus OSS.'
  echo 'Questions are unchanged unless RUBRIC_ADAPTIVE_SEEDS/STRATEGIES are set.'
  echo 'Final evaluation: 900, 990, best (unique steps only).'
  exit 0
fi
export SAFETY_PIPELINE_ENV_FILE="${SAFETY_PIPELINE_ENV_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}"
[[ -f "${SAFETY_PIPELINE_ENV_FILE}" ]] || { echo '[ERROR] Missing API config.' >&2; exit 2; }
export REWARD_MODE=rubric_safe_helpful_balanced
export RUBRIC_GUARD_ENABLE=0 RUBRIC_EVOLVE=0
export TRAINING_DATA_VERSION=new_v3 TRAIN_PROMPT_VERSION=structured_steps
export RUBRIC_JUDGE_GRANULARITY=single_rule RUBRIC_JUDGE_STRICT_PARSE=1
export RUBRIC_JUDGE_GUIDED_OUTPUT=1
export RUBRIC_JUDGE_DROP_FAILED=1
export RUBRIC_REQUIRE_OSS=1
export RUBRIC_INTERMEDIATE_RECORD_ENABLE=0
# The historical 2,640-row configuration ends at 990; targeted arms may run
# beyond 990 while retaining the same comparison checkpoints.
export TRAIN_TOTAL_EPOCHS=3 TRAIN_SAVE_FREQ=30 TRAIN_TEST_FREQ=30 TRAIN_SAVE_BEST_CKPT=True
export FINAL_SAFETY_EVAL_STEPS=900,990,best FINAL_EVAL_MIN_STEP=0
export FINAL_EVAL_EXACT_TRIO=1
export FINAL_SAFETY_EVAL_ENABLE=1 FINAL_GENERAL_EVAL_ENABLE=1
export RUBRIC_DATASET_DIR_NAME=api_revised_rubric
source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"
export SOURCE_DATA="${SOURCE_DATA:-${PROJECT_ROOT}/Evaluation/updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1}"
# Source, API evidence and revised dataset remain in the same immutable run directory.
export PIPELINE_WORK="${RUBRIC_RUN_DIR}/rubric_api"
export RUBRIC_BUILD_OUTPUT="${RUBRIC_RUN_DIR}/revised_dataset"
export RUBRIC_PILOT_OUTPUT="${PIPELINE_WORK}/generated.jsonl"
export RUBRIC_PILOT_VERIFIED_OUTPUT="${PIPELINE_WORK}/verified.jsonl"
export RUBRIC_PILOT_REVIEW_OUTPUT="${PIPELINE_WORK}/review.csv"
export RUBRIC_PILOT_MODEL=qwen3.7-plus RUBRIC_VERIFIER_MODEL=qwen3.7-plus RUBRIC_FACTUAL_MODEL=qwen3.7-plus
export RUBRIC_PILOT_SCOPE="${RUBRIC_PILOT_SCOPE:-all}"
export RUBRIC_PILOT_LIMIT="${RUBRIC_PILOT_LIMIT:-0}"
export RUBRIC_REVIEW_FRACTION="${RUBRIC_REVIEW_FRACTION:-1.0}"
# Backward-compatible default: RUBRIC_ADAPTIVE_SEEDS=0. Targeted launchers may
# override it before entering this script.
export RUBRIC_ADAPTIVE_SEEDS="${RUBRIC_ADAPTIVE_SEEDS:-0}"
export RUBRIC_ADAPTIVE_VAL_SEEDS="${RUBRIC_ADAPTIVE_VAL_SEEDS:-0}"
export RUBRIC_ADAPTIVE_STRATEGIES="${RUBRIC_ADAPTIVE_STRATEGIES:-gcg_style_adversarial_suffix,pair_style_iterative_persuasion}"
export RUBRIC_BUILD_DATASET="${RUBRIC_BUILD_DATASET:-1}"
# Start verification as soon as generated candidates are durable instead of
# waiting for all 2,640 generation jobs to finish. The two pools share the
# configured API quota, so keep their concurrency independently tunable.
export RUBRIC_STREAM_VERIFY="${RUBRIC_STREAM_VERIFY:-1}"
export RUBRIC_GENERATION_CONCURRENCY="${RUBRIC_GENERATION_CONCURRENCY:-4}"
export RUBRIC_VERIFICATION_CONCURRENCY="${RUBRIC_VERIFICATION_CONCURRENCY:-4}"
export RUBRIC_JSONL_CHECKPOINT_EVERY="${RUBRIC_JSONL_CHECKPOINT_EVERY:-25}"
export RUBRIC_PROGRESS_EVERY="${RUBRIC_PROGRESS_EVERY:-10}"
SOURCE_DATA_INPUT="${SOURCE_DATA}"
SOURCE_CACHE="${RUBRIC_RUN_DIR}/source_dataset"
source_cache_complete=1
for split in train val; do
  for filename in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
    [ -f "${SOURCE_CACHE}/${split}/${filename}" ] || source_cache_complete=0
  done
done
[ -f "${SOURCE_CACHE}/manifest.json" ] || source_cache_complete=0
if [ "${source_cache_complete}" = "1" ]; then
  for split in train val; do
    for filename in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
      cmp -s "${SOURCE_DATA_INPUT}/${split}/${filename}" "${SOURCE_CACHE}/${split}/${filename}" || {
        echo "[ERROR] Cached source dataset differs from the configured source: ${split}/${filename}" >&2
        exit 2
      }
    done
  done
  cmp -s "${SOURCE_DATA_INPUT}/manifest.json" "${SOURCE_CACHE}/manifest.json" || {
    echo "[ERROR] Cached source manifest differs from the configured source." >&2
    exit 2
  }
  echo "[RESUME] Reusing complete source dataset cache: ${SOURCE_CACHE}"
elif [ -e "${SOURCE_CACHE}" ]; then
  echo "[ERROR] Refusing to reuse incomplete source dataset cache: ${SOURCE_CACHE}" >&2
  exit 2
else
  source_cache_stage="${RUBRIC_RUN_DIR}/.source_dataset.$$.tmp"
  mkdir -p "${source_cache_stage}"
  cp -R "${SOURCE_DATA}/train" "${SOURCE_DATA}/val" "${SOURCE_DATA}/manifest.json" "${source_cache_stage}/"
  mv "${source_cache_stage}" "${SOURCE_CACHE}"
fi
export SOURCE_DATA="${SOURCE_CACHE}"

build_complete=1
for split in train val; do
  for filename in benign_sampled.parquet sensitive_sampled.parquet harm_sampled.parquet adaptive_harm.parquet; do
    [ -f "${RUBRIC_BUILD_OUTPUT}/${split}/${filename}" ] || build_complete=0
  done
done
[ -f "${RUBRIC_BUILD_OUTPUT}/manifest.json" ] || build_complete=0
if [ "${build_complete}" = "1" ]; then
  echo "[RESUME] Reusing complete API-built dataset: ${RUBRIC_BUILD_OUTPUT}"
elif [ -e "${RUBRIC_BUILD_OUTPUT}" ]; then
  echo "[ERROR] Refusing to reuse incomplete API-built dataset: ${RUBRIC_BUILD_OUTPUT}" >&2
  exit 2
else
  bash "${PROJECT_ROOT}/scripts/run_rubric_pilot.sh"
fi
python3 "${PROJECT_ROOT}/scripts/audit_api_rubric_artifact.py" \
  --source "${SOURCE_DATA}" --output "${RUBRIC_BUILD_OUTPUT}" --work "${PIPELINE_WORK}"
export FILE_PATH="${RUBRIC_BUILD_OUTPUT}"
# Never auto-resume a checkpoint trained on a different rubric.
export CKPT_PATH="${CKPT_PATH:-${PRIMUS_SAVE_CHECKPOINT_DIR:-${RUBRIC_RUN_DIR}/checkpoints}/api_rubric/${RUBRIC_RUN_ID}}"
exec bash "${PROJECT_ROOT}/train_val.sh" "$@"
