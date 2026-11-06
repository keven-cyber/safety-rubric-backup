#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_BASE="$(dirname "$(dirname "${PROJECT_ROOT}")")"

PIPELINE_WORK="${PIPELINE_WORK:-$(dirname "${PROJECT_ROOT}")/safety_pipeline_work}"
SOURCE_DATA="${SOURCE_DATA:-${PROJECT_ROOT}/Evaluation/updated_training_rubric_dataset_independent_v3}"
RUBRIC_PILOT_LIMIT="${RUBRIC_PILOT_LIMIT:-0}"
RUBRIC_PILOT_CONCURRENCY="${RUBRIC_PILOT_CONCURRENCY:-4}"
RUBRIC_GENERATION_CONCURRENCY="${RUBRIC_GENERATION_CONCURRENCY:-${RUBRIC_PILOT_CONCURRENCY}}"
RUBRIC_VERIFICATION_CONCURRENCY="${RUBRIC_VERIFICATION_CONCURRENCY:-${RUBRIC_PILOT_CONCURRENCY}}"
RUBRIC_STREAM_VERIFY="${RUBRIC_STREAM_VERIFY:-0}"
RUBRIC_STREAM_VERIFY_POLL_SECONDS="${RUBRIC_STREAM_VERIFY_POLL_SECONDS:-5}"
RUBRIC_PILOT_MODEL="${RUBRIC_PILOT_MODEL:-qwen3.7-plus}"
RUBRIC_VERIFIER_MODEL="${RUBRIC_VERIFIER_MODEL:-qwen3.7-plus}"
RUBRIC_FACTUAL_MODEL="${RUBRIC_FACTUAL_MODEL:-${RUBRIC_VERIFIER_MODEL}}"
RUBRIC_GENERATION_VALIDATION_ATTEMPTS="${RUBRIC_GENERATION_VALIDATION_ATTEMPTS:-3}"
RUBRIC_VERIFICATION_VALIDATION_ATTEMPTS="${RUBRIC_VERIFICATION_VALIDATION_ATTEMPTS:-3}"
RUBRIC_ADAPTIVE_REGENERATION_ATTEMPTS="${RUBRIC_ADAPTIVE_REGENERATION_ATTEMPTS:-2}"
RUBRIC_RUBRIC_REPAIR_ATTEMPTS="${RUBRIC_RUBRIC_REPAIR_ATTEMPTS:-2}"
RUBRIC_PIPELINE_MAX_ROUNDS="${RUBRIC_PIPELINE_MAX_ROUNDS:-8}"
RUBRIC_PIPELINE_RETRY_DELAY="${RUBRIC_PIPELINE_RETRY_DELAY:-15}"
RUBRIC_PILOT_SCOPE="${RUBRIC_PILOT_SCOPE:-all}"
RUBRIC_REVIEW_FRACTION="${RUBRIC_REVIEW_FRACTION:-1.0}"
RUBRIC_ADAPTIVE_SEEDS="${RUBRIC_ADAPTIVE_SEEDS:-120}"
RUBRIC_ADAPTIVE_VAL_SEEDS="${RUBRIC_ADAPTIVE_VAL_SEEDS:-0}"
RUBRIC_ADAPTIVE_STRATEGIES="${RUBRIC_ADAPTIVE_STRATEGIES:-gcg_style_adversarial_suffix,pair_style_iterative_persuasion}"
RUBRIC_BUILD_OUTPUT="${RUBRIC_BUILD_OUTPUT:-${PROJECT_ROOT}/Evaluation/updated_training_rubric_dataset_independent_v3_enriched_gcg_pair_v1}"
RUBRIC_BUILD_DATASET="${RUBRIC_BUILD_DATASET:-1}"
MODEL_SLUG="${RUBRIC_PILOT_MODEL//[^A-Za-z0-9._-]/_}"
LIMIT_SLUG="${RUBRIC_PILOT_LIMIT}"
if [ "${RUBRIC_PILOT_LIMIT}" = "0" ]; then
  LIMIT_SLUG="all"
fi

case "${RUBRIC_PILOT_SCOPE}" in
  required)
    OUTPUT_PREFIX="rubric_pilot"
    SELECTION_ARGS=(--job-kind rubric_review --required-repair-only)
    ;;
  complete)
    OUTPUT_PREFIX="rubric_improvement_pilot"
    SELECTION_ARGS=(--job-kind rubric_review --complete-review-only)
    ;;
  all)
    OUTPUT_PREFIX="safety_pipeline"
    SELECTION_ARGS=(--job-kind all)
    ;;
  *)
    echo "[ERROR] RUBRIC_PILOT_SCOPE must be required, complete, or all." >&2
    exit 2
    ;;
esac

SERVER_ENV_FILE="${WORKSPACE_BASE}/agentic_reward/reward_system/rewards/utils/env.yaml"
LOCAL_REFERENCE_ENV_FILE="${WORKSPACE_BASE}/code/rl_benchmark/agentic_bench_ruip_v2/agentic_reward/reward_system/rewards/utils/env.yaml"
if [ -n "${SAFETY_PIPELINE_ENV_FILE:-}" ]; then
  ENV_FILE="${SAFETY_PIPELINE_ENV_FILE}"
elif [ -f "${SERVER_ENV_FILE}" ]; then
  ENV_FILE="${SERVER_ENV_FILE}"
elif [ -f "${LOCAL_REFERENCE_ENV_FILE}" ]; then
  ENV_FILE="${LOCAL_REFERENCE_ENV_FILE}"
else
  echo "[ERROR] API env.yaml was not found." >&2
  echo "[ERROR] Set SAFETY_PIPELINE_ENV_FILE to its absolute path." >&2
  exit 2
fi

PLAN_PATH="${PIPELINE_WORK}/plan.jsonl"
REPORT_PATH="${PIPELINE_WORK}/plan_report.json"
RESULTS_PATH="${RUBRIC_PILOT_OUTPUT:-${PIPELINE_WORK}/${OUTPUT_PREFIX}${LIMIT_SLUG}_${MODEL_SLUG}.jsonl}"
VERIFIED_RESULTS_PATH="${RUBRIC_PILOT_VERIFIED_OUTPUT:-${RESULTS_PATH%.jsonl}.verified.jsonl}"
REVIEW_PATH="${RUBRIC_PILOT_REVIEW_OUTPUT:-${VERIFIED_RESULTS_PATH%.jsonl}.review.csv}"

mkdir -p "${PIPELINE_WORK}"

python3 "${PROJECT_ROOT}/scripts/safety_data_pipeline.py" plan \
  --dataset-root "${SOURCE_DATA}" \
  --plan-output "${PLAN_PATH}" \
  --report-output "${REPORT_PATH}" \
  --review-fraction "${RUBRIC_REVIEW_FRACTION}" \
  --adaptive-seeds "${RUBRIC_ADAPTIVE_SEEDS}" \
  --adaptive-val-seeds "${RUBRIC_ADAPTIVE_VAL_SEEDS}" \
  --adaptive-strategies "${RUBRIC_ADAPTIVE_STRATEGIES}"

run_generation() {
  python3 "${PROJECT_ROOT}/scripts/safety_data_pipeline.py" generate \
    --plan "${PLAN_PATH}" \
    --output "${RESULTS_PATH}" \
    --env-file "${ENV_FILE}" \
    --model "${RUBRIC_PILOT_MODEL}" \
    "${SELECTION_ARGS[@]}" \
    --stratify-by profile \
    --limit "${RUBRIC_PILOT_LIMIT}" \
    --validation-attempts "${RUBRIC_GENERATION_VALIDATION_ATTEMPTS}" \
    --concurrency "${RUBRIC_GENERATION_CONCURRENCY}"
}

run_verification() {
  python3 "${PROJECT_ROOT}/scripts/safety_data_pipeline.py" verify \
    --plan "${PLAN_PATH}" \
    --results "${RESULTS_PATH}" \
    --output "${VERIFIED_RESULTS_PATH}" \
    --env-file "${ENV_FILE}" \
    --model "${RUBRIC_VERIFIER_MODEL}" \
    --factual-model "${RUBRIC_FACTUAL_MODEL}" \
    --validation-attempts "${RUBRIC_VERIFICATION_VALIDATION_ATTEMPTS}" \
    --adaptive-regeneration-attempts "${RUBRIC_ADAPTIVE_REGENERATION_ATTEMPTS}" \
    --rubric-repair-attempts "${RUBRIC_RUBRIC_REPAIR_ATTEMPTS}" \
    --concurrency "${RUBRIC_VERIFICATION_CONCURRENCY}"
}

run_generation_and_verification() {
  if [ "${RUBRIC_STREAM_VERIFY}" != "1" ]; then
    run_generation
    run_verification
    return
  fi

  echo "[INFO] Streaming rubric generation and verification"
  echo "[INFO] Generation concurrency: ${RUBRIC_GENERATION_CONCURRENCY}"
  echo "[INFO] Verification concurrency: ${RUBRIC_VERIFICATION_CONCURRENCY}"
  run_generation &
  generation_pid=$!

  while kill -0 "${generation_pid}" 2>/dev/null; do
    if [ -s "${RESULTS_PATH}" ]; then
      verification_status=0
      run_verification || verification_status=$?
      if [ "${verification_status}" -ne 0 ]; then
        echo "[ERROR] Rubric verification failed with status ${verification_status}." >&2
        kill "${generation_pid}" 2>/dev/null || true
        wait "${generation_pid}" 2>/dev/null || true
        return "${verification_status}"
      fi
      if kill -0 "${generation_pid}" 2>/dev/null; then
        sleep "${RUBRIC_STREAM_VERIFY_POLL_SECONDS}"
      fi
    else
      sleep "${RUBRIC_STREAM_VERIFY_POLL_SECONDS}"
    fi
  done

  generation_status=0
  wait "${generation_pid}" || generation_status=$?
  if [ "${generation_status}" -ne 0 ]; then
    echo "[ERROR] Rubric generation failed with status ${generation_status}." >&2
    return "${generation_status}"
  fi

  # Pick up candidates produced after the last verifier snapshot.
  run_verification
}

run_review() {
  python3 "${PROJECT_ROOT}/scripts/safety_data_pipeline.py" review \
    --plan "${PLAN_PATH}" \
    --results "${VERIFIED_RESULTS_PATH}" \
    --output "${REVIEW_PATH}" \
    --only-present
}

run_build() {
  python3 "${PROJECT_ROOT}/scripts/safety_data_pipeline.py" build \
    --dataset-root "${SOURCE_DATA}" \
    --plan "${PLAN_PATH}" \
    --results "${VERIFIED_RESULTS_PATH}" \
    --output-root "${RUBRIC_BUILD_OUTPUT}" \
    --accept-reviewed-rubrics
}

build_full_dataset=0
if [ "${RUBRIC_PILOT_SCOPE}" = "all" ] && [ "${RUBRIC_PILOT_LIMIT}" = "0" ] && [ "${RUBRIC_BUILD_DATASET}" = "1" ]; then
  build_full_dataset=1
fi

if [ "${build_full_dataset}" = "1" ]; then
  round=1
  while [ "${round}" -le "${RUBRIC_PIPELINE_MAX_ROUNDS}" ]; do
    echo "[INFO] Pipeline recovery round ${round}/${RUBRIC_PIPELINE_MAX_ROUNDS}"
    run_generation_and_verification
    run_review
    if run_build; then
      echo "[INFO] Built training dataset: ${RUBRIC_BUILD_OUTPUT}"
      break
    else
      build_status=$?
    fi
    if [ -e "${RUBRIC_BUILD_OUTPUT}" ]; then
      echo "[ERROR] Build left an output path and cannot be retried safely: ${RUBRIC_BUILD_OUTPUT}" >&2
      exit "${build_status}"
    fi
    if [ "${round}" -ge "${RUBRIC_PIPELINE_MAX_ROUNDS}" ]; then
      echo "[ERROR] Dataset is still incomplete after ${RUBRIC_PIPELINE_MAX_ROUNDS} rounds." >&2
      exit "${build_status}"
    fi
    echo "[WARN] Dataset is not complete; retrying only unresolved jobs after ${RUBRIC_PIPELINE_RETRY_DELAY}s."
    sleep "${RUBRIC_PIPELINE_RETRY_DELAY}"
    round=$((round + 1))
  done
else
  run_generation_and_verification
  run_review
fi

echo "[INFO] First-pass rubric results: ${RESULTS_PATH}"
echo "[INFO] Verified rubric results: ${VERIFIED_RESULTS_PATH}"
echo "[INFO] Review CSV: ${REVIEW_PATH}"
