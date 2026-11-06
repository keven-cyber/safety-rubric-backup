#!/usr/bin/env bash
# Exactly the twelve authorized fresh arms; model preparation precedes services.
set -euo pipefail
# Per-process package transport: retain the selected internal index and avoid
# inherited unreachable extra indexes. A nonempty whitespace value parses as
# an empty list in pip; no TLS checks or installed versions are changed.
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.org/simple/}"
export PIP_EXTRA_INDEX_URL=" "
export PIP_DISABLE_PIP_VERSION_CHECK=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
key="${1:?Usage: train_cross_model_twelve_p1.sh model_family_method}"
[[ $# = 1 ]] || { echo 'No unreviewed experiment arguments allowed' >&2; exit 2; }
case "${key}" in
  llama32_3b_*) model=meta-llama/Llama-3.2-3B-Instruct ;;
  qwen25_7b_*) model=Qwen/Qwen2.5-7B-Instruct ;;
  qwen3_4b_*) model=Qwen/Qwen3-4B ;;
  qwen3_8b_*) model=Qwen/Qwen3-8B ;;
  *) echo 'Unknown policy model' >&2; exit 2 ;;
esac
method="${key##*_}"
case "${method}" in alphaalign|nspo|p1) ;; *) echo 'Unknown method' >&2; exit 2 ;; esac
case "${key}" in "${key%_*}_${method}") ;; *) exit 2 ;; esac
[[ "${key}" =~ ^(llama32_3b|qwen25_7b|qwen3_4b|qwen3_8b)_(alphaalign|nspo|p1)$ ]] || exit 2
[[ "${PRIMUS_JOB_UUID:?Missing actual job ID}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ "${PRIMUS_JOB_SUBMIT_TIMESTAMP:?Missing submission timestamp}" =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
for setting in RUBRIC_RECOVERY_SOURCE_CHECKPOINT RUBRIC_EVAL_SOURCE_DIR TRAIN_RESUME_FROM_PATH; do
  [[ -z "${!setting:-}" ]] || { echo "Fresh training forbids ${setting}" >&2; exit 2; }
done
export TRAIN_RESUME_MODE=disable RESUME_TRAINING=0 RUN_TRAIN=1
export RUBRIC_FROZEN_REVIEW_ENABLE=0 RUBRIC_EVAL_ONLY=0 RUBRIC_EVAL_USE_RECOVERY_COPY=0
export RUBRIC_VALIDATION_REPLAY_ENABLE=0 RUBRIC_CHECKPOINT_RECOVERY_ENABLE=0
unset RUBRIC_EVAL_STEP BASELINE_GENERAL_EVAL_STEP
export POLICY_MODEL_ID="${model}"
export CROSS_MODEL_EXPERIMENT_KEY="${key}"
export MODEL_SCOPE_PIN=1.37.1
export NSPO_NULLSPACE_DATASET_REVISION="$(python3 "${SCRIPT_DIR}/cross_model_dependency_revision.py" tatsu-lab/alpaca HF)"
export GUARD_MODEL_REVISION="$(python3 "${SCRIPT_DIR}/cross_model_dependency_revision.py" Qwen/Qwen3Guard-Gen-8B ModelScope)"
export JUDGE_MODEL_REVISION="$(python3 "${SCRIPT_DIR}/cross_model_dependency_revision.py" Qwen/Qwen3-32B ModelScope)"
if [[ "${model}" = Qwen/Qwen3-* ]]; then
  export POLICY_ENABLE_THINKING=false
else
  unset POLICY_ENABLE_THINKING
fi
run="${PRIMUS_OSS_WORKSPACE_DIR:?Missing persistent OSS workspace}/cross_model_twelve_p1/${key}/${PRIMUS_JOB_UUID}_${PRIMUS_JOB_SUBMIT_TIMESTAMP}"
[[ ! -e "${run}" ]] || { echo 'Existing submission evidence; refuse an implicit duplicate/fresh restart' >&2; exit 2; }
mkdir -p "$(dirname "${run}")"
mkdir "${run}"
export CROSS_MODEL_RUN_DIR="${run}"
export TRAIN_LOGGER_BACKENDS='["console","tensorboard"]'
export TENSORBOARD_DIR="${PRIMUS_TENSORBOARD_LOG_DIR:?Missing Primus TensorBoard log directory}/cross_model_twelve_p1/${key}/${PRIMUS_JOB_UUID}_${PRIMUS_JOB_SUBMIT_TIMESTAMP}"
mkdir -p "${TENSORBOARD_DIR}"
exec > >(tee -a "${run}/launcher.log") 2>&1
export MODEL_BASE_PATH="${CROSS_MODEL_NODE_ROOT:-/root/cross_model_twelve_p1}/${key}/${PRIMUS_JOB_UUID}_${PRIMUS_JOB_SUBMIT_TIMESTAMP}/models"
export POLICY_MODEL_PATH="${AUTHORIZED_POLICY_SNAPSHOT_PATH:-${MODEL_BASE_PATH}/policy}"
policy_args=(--model "${model}" --destination "${MODEL_BASE_PATH}/policy" --report "${run}/policy_snapshot.json")
if [[ -n "${AUTHORIZED_POLICY_SNAPSHOT_PATH:-}" ]]; then
  policy_args+=(--existing-authorized-snapshot "${AUTHORIZED_POLICY_SNAPSHOT_PATH}")
fi
python3 "${SCRIPT_DIR}/cross_model_policy.py" "${policy_args[@]}"
export FILE_PATH="$(python3 "${SCRIPT_DIR}/cross_model_dataset.py" --report "${run}/input_dataset.json")"
export RUBRIC_SOURCE_DATASET_DIR="${FILE_PATH}"
export RUBRIC_DATASET_FALLBACK_DIR_NAME=""
export TRAIN_TOTAL_EPOCHS=3 TRAIN_SAVE_FREQ=30 TRAIN_TEST_FREQ=30
export TRAIN_TOTAL_STEPS=1170 TRAIN_TOTAL_TRAINING_STEPS=1170
export TRAIN_BATCH_SIZE=8 ROLLOUT_N=8 ACTOR_LR=1e-6 TRAIN_LEARNING_RATE=1e-6
export TRAIN_MAX_PROMPT_LENGTH=1536 TRAIN_MAX_RESPONSE_LENGTH=2048
export FINAL_SAFETY_EVAL_STEPS=900,990,1080,1170,best
export FINAL_EVAL_EXPECTED_LAST_STEP=1170 FINAL_EVAL_EXACT_TRIO=1
export FINAL_SAFETY_EVAL_ENABLE=1 FINAL_GENERAL_EVAL_ENABLE=1
export BASELINE_GENERAL_EVAL_SCOPE=selected BASELINE_REQUIRED_FINAL_STEPS=900,990,1080,1170
export TRAIN_GPU_IDS=4,5,6,7 TRAIN_N_GPUS_PER_NODE=4 ALL_JOB_GPU_IDS=0,1,2,3,4,5,6,7
export PRIMUS_SAVE_CHECKPOINT_DIR="${run}/checkpoints"
export CKPT_PATH="${PRIMUS_SAVE_CHECKPOINT_DIR}"
export API_LOG_DIR="${run}/services" EVAL_OUTPUT_DIR="${run}/eval_outputs"
export RUBRIC_ARTIFACT_ROOT="${run}/rubricalign_runs"
export BASELINE_RUN_DIR="${run}" BASELINE_FIRST_BATCH_GATE=1
export BASELINE_VALIDATION_ARCHIVE=1 BASELINE_VAL_BATCH_SIZE=304
# Full native304 originals/scores remain mandatory. Avoid the optional W&B
# preview Table: its offline artifact service crashed in the actual first validation.
export BASELINE_LOG_VAL_GENERATIONS=0
export BASELINE_DURABLE_TRACKING_ENABLE=1
# Ray appends session/socket names; job IDs belong in persistent evidence, not AF_UNIX paths.
export RAY_TMPDIR="$(python3 "${SCRIPT_DIR}/cross_model_ray_temp.py")"
export CLEAR_RAY_TMPDIR=0
export NSPO_VENV_DIR="${MODEL_BASE_PATH%/models}/nspo_venv"
export NSPO_VLLM_OVERLAY_DIR="${NSPO_VENV_DIR}/nspo_vllm_overlay"
export NSPO_NULLSPACE_STAGING_ROOT="${MODEL_BASE_PATH%/models}/nspo_nullspace"
export BASELINE_PROMPT_SOURCE=raw_question
mkdir -p "${CKPT_PATH}" "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}" "${RAY_TMPDIR}"
python3 "${SCRIPT_DIR}/cross_model_run_contract.py" --key "${key}" --model "${model}" --method "${method}"
case "${method}" in
  p1)
    exec bash "${SCRIPT_DIR}/train_rubricevolve_agentic_eight.sh" p1_earlytarget --policy-model "${model}" --policy-path "${POLICY_MODEL_PATH}" --skip-policy-download ;;
  alphaalign|nspo)
    export METHOD_TAG="${method}"
    exec bash "${ROOT}/train_val_${method}.sh" --policy-model "${model}" --policy-path "${POLICY_MODEL_PATH}" --skip-policy-download ;;
esac
