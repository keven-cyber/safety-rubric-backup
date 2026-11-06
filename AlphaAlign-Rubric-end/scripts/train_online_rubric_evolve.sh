#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ "${1:-}" = "-h" || "${1:-}" = "--help" ]]; then
  exec bash "${PROJECT_ROOT}/train_val.sh" "$@"
fi

if [ "${RUBRIC_EVOLVE_EXPERIMENT_FULL:-0}" = "1" ]; then
  # E1-E4 are comparable full runs, even if the submitting shell previously
  # held a smoke-test or truncated-training override.
  export RUBRIC_EVOLVE_SMOKE_TEST=0
  unset TRAIN_TOTAL_TRAINING_STEPS
  export TRAIN_LEARNING_RATE=1e-6
  export TRAIN_USE_KL_LOSS=False
  export TRAIN_KL_LOSS_COEF=0.001
  export TRAIN_TOTAL_EPOCHS=3
  export TRAIN_SAVE_FREQ=30
  export TRAIN_TEST_FREQ=30
  export FINAL_SAFETY_EVAL_STEPS=900,990,1080,last,best
  export FINAL_EVAL_EXACT_TRIO=1
  export FINAL_EVAL_EXPECTED_LAST_STEP=1170
  export RUBRIC_EVOLVE_CACHE_REFRESH=1
fi

# Reuse the exact targeted-C corpus that produced RubricAlign best-1080.  The
# Primus asset is mounted below /primus_datasets, but the exact mount directory
# is platform generated.  Resolve it by its immutable manifest and refuse to
# fall back to the older 2,640-row repository corpus.
export RUBRIC_ONLINE_DATASET_NAME="${RUBRIC_ONLINE_DATASET_NAME:-rubricalign-targeted-c-v4-3120}"
export RUBRIC_ONLINE_DATASET_VERSION="${RUBRIC_ONLINE_DATASET_VERSION:-updated_training_rubric_dataset_independent_v4_targeted_hard_negative}"
export RUBRIC_ONLINE_DATASET_TRAIN_ROWS="${RUBRIC_ONLINE_DATASET_TRAIN_ROWS:-3120}"
export RUBRIC_ONLINE_DATASET_VAL_ROWS="${RUBRIC_ONLINE_DATASET_VAL_ROWS:-304}"

resolved_dataset="$({
  FILE_PATH="${FILE_PATH:-}" \
  PRIMUS_TRAIN_DATASET_DIR="${PRIMUS_TRAIN_DATASET_DIR:-}" \
  PRIMUS_DATASETS_DIR="${PRIMUS_DATASETS_DIR:-/primus_datasets}" \
  RUBRIC_ONLINE_DATASET_NAME="${RUBRIC_ONLINE_DATASET_NAME}" \
  RUBRIC_ONLINE_DATASET_VERSION="${RUBRIC_ONLINE_DATASET_VERSION}" \
  RUBRIC_ONLINE_DATASET_TRAIN_ROWS="${RUBRIC_ONLINE_DATASET_TRAIN_ROWS}" \
  RUBRIC_ONLINE_DATASET_VAL_ROWS="${RUBRIC_ONLINE_DATASET_VAL_ROWS}" \
  PROJECT_ROOT="${PROJECT_ROOT}" \
  python3 - <<'PY'
import json
import os
from pathlib import Path

expected_version = os.environ["RUBRIC_ONLINE_DATASET_VERSION"]
expected_train = int(os.environ["RUBRIC_ONLINE_DATASET_TRAIN_ROWS"])
expected_val = int(os.environ["RUBRIC_ONLINE_DATASET_VAL_ROWS"])
dataset_name = os.environ["RUBRIC_ONLINE_DATASET_NAME"]

roots = []
for value in (
    os.environ.get("FILE_PATH"),
    os.environ.get("PRIMUS_TRAIN_DATASET_DIR"),
    os.environ.get("PRIMUS_DATASETS_DIR"),
    str(Path(os.environ["PROJECT_ROOT"]) / "Evaluation" / dataset_name),
):
    if value and value != "-":
        path = Path(value)
        if path not in roots:
            roots.append(path)


def candidates(root: Path):
    if (root / "manifest.json").is_file():
        yield root
    named = root / dataset_name
    if (named / "manifest.json").is_file():
        yield named
    if not root.is_dir():
        return
    # Primus may insert an asset ID or version directory below the mount root.
    for manifest in root.glob("*/manifest.json"):
        yield manifest.parent
    for manifest in root.glob("*/*/manifest.json"):
        yield manifest.parent


checked = []
for root in roots:
    for candidate in candidates(root):
        if candidate in checked:
            continue
        checked.append(candidate)
        try:
            manifest = json.loads((candidate / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("version") != expected_version
            or int(manifest.get("training_rows", -1)) != expected_train
            or int(manifest.get("validation_rows", -1)) != expected_val
        ):
            continue
        required = [
            candidate / split / filename
            for split in ("train", "val")
            for filename in (
                "benign_sampled.parquet",
                "sensitive_sampled.parquet",
                "harm_sampled.parquet",
                "adaptive_harm.parquet",
            )
        ]
        if all(path.is_file() and path.stat().st_size > 0 for path in required):
            print(candidate.resolve())
            raise SystemExit(0)

searched = ", ".join(str(path) for path in roots) or "<none>"
raise SystemExit(
    "[ERROR] Exact RubricAlign targeted-C dataset was not mounted. "
    f"Expected version={expected_version}, train={expected_train}, val={expected_val}; "
    f"searched: {searched}"
)
PY
} 2>&1)" || {
  printf '%s\n' "${resolved_dataset}" >&2
  exit 2
}
export FILE_PATH="${resolved_dataset}"
export RUBRIC_DATASET_DIR_NAME="${RUBRIC_ONLINE_DATASET_NAME}"
echo "[INFO] ONLINE_EVOLUTION_DATASET=${FILE_PATH} (targeted-C, 3120 train / 304 val)"

# Keep the established RubricAlign objective. Evolution begins only after
# policy warm-up and uses live rollout failures; no pre-training rewrite is
# performed by this launcher.
export REWARD_MODE="${REWARD_MODE:-rubric_safe_helpful_balanced}"
export TRAINING_DATA_VERSION="${TRAINING_DATA_VERSION:-new_v3}"
export TRAIN_PROMPT_VERSION="${TRAIN_PROMPT_VERSION:-structured_steps}"
export RUBRIC_GUARD_ENABLE=0
# Match the successful targeted RubricAlign arm: select the checkpoint by the
# safety-constrained validation score rather than average Pareto reward.
export TRAIN_BEST_METRIC_KEY="${TRAIN_BEST_METRIC_KEY:-val-core/balanced/safety_constrained_score}"
export TRAIN_BEST_METRIC_MODE="${TRAIN_BEST_METRIC_MODE:-max}"

export RUBRIC_EVOLVE=1
export RUBRIC_EVOLVE_START_EPOCH="${RUBRIC_EVOLVE_START_EPOCH:-1}"
export RUBRIC_EVOLVE_START_STEP="${RUBRIC_EVOLVE_START_STEP:-360}"
export RUBRIC_EVOLVE_INTERVAL_STEPS="${RUBRIC_EVOLVE_INTERVAL_STEPS:-120}"
export RUBRIC_EVOLVE_STEPS="${RUBRIC_EVOLVE_STEPS:-360,480,600,720,840,960}"
export RUBRIC_EVOLVE_SCOPE="${RUBRIC_EVOLVE_SCOPE:-profile}"
export RUBRIC_EVOLVE_ALLOWED_PROFILES="${RUBRIC_EVOLVE_ALLOWED_PROFILES:-}"
export RUBRIC_EVOLVE_ALLOWED_CATEGORIES="${RUBRIC_EVOLVE_ALLOWED_CATEGORIES:-}"
export RUBRIC_EVOLVE_PROVIDER="${RUBRIC_EVOLVE_PROVIDER:-quota}"
export RUBRIC_EVOLVE_API_MODEL="${RUBRIC_EVOLVE_API_MODEL:-qwen3.7-plus}"
export RUBRIC_EVOLVE_API_AUTH_FILE="${RUBRIC_EVOLVE_API_AUTH_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}"
export RUBRIC_EVOLVE_MAX_TRAJECTORIES="${RUBRIC_EVOLVE_MAX_TRAJECTORIES:-16}"
export RUBRIC_EVOLVE_MIN_LABEL_COUNT="${RUBRIC_EVOLVE_MIN_LABEL_COUNT:-4}"
export RUBRIC_EVOLVE_MIN_CATEGORY_COUNT="${RUBRIC_EVOLVE_MIN_CATEGORY_COUNT:-2}"
export RUBRIC_EVOLVE_MIN_GAP="${RUBRIC_EVOLVE_MIN_GAP:-0.40}"
export RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE="${RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE:-0.80}"
export RUBRIC_EVOLVE_MAX_BAD_PASS_RATE="${RUBRIC_EVOLVE_MAX_BAD_PASS_RATE:-0.20}"
export RUBRIC_EVOLVE_MIN_RULES="${RUBRIC_EVOLVE_MIN_RULES:-0}"
export RUBRIC_EVOLVE_MAX_RULES="${RUBRIC_EVOLVE_MAX_RULES:-2}"
export RUBRIC_EVOLVE_TOTAL_MAX_RULES="${RUBRIC_EVOLVE_TOTAL_MAX_RULES:-4}"
export RUBRIC_EVOLVE_BANK_MAX_RULES="${RUBRIC_EVOLVE_BANK_MAX_RULES:-4}"
export RUBRIC_EVOLVE_MAX_PROFILES_PER_STEP="${RUBRIC_EVOLVE_MAX_PROFILES_PER_STEP:-4}"
export RUBRIC_EVOLVE_TWO_STAGE="${RUBRIC_EVOLVE_TWO_STAGE:-1}"
export RUBRIC_EVOLVE_PRINT="${RUBRIC_EVOLVE_PRINT:-1}"
export RUBRIC_EVOLVE_REQUIRE_AUDIT="${RUBRIC_EVOLVE_REQUIRE_AUDIT:-1}"

# Accumulate independently labeled live trajectories across proposal steps.
# Candidate rules are written from a balanced proposal split and promoted only
# after passing a disjoint balanced holdout. A failed candidate gets one
# evidence-guided rewrite; at most one new rule can enter the reward per step.
export RUBRIC_EVOLVE_REPLAY_ENABLE="${RUBRIC_EVOLVE_REPLAY_ENABLE:-1}"
export RUBRIC_EVOLVE_REPLAY_MAX_PER_LABEL="${RUBRIC_EVOLVE_REPLAY_MAX_PER_LABEL:-32}"
export RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL="${RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL:-8}"
export RUBRIC_EVOLVE_REPLAY_MIN_PROPOSAL_PER_LABEL="${RUBRIC_EVOLVE_REPLAY_MIN_PROPOSAL_PER_LABEL:-3}"
export RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL="${RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL:-4}"
export RUBRIC_EVOLVE_REPAIR_ATTEMPTS="${RUBRIC_EVOLVE_REPAIR_ATTEMPTS:-1}"
export RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP="${RUBRIC_EVOLVE_MAX_PROMOTIONS_PER_STEP:-1}"
export RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN="${RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN:-1}"
export RUBRIC_EVOLVE_BASELINE_PASS_THRESHOLD="${RUBRIC_EVOLVE_BASELINE_PASS_THRESHOLD:-0.75}"
export RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT="${RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT:-2}"

# Evolution must not learn from fabricated zero-padded judge arrays. Keep the
# baseline launcher unchanged; only online-evolution runs retry malformed or
# truncated JSON and mark an unrecoverable result invalid for filtering.
export RUBRIC_JUDGE_STRICT_PARSE="${RUBRIC_JUDGE_STRICT_PARSE:-1}"
export RUBRIC_JUDGE_GUIDED_OUTPUT="${RUBRIC_JUDGE_GUIDED_OUTPUT:-1}"
export RUBRIC_JUDGE_DROP_FAILED="${RUBRIC_JUDGE_DROP_FAILED:-1}"
export RUBRIC_JUDGE_MAX_NEW_TOKENS="${RUBRIC_JUDGE_MAX_NEW_TOKENS:-1024}"
export RUBRIC_JUDGE_REPAIR_MAX_TOKENS="${RUBRIC_JUDGE_REPAIR_MAX_TOKENS:-2048}"

# A bounded smoke run validates online-evolution dispatch and audit persistence,
# not the full validation suite.  RayPPOTrainer always validates on the final
# step when test_freq is positive, so disable that terminal validation here;
# otherwise one malformed Judge batch can prevent the smoke assertions from
# running and AIMaster will repeatedly fail over the pod.
if [ "${RUBRIC_EVOLVE_SMOKE_TEST:-0}" = "1" ]; then
  # A smoke run is a deterministic functional test, not a miniature resume.
  # Force a short fresh schedule whose trigger is guaranteed to be visited.
  export RUBRIC_EVOLVE_START_EPOCH=1
  export RUBRIC_EVOLVE_START_STEP="${RUBRIC_EVOLVE_SMOKE_TRIGGER_STEP:-2}"
  export RUBRIC_EVOLVE_STEPS="${RUBRIC_EVOLVE_START_STEP}"
  export TRAIN_TOTAL_TRAINING_STEPS="${RUBRIC_EVOLVE_SMOKE_TOTAL_STEPS:-4}"
  export TRAIN_RESUME_MODE=disable
  export TRAIN_TEST_FREQ=-1
  echo "[INFO] Smoke run: fresh ${TRAIN_TOTAL_TRAINING_STEPS}-step schedule; evolution trigger=${RUBRIC_EVOLVE_START_STEP}; terminal validation disabled."
fi

# Optional variance-aware rule calibration. It is deliberately opt-in so the
# plain online-evolution arm remains directly comparable with RubricAlign.
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE:-0}"
export RUBRIC_EVOLVE_VARIANCE_START_STEP="${RUBRIC_EVOLVE_VARIANCE_START_STEP:-${RUBRIC_EVOLVE_START_STEP}}"
export RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS="${RUBRIC_EVOLVE_VARIANCE_MIN_ROLLOUTS:-8}"
export RUBRIC_EVOLVE_VARIANCE_ALPHA="${RUBRIC_EVOLVE_VARIANCE_ALPHA:-0.4}"
export RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA="${RUBRIC_EVOLVE_VARIANCE_SAFETY_FAILURE_BETA:-0.3}"
export RUBRIC_EVOLVE_VARIANCE_EMA_DECAY="${RUBRIC_EVOLVE_VARIANCE_EMA_DECAY:-0.9}"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MIN:-0.5}"
export RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX="${RUBRIC_EVOLVE_VARIANCE_WEIGHT_MAX:-1.5}"

# Full rollout/API traces are unnecessary for online learning. The compact
# promotion audit (hashes, rules, pass-rate gaps, and decisions) is always kept.
export RUBRIC_INTERMEDIATE_RECORD_ENABLE="${RUBRIC_INTERMEDIATE_RECORD_ENABLE:-0}"
export ROLLOUT_TRACE_ENABLE="${ROLLOUT_TRACE_ENABLE:-0}"

# Compare every arm at identical fixed checkpoints, the completed last step,
# and the internally selected best. 3120 rows / batch 8 * 3 epochs = 1170.
export TRAIN_SAVE_FREQ="${TRAIN_SAVE_FREQ:-30}"
export TRAIN_TOTAL_EPOCHS="${TRAIN_TOTAL_EPOCHS:-3}"
export FINAL_EVAL_MIN_STEP="${FINAL_EVAL_MIN_STEP:-900}"
export FINAL_SAFETY_EVAL_STEPS="${FINAL_SAFETY_EVAL_STEPS:-900,990,1080,last,best}"
export FINAL_EVAL_EXACT_TRIO="${FINAL_EVAL_EXACT_TRIO:-1}"
export FINAL_EVAL_EXPECTED_LAST_STEP="${FINAL_EVAL_EXPECTED_LAST_STEP:-1170}"

if [ ! -s "${RUBRIC_EVOLVE_API_AUTH_FILE}" ]; then
  echo "[ERROR] Online evolution auth file not found: ${RUBRIC_EVOLVE_API_AUTH_FILE}" >&2
  exit 2
fi

echo "[INFO] Online rubric evolution is enabled during training."
echo "[INFO] Evolution writer=${RUBRIC_EVOLVE_API_MODEL}; proposal steps=${RUBRIC_EVOLVE_STEPS}; two_stage=${RUBRIC_EVOLVE_TWO_STAGE}."

# Explicit fresh-comparison arms are applied after historical wrappers so their
# defaults cannot silently erase the declared experiment. Unset means unchanged.
if [ -n "${RUBRIC_AGENTIC_EIGHT_ARM:-}" ]; then
  agentic_exports="$(python3 "${SCRIPT_DIR}/agentic_eight.py" "${RUBRIC_AGENTIC_EIGHT_ARM}" --shell)"
  eval "${agentic_exports}"
fi

exec bash "${PROJECT_ROOT}/train_val.sh" "$@"
