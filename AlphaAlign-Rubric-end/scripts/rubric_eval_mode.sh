#!/usr/bin/env bash
# Explicit recovery mode, configured via task environment (not startup command).
rubric_configure_eval_mode() {
  # Primus passes business configuration as JSON, not environment variables.
  # Read only our allowlisted fields; never source/eval arbitrary task content.
  if [ -n "${PRIMUS_JOB_CONFIG_FILE_PATH:-}" ]; then
    local settings key value
    settings="$(python3 - "${PRIMUS_JOB_CONFIG_FILE_PATH}" <<'PY'
import json
import os
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if not isinstance(config, dict):
    raise SystemExit("[ERROR] Primus task configuration must be a JSON object.")
if "rubric_eval" in config:
    recovery = config["rubric_eval"]
    if not isinstance(recovery, dict) or type(recovery.get("enabled")) is not bool:
        raise SystemExit("[ERROR] rubric_eval.enabled must be a JSON boolean.")
    values = {"RUBRIC_EVAL_ONLY": "1" if recovery["enabled"] else "0"}
    if recovery["enabled"]:
        step = recovery.get("step")
        if type(step) is not int or step < 0:
            raise SystemExit("[ERROR] rubric_eval.step must be a non-negative integer.")
        source_kind = recovery.get("source", "dataset")
        if source_kind not in ("dataset", "actor_model"):
            raise SystemExit("[ERROR] rubric_eval.source must be dataset or actor_model.")
        if source_kind == "actor_model":
            source = os.environ.get("PRIMUS_SOURCE_ACTOR_CHECKPOINT_DIR", "")
            iteration = os.environ.get("PRIMUS_SOURCE_ACTOR_CHECKPOINT_ITERATION_DIRNAME", "")
            if source and iteration:
                if os.path.isabs(iteration) or ".." in iteration.split("/"):
                    raise SystemExit("[ERROR] Invalid actor model iteration directory.")
                candidate = os.path.join(source, iteration)
                if os.path.isdir(candidate):
                    source = candidate
        else:
            source = os.environ.get("PRIMUS_OTHER_DATA_PATH", "")
        source = recovery.get("source_dir", source)
        if not isinstance(source, str) or not source or any(c in source for c in "\n\r\0"):
            raise SystemExit("[ERROR] rubric_eval needs an existing source_dir or the selected Primus model/dataset mount.")
        values.update(RUBRIC_EVAL_STEP=str(step), RUBRIC_EVAL_SOURCE_DIR=source)
    for key, value in values.items():
        print(f"{key}={value}")
PY
    )" || return 2
    while IFS='=' read -r key value; do
      [ -z "${key}" ] || export "${key}=${value}"
    done <<< "${settings}"
  fi
  export RUBRIC_EVAL_ONLY="${RUBRIC_EVAL_ONLY:-0}"
  case "${RUBRIC_EVAL_ONLY}" in
    0) return 0 ;;
    1) ;;
    *) echo "[ERROR] RUBRIC_EVAL_ONLY must be 0 or 1." >&2; return 2 ;;
  esac
  if [ -z "${RUBRIC_EVAL_SOURCE_DIR:-}" ] || [ ! -d "${RUBRIC_EVAL_SOURCE_DIR}" ]; then
    echo "[ERROR] Eval-only requires RUBRIC_EVAL_SOURCE_DIR: an existing mounted checkpoint root." >&2
    return 2
  fi
  # Require an explicit numbered checkpoint, avoiding fallback to stale results.
  if ! [[ "${RUBRIC_EVAL_STEP:-}" =~ ^[0-9]+$ ]]; then
    echo "[ERROR] Eval-only requires a numeric RUBRIC_EVAL_STEP." >&2
    return 2
  fi
  local actor_dir="${RUBRIC_EVAL_SOURCE_DIR}/global_step_${RUBRIC_EVAL_STEP}/actor"
  if [ ! -d "${actor_dir}" ]; then
    if [ -d "${RUBRIC_EVAL_SOURCE_DIR}/actor" ]; then
      actor_dir="${RUBRIC_EVAL_SOURCE_DIR}/actor"
    else
      # Primus may mount only the exported actor directory, without its parents.
      actor_dir="${RUBRIC_EVAL_SOURCE_DIR}"
    fi
  fi
  local rank_zero world_size rank
  local shards=("${actor_dir}"/model_world_size_*_rank_0.pt)
  if [ "${#shards[@]}" -ne 1 ] || [ ! -s "${shards[0]}" ]; then
    echo "[ERROR] Missing or ambiguous FSDP rank-0 shard: ${actor_dir}" >&2
    return 2
  fi
  rank_zero="${shards[0]##*/}"
  world_size="${rank_zero#model_world_size_}"
  world_size="${world_size%_rank_0.pt}"
  if ! [[ "${world_size}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] Invalid FSDP world size: ${rank_zero}" >&2
    return 2
  fi
  for ((rank=0; rank<world_size; rank++)); do
    if [ ! -s "${actor_dir}/model_world_size_${world_size}_rank_${rank}.pt" ]; then
      echo "[ERROR] Missing or empty FSDP model shard for rank ${rank}: ${actor_dir}" >&2
      return 2
    fi
  done
  if [ "${FINAL_SAFETY_EVAL_ENABLE:-1}" = "0" ] && [ "${FINAL_GENERAL_EVAL_ENABLE:-1}" = "0" ]; then
    echo "[ERROR] Eval-only cannot disable both evaluations." >&2
    return 2
  fi
  export FINAL_SAFETY_EVAL_STEPS="${RUBRIC_EVAL_STEP}"
  export FINAL_EVAL_MIN_STEP=0
  export RUBRIC_EVAL_ACTOR_DIR="${actor_dir}"
  echo "[EVAL_ONLY] Verified ${world_size} nonempty model shards for step ${RUBRIC_EVAL_STEP}: ${actor_dir}"
}
