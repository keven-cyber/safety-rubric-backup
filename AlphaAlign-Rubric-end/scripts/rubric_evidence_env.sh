#!/usr/bin/env bash
# Source once; child launchers inherit the same run ID.
# Detailed training evidence is opt-in: synchronous per-rule OSS appends can
# serialize judge workers. Keep run metadata, scalar metrics and final evals.
export RUBRIC_INTERMEDIATE_RECORD_ENABLE="${RUBRIC_INTERMEDIATE_RECORD_ENABLE:-0}"
case "${RUBRIC_INTERMEDIATE_RECORD_ENABLE}" in
  0)
    export ROLLOUT_TRACE_ENABLE=0
    export RUBRIC_JUDGE_API_LOG_ENABLE=0
    unset RUBRIC_JUDGE_TRACE_DIR
    ;;
  1) ;;
  *) echo "[ERROR] RUBRIC_INTERMEDIATE_RECORD_ENABLE must be 0 or 1." >&2; exit 1 ;;
esac
if [ "${RUBRIC_EVIDENCE_INITIALIZED:-0}" != "1" ]; then
  if [ -n "${PRIMUS_OSS_WORKSPACE_DIR:-}" ]; then
    if [ ! -d "${PRIMUS_OSS_WORKSPACE_DIR}" ] || [ ! -w "${PRIMUS_OSS_WORKSPACE_DIR}" ]; then
      echo "[ERROR] Primus OSS workspace is not an existing writable mount." >&2
      exit 1
    fi
    evidence_root="${RUBRIC_ARTIFACT_ROOT:-${PRIMUS_OSS_WORKSPACE_DIR}/rubricalign_runs}"
  elif [ -d /primus_oss ] && [ -w /primus_oss ]; then
    evidence_root="${RUBRIC_ARTIFACT_ROOT:-/primus_oss/rubricalign_runs}"
  elif [ "${RUBRIC_REQUIRE_OSS:-0}" = "1" ]; then
    echo "[ERROR] No Primus OSS mount. Set PRIMUS_OSS_WORKSPACE_DIR to the real mounted path." >&2
    exit 1
  else
    evidence_root="${RUBRIC_ARTIFACT_ROOT:-${PROJECT_ROOT}/artifacts/rubricalign_runs}"
    echo "[EVIDENCE] No OSS mount: recording locally at ${evidence_root}." >&2
  fi
  evidence_init_args=(init --root "${evidence_root}" --repo "${PROJECT_ROOT}")
  if [ -n "${PRIMUS_JOB_UUID:-}" ]; then
    stable_run_id="${PRIMUS_JOB_UUID}"
    if [ -n "${RUBRIC_TARGETED_EXPERIMENT_ID:-}" ]; then
      stable_run_id="${stable_run_id}_${RUBRIC_TARGETED_EXPERIMENT_ID}"
    fi
    if [ "${RUBRIC_FROZEN_REVIEW_ENABLE:-0}" = "1" ]; then
      stable_run_id="${stable_run_id}_new_reevaluation"
    fi
    # Primus direct restart preserves PRIMUS_JOB_UUID.  Qualify the stable
    # directory by commit so pod failover on the same code can resume, while a
    # deliberate branch update starts an isolated auditable run instead of
    # repeatedly failing the commit-drift guard.
    evidence_git_commit="$(git -C "${PROJECT_ROOT}" rev-parse --short=12 HEAD)"
    if [ -z "${evidence_git_commit}" ]; then
      echo "[ERROR] Could not determine the repository commit for evidence isolation." >&2
      exit 1
    fi
    stable_run_id="${stable_run_id}_g${evidence_git_commit}"
    evidence_init_args+=(--run-id "${stable_run_id}" --resume-existing)
  fi
  RUBRIC_RUN_DIR="$(python3 "${PROJECT_ROOT}/scripts/run_evidence.py" "${evidence_init_args[@]}")"
  export RUBRIC_RUN_DIR
  export RUBRIC_RUN_ID="${RUBRIC_RUN_DIR##*/}"
  export ROLLOUT_TRACE_DIR="${RUBRIC_RUN_DIR}/rollouts"
  export EVAL_OUTPUT_DIR="${RUBRIC_RUN_DIR}/eval_outputs"
  export API_LOG_DIR="${RUBRIC_RUN_DIR}/services"
  export RUBRIC_EVIDENCE_INITIALIZED=1
  mkdir -p "${API_LOG_DIR}" "${EVAL_OUTPUT_DIR}"
  echo "[EVIDENCE] Run directory: ${RUBRIC_RUN_DIR}"
fi
# Keep online-evolution writers and parent-process smoke validation on the
# same persistent run directory. Child shells cannot export this value back to
# train_val.sh, so it must be established while the shared evidence
# environment is sourced by the parent.
#
# A Primus failover reuses the job's OSS directory. Give every smoke attempt a
# fresh evolution directory so a previous failed attempt cannot either mask a
# missing dispatch or overwrite the new attempt's counters/audit.
if [ "${RUBRIC_EVOLVE_SMOKE_TEST:-0}" = "1" ]; then
  if [ -z "${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID:-}" ]; then
    RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID="$(date -u +%Y%m%dT%H%M%SZ)_p$$"
    export RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID
  fi
  default_evolve_save_dir="${RUBRIC_RUN_DIR}/online_rubric_evolution_smoke/${RUBRIC_EVOLVE_SMOKE_ATTEMPT_ID}"
else
  default_evolve_save_dir="${RUBRIC_RUN_DIR}/online_rubric_evolution"
fi
export RUBRIC_EVOLVE_SAVE_DIR="${RUBRIC_EVOLVE_SAVE_DIR:-${default_evolve_save_dir}}"
if [ "${RUBRIC_EVOLVE:-0}" = "1" ]; then
  # Primus OSS does not support flock(2). Keep the mutable replay and its lock
  # on the single-node local filesystem; the Python writer snapshots it to OSS.
  export RUBRIC_EVOLVE_REPLAY_DIR="${RUBRIC_EVOLVE_REPLAY_DIR:-${RUBRIC_EVOLVE_LOCAL_STATE_ROOT:-/tmp/rubricalign_evolve}/${RUBRIC_RUN_ID}}"
  mkdir -p "${RUBRIC_EVOLVE_REPLAY_DIR}"
  python3 - "${RUBRIC_EVOLVE_REPLAY_DIR}" "${RUBRIC_EVOLVE_SAVE_DIR}" <<'PY'
import fcntl
import os
import sys
from pathlib import Path

local_dir, artifact_dir = map(Path, sys.argv[1:])
artifact_dir.mkdir(parents=True, exist_ok=True)
lock_path = local_dir / "contrastive_replay.json.lock"
with lock_path.open("a+") as handle:
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    fcntl.flock(handle, fcntl.LOCK_UN)
probe = artifact_dir / f".evolution_storage_probe_{os.getpid()}"
temporary = probe.with_suffix(".tmp")
temporary.write_text("ok", encoding="utf-8")
temporary.replace(probe)
probe.unlink()
PY
  echo "[EVIDENCE] Evolution replay lock directory: ${RUBRIC_EVOLVE_REPLAY_DIR}"
fi
if [ "${RUBRIC_INTERMEDIATE_RECORD_ENABLE}" = "1" ]; then
  export RUBRIC_JUDGE_TRACE_DIR="${RUBRIC_RUN_DIR}/judge"
  mkdir -p "${ROLLOUT_TRACE_DIR}" "${RUBRIC_JUDGE_TRACE_DIR}"
fi
if [ "${RUBRIC_LIGHT_DIAGNOSTICS_ENABLE:-0}" = "1" ]; then
  mkdir -p "${RUBRIC_RUN_DIR}/validation_diagnostics"
fi
echo "[EVIDENCE] Intermediate sample/request recording: ${RUBRIC_INTERMEDIATE_RECORD_ENABLE}"
echo "[EVIDENCE] Lightweight checkpoint diagnostics: ${RUBRIC_LIGHT_DIAGNOSTICS_ENABLE:-0}"
