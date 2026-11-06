"""Checkpoint online reward state together with the policy, without importing ML runtimes."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time


def _enabled():
    return os.getenv("RUBRIC_EVOLVE", "0").lower() in {"1", "true", "yes", "on"}


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_rubric_state(checkpoint, step, epoch):
    if not _enabled():
        return
    source = Path(os.environ["RUBRIC_EVOLVE_SAVE_DIR"])
    checkpoint = Path(checkpoint)
    target = checkpoint / "rubric_evolution"
    staging = Path(tempfile.mkdtemp(prefix=".rubric-state-", dir=str(checkpoint)))
    files = {}
    for path in sorted(source.glob("*")):
        if path.is_file() and path.suffix in {".json", ".jsonl"}:
            shutil.copy2(path, staging / path.name)
            files[path.name] = _digest(staging / path.name)
    manifest = {"schema_version": 1, "global_step": int(step), "epoch": int(epoch), "files": files}
    (staging / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")
    if target.exists():
        target.rename(checkpoint / f".rubric-state-previous-{time.time_ns()}")
    staging.rename(target)


def restore_rubric_state(checkpoint, step):
    if not _enabled():
        return None
    snapshot = Path(checkpoint) / "rubric_evolution"
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            "This training checkpoint has no synchronized rubric state. Use it for evaluation, "
            "or start a fresh evolution run; refusing to resume it with a later OSS rule bank."
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("global_step") != int(step):
        raise RuntimeError("Rubric snapshot does not match the policy checkpoint")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or not (snapshot / name).is_file() or _digest(snapshot / name) != digest:
            raise RuntimeError(f"Incomplete or corrupt rubric checkpoint state: {name}")
    destination = Path(os.environ["RUBRIC_EVOLVE_SAVE_DIR"])
    destination.mkdir(parents=True, exist_ok=True)
    history = destination / "restore_history" / str(time.time_ns())
    history.mkdir(parents=True)
    # Preserve later observations for diagnosis, but never train with them after rollback.
    for path in list(destination.iterdir()):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".parquet"}:
            path.rename(history / path.name)
    for name in manifest["files"]:
        shutil.copy2(snapshot / name, destination / name)
    local_replay = os.getenv("RUBRIC_EVOLVE_REPLAY_DIR", "").strip()
    if local_replay:
        local = Path(local_replay)
        local.mkdir(parents=True, exist_ok=True)
        replay = local / "contrastive_replay.json"
        if replay.exists():
            replay.rename(history / "local_contrastive_replay.json")
        persistent = destination / "contrastive_replay_snapshot.json"
        if persistent.is_file():
            shutil.copy2(persistent, replay)
    return int(manifest["epoch"])
