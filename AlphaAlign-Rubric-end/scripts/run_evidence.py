#!/usr/bin/env python3
"""Create auditable run directories and snapshots on a mounted Primus OSS path."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_versions():
    versions = {}
    for package in ("torch", "vllm", "ray", "transformers", "numpy", "pyarrow", "verl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def save(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def init(root, repo, run_id=None, resume_existing=False):
    root, repo = Path(root).resolve(), Path(repo).resolve()
    if run_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9._-]{1,160}", run_id):
            raise ValueError("run_id must contain only letters, digits, dot, underscore, or dash")
        directory = root / run_id
    else:
        directory = root / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:10]
        )
    if directory.exists():
        if not resume_existing:
            raise FileExistsError(f"run directory already exists: {directory}")
        manifest_path = directory / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"cannot resume incomplete run directory: {directory}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_commit = git(repo, "rev-parse", "HEAD")
        if manifest.get("git_commit") != current_commit:
            raise ValueError(
                "refusing to resume with a different git commit: "
                f"stored={manifest.get('git_commit')}, current={current_commit}"
            )
        save(directory / "status.json", {"phase": "resuming", "updated_at": now()})
        return directory
    directory.mkdir(parents=True, exist_ok=False)
    versions = package_versions()
    # An explicit allowlist prevents API keys and credential-bearing URLs being copied.
    allowed = ("REWARD_MODE", "TRAINING_DATA_VERSION", "RUBRIC_DATASET_DIR_NAME", "TRAIN_PROMPT_VERSION",
               "RUBRIC_JUDGE_GRANULARITY", "RUBRIC_JUDGE_STRICT_PARSE", "RUBRIC_GUARD_ENABLE",
               "TRAIN_TOTAL_EPOCHS", "TRAIN_SAVE_FREQ", "ROLLOUT_TRACE_MAX_PER_STEP",
               "JUDGE_REPLICA_COUNT", "TRAIN_N_GPUS_PER_NODE", "PRIMUS_SAVE_CHECKPOINT_DIR",
               "RUBRIC_EXPERIMENT_ID", "RUBRIC_REVISION_MODE", "TRAIN_LEARNING_RATE",
               "RUBRIC_TARGETED_EXPERIMENT_ID", "PRIMUS_JOB_UUID",
               "TRAIN_USE_KL_LOSS", "TRAIN_KL_LOSS_COEF", "RUBRIC_CRITICAL_FAILURE_GATE",
               "TRAIN_TEST_FREQ", "FINAL_EVAL_EXACT_TRIO",
               "RUBRIC_SELECTION_ORDER_V5_ENABLE", "RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE",
               "RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE", "RUBRIC_EVOLVE_COLLECT_START_STEP",
               "RUBRIC_EVOLVE_COLLECT_INTERVAL_STEPS", "RUBRIC_EVOLVE_MIN_GAP",
               "RUBRIC_EVOLVE_MIN_GOOD_PASS_RATE", "RUBRIC_EVOLVE_MAX_BAD_PASS_RATE",
               "RUBRIC_JUDGE_DROP_FAILED", "RUBRIC_EVOLVE_API_MODEL",
               "RUBRIC_EVOLVE_REWARD_MODE", "RUBRIC_EVOLVE_STD_FLOOR_SCOPE")
    save(directory / "run_manifest.json", {
        "schema_version": 1, "run_id": directory.name, "created_at": now(),
        "git_commit": git(repo, "rev-parse", "HEAD"), "git_branch": git(repo, "branch", "--show-current"),
        "git_status": git(repo, "status", "--short"), "repository": str(repo),
        "python": sys.version, "platform": platform.platform(), "packages_before_setup": versions,
        "settings": {key: os.environ[key] for key in allowed if key in os.environ},
        "artifact_root": str(root), "oss_mount_environment": os.getenv("PRIMUS_OSS_WORKSPACE_DIR"),
        "note": "Mounted-path writes; remote persistence depends on the Primus mount. No credentials captured.",
    })
    save(directory / "status.json", {"phase": "initializing", "updated_at": now()})
    return directory


def snapshot(run_dir, repo, source, train, val):
    run_dir = Path(run_dir)
    if not (run_dir / "run_manifest.json").is_file():
        raise ValueError("Run directory was not initialized")
    inputs = []
    # Store actual rewritten training inputs, not just the dataset name.
    for split, location in (("train", train), ("val", val)):
        target = run_dir / "dataset" / split
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(Path(location).glob("*.parquet")):
            dst = target / path.name
            if dst.exists() and sha(dst) != sha(path):
                raise ValueError(f"Dataset snapshot would overwrite different content: {dst}")
            if not dst.exists():
                shutil.copy2(path, dst)
            inputs.append({"split": split, "name": path.name, "sha256": sha(dst), "bytes": dst.stat().st_size})
    for filename in ("manifest.json", "rubric_changes.jsonl", "sample_index.jsonl"):
        path = Path(source) / filename
        if path.is_file():
            shutil.copy2(path, run_dir / "dataset" / filename)
    code = {}
    for relative in ("train_val.sh", "examples/ppo_trainer/qwen3B-instruct-GRPO.sh",
                     "verl/utils/reward_score/rubric_judge.py", "verl/trainer/ppo/core_algos.py",
                     "verl/trainer/ppo/ray_trainer.py", "verl/utils/training_evidence.py",
                     "verl/trainer/config/ppo_trainer.yaml", "scripts/prepare_safety_rubric_v4.py",
                     "verl/utils/validation_selection.py", "verl/utils/benign_contrast.py",
                     "verl/utils/evolution_evidence.py", "verl/utils/reward_score/rubric_judge_save_evolve_rubric.py",
                     "scripts/train_rubricevolve_evidence_v5.sh"):
        path = Path(repo) / relative
        if path.is_file():
            code[relative] = sha(path)
    save(run_dir / "input_manifest.json", {"created_at": now(), "datasets": inputs,
                                           "code_sha256": code, "packages_after_setup": package_versions()})
    save(run_dir / "status.json", {"phase": "training_ready", "updated_at": now()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("init")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--run-id")
    p.add_argument("--resume-existing", action="store_true")
    p = subs.add_parser("snapshot")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--val", type=Path, required=True)
    p = subs.add_parser("finish")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    if args.command == "init":
        print(init(args.root, args.repo, args.run_id, args.resume_existing))
    elif args.command == "snapshot":
        snapshot(args.run_dir, args.repo, args.source, args.train, args.val)
    else:
        save(args.run_dir / "status.json", {"phase": "completed" if args.exit_code == 0 else "failed",
                                           "exit_code": args.exit_code, "updated_at": now()})


if __name__ == "__main__":
    main()
