#!/usr/bin/env python3
"""Check the distributable tree and optional Git history without printing secrets."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = "AlphaAlign-Rubric-end/Evaluation/rubricalign-targeted-c-v4-3120/"
DATA_FILES = {
    DATA + split + "/" + name + ".parquet"
    for split in ("train", "val")
    for name in ("adaptive_harm", "benign_sampled", "harm_sampled", "sensitive_sampled")
}
TOKEN = re.compile(
    rb"(?<![A-Za-z0-9_-])(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
    rb"|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
)
PRIVATE_HOST = re.compile(rb"(?:[a-z0-9.-]+\.)?(?:alibaba|antfin)-inc\.com", re.I)
PERSONAL_PATH = re.compile(rb"/" + rb"(?:Users|home)/[A-Za-z0-9_.-]+/")
FORBIDDEN = {".log", ".zip", ".tar", ".gz", ".whl", ".xlsx", ".csv", ".jsonl",
             ".pt", ".pth", ".safetensors", ".pem", ".key", ".p12", ".pfx"}
IDENTITY = ("RubricAlign Maintainers", "rubricalign-maintainers@users.noreply.github.com")


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def check_bytes(name: str, raw: bytes, errors: list[str]) -> None:
    for label, pattern in (("credential-like token", TOKEN), ("private host", PRIVATE_HOST),
                           ("personal home path", PERSONAL_PATH)):
        if pattern.search(raw):
            errors.append(f"{name}: {label}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--history", action="store_true", help="Also check all reachable commits and blobs")
    scope.add_argument("--commits", metavar="REV_RANGE", help="Also check commits and blobs introduced by a push")
    args = parser.parse_args()
    has_git = (ROOT / ".git").exists()
    paths = git("ls-files").splitlines() if has_git else [
        p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
        if p.is_file() and not any(x in p.parts for x in ("__pycache__", ".pytest_cache"))
    ]
    errors: list[str] = []
    for name in paths:
        path = Path(name)
        if (path.suffix in FORBIDDEN or path.name in ("api_auth.json", "primus_api_auth.json")
                or (path.name.startswith(".env") and path.name != ".env.example")):
            errors.append(f"{name}: excluded release artifact")
        if path.suffix == ".parquet":
            if name not in DATA_FILES:
                errors.append(f"{name}: unapproved data file")
            continue
        check_bytes(name, (ROOT / path).read_bytes(), errors)
    integrity = json.loads((ROOT / "data/dataset_integrity.json").read_text())
    import hashlib
    for name, record in integrity["files"].items():
        path = ROOT / DATA / name
        if path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(f"{name}: dataset integrity mismatch")
    if hashlib.sha256((ROOT / DATA / "manifest.json").read_bytes()).hexdigest() != integrity["manifest_sha256"]:
        errors.append("dataset manifest integrity mismatch")
    if args.history or args.commits:
        revision = [args.commits] if args.commits else ["--all"]
        rows = git("log", *revision, "--format=%H%x09%an%x09%ae%x09%cn%x09%ce").splitlines()
        for row in rows:
            rev, an, ae, cn, ce = row.split("\t")
            if (an, ae) != IDENTITY or (cn, ce) != IDENTITY:
                errors.append(f"{rev[:12]}: non-release commit identity")
        objects = git("rev-list", "--objects", *revision).splitlines()
        process = subprocess.Popen(["git", "-C", str(ROOT), "cat-file", "--batch"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        assert process.stdin is not None and process.stdout is not None
        for row in objects:
            oid, _, name = row.partition(" ")
            process.stdin.write((oid + "\n").encode()); process.stdin.flush()
            header = process.stdout.readline().split()
            raw = process.stdout.read(int(header[2])); process.stdout.read(1)
            if header[1] == b"blob" and not name.endswith(".parquet"):
                check_bytes(f"history:{oid[:12]}:{name}", raw, errors)
        process.stdin.close(); process.wait()
    if errors:
        print("Release check failed (values suppressed):\n" + "\n".join(errors))
        raise SystemExit(1)
    print(f"Release check passed: {len(paths)} files; dataset hashes match.")


if __name__ == "__main__":
    main()
