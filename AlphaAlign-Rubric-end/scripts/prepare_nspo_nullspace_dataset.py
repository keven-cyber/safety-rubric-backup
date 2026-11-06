#!/usr/bin/env python3
"""Build the general-capability corpus that NSPO's null-space projection needs.

NSPO estimates the null space of the MLP input activations on a general
instruction corpus, then keeps every safety update inside that null space. The
released code expects a directory that ``datasets.load_dataset`` can open with a
``train`` split and a plain-string ``prompt`` column.

The corpus is downloaded from a Hugging Face mirror. If the download or the column
detection fails, the script can fall back to the benign slice of the RubricAlign
dataset, which keeps the pipeline runnable offline at the cost of a narrower
activation estimate. The fallback is always reported.

Output layout::

    <output-dir>/train/nullspace.parquet   # single column: prompt (string)
    <output-dir>/manifest.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd

# Ordered by preference. Alpaca-style corpora expose instruction (+ optional input).
TEXT_COLUMN_CANDIDATES = ("instruction", "prompt", "question", "query", "text", "content", "input")
DATA_SUFFIXES = (".parquet", ".jsonl", ".json", ".csv", ".tsv")
DEFAULT_DATASET_ID = "tatsu-lab/alpaca"


def download_dataset(dataset_id: str, raw_dir: Path) -> bool:
    # Primus OSS mounts commonly reject symlink/hardlink creation.  The
    # Hugging Face ``local_dir`` downloader may create links while materializing
    # its cache, so stage the download on the node-local filesystem and copy
    # regular files into OSS afterwards.
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get("HF_MIRROR_ENDPOINT", "https://hf-mirror.com")
    print(f"[INFO] Downloading null-space corpus from {endpoint}: {dataset_id}")
    staging_root = os.environ.get("NSPO_NULLSPACE_STAGING_ROOT", "")
    if staging_root:
        Path(staging_root).mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix="nspo_nullspace_hf_",
            dir=staging_root or None,
        )
    )
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=os.environ.get("NSPO_NULLSPACE_DATASET_REVISION"),
            local_dir=str(staging_dir),
            endpoint=endpoint,
            max_workers=8,
        )
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
        # copytree follows links in the node-local staging directory and writes
        # ordinary files to the OSS mount, avoiding Errno 95 there.
        shutil.copytree(staging_dir, raw_dir, symlinks=False)
    except Exception as exc:  # noqa: BLE001 - caller decides whether fallback is allowed
        print(f"[WARN] Hugging Face mirror download failed for {dataset_id}: {exc}")
        return False
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    return True


def read_any(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if path.suffix == ".json":
            return pd.read_json(path)
        if path.suffix in (".csv", ".tsv"):
            return pd.read_csv(path, sep="\t" if path.suffix == ".tsv" else ",")
    except Exception as exc:  # noqa: BLE001 - any malformed shard is simply skipped
        print(f"[WARN] Could not read {path}: {exc}")
    return None


def extract_prompts(raw_dir: Path) -> tuple[list[str], str]:
    """Return (prompts, description of what was used)."""
    files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix in DATA_SUFFIXES)
    if not files:
        raise ValueError(f"no readable data files under {raw_dir}")

    for path in files:
        df = read_any(path)
        if df is None or df.empty:
            continue
        column = next((c for c in TEXT_COLUMN_CANDIDATES if c in df.columns), None)
        if column is None:
            print(f"[WARN] {path.name} has columns {list(df.columns)}; no usable text column.")
            continue

        series = df[column].astype(str)
        # Alpaca-style corpora keep the optional context in a separate column.
        if column == "instruction" and "input" in df.columns:
            extra = df["input"].fillna("").astype(str)
            series = [
                f"{i.strip()}\n\n{e.strip()}" if e.strip() else i.strip()
                for i, e in zip(series, extra)
            ]
            series = pd.Series(series)

        prompts = [p.strip() for p in series.tolist() if isinstance(p, str) and p.strip()]
        if prompts:
            return prompts, f"{path.relative_to(raw_dir)}::{column}"

    raise ValueError(f"no usable text column found under {raw_dir}")


def fallback_prompts(rubric_dataset_dir: Path) -> tuple[list[str], str]:
    path = rubric_dataset_dir / "train" / "benign_sampled.parquet"
    if not path.is_file():
        raise SystemExit(f"[ERROR] Fallback corpus not found: {path}")
    df = pd.read_parquet(path)
    prompts = []
    for _, row in df.iterrows():
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            value = extra_info.get("raw_question") or extra_info.get("question")
            if isinstance(value, str) and value.strip():
                prompts.append(value.strip())
    if not prompts:
        raise SystemExit(f"[ERROR] Fallback corpus {path} yielded no prompts")
    return prompts, f"{path.name}::extra_info.raw_question"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Hugging Face dataset id")
    parser.add_argument("--raw-dir", default="", help="Download cache directory (default: <output-dir>/_raw)")
    parser.add_argument("--output-dir", required=True, help="Destination directory consumed by datasets.load_dataset")
    parser.add_argument("--max-rows", type=int, default=4096, help="Cap on the number of prompts kept")
    parser.add_argument("--skip-download", action="store_true", help="Reuse an already populated --raw-dir")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Fall back to the RubricAlign benign slice when the download or parsing fails",
    )
    parser.add_argument(
        "--fallback-dataset-dir",
        default=str(project_root / "Evaluation" / "updated_training_rubric_dataset_independent_v3"),
        help="RubricAlign dataset directory used by the fallback",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    raw_dir = Path(args.raw_dir) if args.raw_dir else output_dir / "_raw"

    downloaded = True
    if not args.skip_download:
        downloaded = download_dataset(args.dataset_id, raw_dir)

    source_kind = "hf_mirror"
    try:
        if not downloaded:
            raise ValueError("download failed")
        prompts, source_detail = extract_prompts(raw_dir)
    except ValueError as exc:
        if not args.allow_fallback:
            raise SystemExit(
                f"[ERROR] Could not build the null-space corpus from {args.dataset_id}: {exc}\n"
                f"[ERROR] Point --dataset-id at a reachable Hugging Face dataset, or pass --allow-fallback "
                f"to use the RubricAlign benign slice instead."
            ) from exc
        print(f"[WARN] {exc}")
        print("[WARN] Falling back to the RubricAlign benign slice. The null space will be estimated from")
        print("[WARN] safety-domain prompts only, which is narrower than the corpus NSPO used.")
        prompts, source_detail = fallback_prompts(Path(args.fallback_dataset_dir))
        source_kind = "rubricalign_benign_fallback"

    # Deterministic de-duplication that preserves order.
    seen = set()
    unique_prompts = []
    for prompt in prompts:
        if prompt in seen:
            continue
        seen.add(prompt)
        unique_prompts.append(prompt)
        if len(unique_prompts) >= args.max_rows:
            break

    train_dir = output_dir / "train"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True, exist_ok=True)
    out_path = train_dir / "nullspace.parquet"
    pd.DataFrame({"prompt": unique_prompts}).to_parquet(out_path, index=False)
    print(f"[INFO] Wrote {len(unique_prompts)} prompts -> {out_path}")

    manifest = {
        "source_kind": source_kind,
        "dataset_id": args.dataset_id,
        "dataset_revision": os.environ.get("NSPO_NULLSPACE_DATASET_REVISION"),
        "source_detail": source_detail,
        "rows": len(unique_prompts),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    # The training worker opens this directory with datasets.load_dataset, so the
    # layout is verified here rather than failing inside the FSDP worker.
    try:
        from datasets import load_dataset

        checked = load_dataset(str(output_dir))["train"]
    except ImportError:
        print("[WARN] datasets is not installed here; skipping the load_dataset verification.")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"[ERROR] datasets.load_dataset('{output_dir}') failed: {exc}") from exc
    else:
        if "prompt" not in checked.column_names:
            raise SystemExit(f"[ERROR] Verification failed: columns are {checked.column_names}")
        print(f"[INFO] Verified: load_dataset('{output_dir}')['train'] has {len(checked)} rows.")

    if source_kind != "hf_mirror":
        print("[WARN] Null-space corpus is the fallback, not the Hugging Face general instruction set.", file=sys.stderr)


if __name__ == "__main__":
    main()
