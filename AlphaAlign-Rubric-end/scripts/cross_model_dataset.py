"""Resolve and seal the exact shared corpus without modifying source files."""
import json
import os
from pathlib import Path

expected_version = "updated_training_rubric_dataset_independent_v4_targeted_hard_negative"
expected_train = 3120
expected_val = 304
dataset_name = "rubricalign-targeted-c-v4-3120"

roots = []
for value in (
    os.environ.get("FILE_PATH"),
    os.environ.get("PRIMUS_TRAIN_DATASET_DIR"),
    os.environ.get("PRIMUS_DATASETS_DIR"),
    str(Path(str(Path(__file__).resolve().parents[1])) / "Evaluation" / dataset_name),
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
            import argparse, hashlib
            import pyarrow.parquet as pq
            parser = argparse.ArgumentParser()
            parser.add_argument('--report', required=True)
            args = parser.parse_args()
            records = {}
            counts = {}
            for split, expected in [('train', expected_train), ('val', expected_val)]:
                counts[split] = 0
                for path in [p for p in required if p.parent.name == split]:
                    rows = pq.ParquetFile(path).metadata.num_rows
                    counts[split] += rows
                    records[str(path.relative_to(candidate))] = dict(rows=rows, bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                if counts[split] != expected:
                    raise ValueError('Actual parquet rows differ from shared budget')
            with Path(args.report).open('x') as stream:
                json.dump(dict(dataset=str(candidate.resolve()), version=expected_version, counts=counts,
                               manifest_sha256=hashlib.sha256((candidate/'manifest.json').read_bytes()).hexdigest(),
                               files=records), stream, indent=2)
                stream.flush(); os.fsync(stream.fileno())
            print(candidate.resolve())
            raise SystemExit(0)

searched = ", ".join(str(path) for path in roots) or "<none>"
raise SystemExit(
    "[ERROR] Exact RubricAlign targeted-C dataset was not mounted. "
    f"Expected version={expected_version}, train={expected_train}, val={expected_val}; "
    f"searched: {searched}"
)
