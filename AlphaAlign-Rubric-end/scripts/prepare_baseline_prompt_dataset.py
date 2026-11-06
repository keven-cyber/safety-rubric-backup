#!/usr/bin/env python3
"""Convert the RubricAlign training dataset into the forms the AlphaAlign and
NSPO baselines can actually be trained on.

RubricAlign ships its prompts with the ``<think>``/``<answer>`` instruction
template already embedded in the ``prompt`` column. That template is part of the
RubricAlign method, so the baselines are trained on the same sample set but with
the plain user request recovered from ``extra_info.raw_question``.

Two output variants exist because the two baselines have different contracts:

``plain`` (NSPO)
    The prompt is the bare user request. NSPO's reward is a Qwen3Guard verdict on
    the response alone, so it imposes no output format.

``alphaalign`` (AlphaAlign)
    AlphaAlign's reward function is not format-agnostic: it extracts
    ``\\boxed{}`` out of ``<answer>`` tags, requires ``<safety_reasoning>`` tags
    for its format term, and branches on whether ``reward_model.ground_truth``
    contains ``unsafe``. Training it on a bare prompt yields a constant-zero
    reward, so this variant appends AlphaAlign's own instruction text (taken
    verbatim from examples/data_preprocess/xstest.py, the variant that matches
    the ``<answer>`` regexes in verl/utils/reward_score/curcuit.py) and fills
    ground_truth with ``safe::``/``unsafe::`` labels derived from the v3 reward
    profile.

The output keeps the source directory layout (``train/`` and ``val/`` with the
same four Parquet file names) so that the baseline launchers can reuse the same
file-list construction as ``examples/ppo_trainer/qwen3B-instruct-GRPO.sh``.

Source Parquet files are never modified.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scripts.rubric_dataset_schema import EXTRA_INFO_ARROW_TYPE, canonical_extra_info
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from rubric_dataset_schema import EXTRA_INFO_ARROW_TYPE, canonical_extra_info

SPLITS = ("train", "val")
DATASET_FILES = (
    "benign_sampled.parquet",
    "sensitive_sampled.parquet",
    "harm_sampled.parquet",
    "adaptive_harm.parquet",
)
# Rubric rules stay available through extra_info; the top-level duplicates are
# dropped so the baseline Parquet files only carry what the baselines consume.
RUBRIC_COLUMNS = (
    "outcome_safety_rule",
    "outcome_useful_rule",
    "reasoning_process_rule",
    "refusal_rule",
    "instruction_prompt",
)
PROMPT_SOURCES = ("raw_question", "instruction")
VARIANTS = ("plain", "alphaalign")

# Keep the source audit metadata, then append the fields introduced by this
# disposable baseline conversion.  Pandas/Arrow inference is deliberately not
# used for this nested column: files whose optional lists are all empty would
# otherwise become ``list<null>``, while adaptive files can acquire additional
# struct fields.  Hugging Face requires the nested struct to match exactly when
# it concatenates the four profile files.
BASELINE_EXTRA_INFO_ARROW_TYPE = pa.struct(
    [
        *list(EXTRA_INFO_ARROW_TYPE),
        pa.field("index", pa.int64()),
        pa.field("split", pa.string()),
        pa.field("baseline_prompt_source", pa.string()),
        pa.field("baseline_variant", pa.string()),
    ]
)

# Verbatim from examples/data_preprocess/xstest.py. Do not reword: the reward
# regexes in AlphaAlign's curcuit.py depend on the tags this text asks for.
ALPHAALIGN_INSTRUCTION = (
    r'You FIRST carefully evaluate the safety implications of this question through reasoning. '
    r'Your safety reasoning MUST BE enclosed within <safety_reasoning> </safety_reasoning> tags. '
    r"Your answer MUST BE enclosed within <answer> </answer> tags. "
    r"If the question is deemed unsafe, provide a harmless explanation for the refusal, and refuse like \boxed{Sorry, I can't comply} within <answer> </answer> tags, which will be extracted automatically by the \boxed{} tag. "
    r"If the question is safe, then the content within <answer> </answer> tags will be shown to users."
)

# AlphaAlign's safe_acc_reward branches on "unsafe" appearing in ground_truth.
# The v3 reward profiles map onto that binary label directly.
PROFILE_SAFETY_LABEL = {
    "benign": "safe",
    "sensitive_benign": "safe",
    "ordinary_harm": "unsafe",
    "adaptive_harm": "unsafe",
}


def _extract_instruction_prompt(row) -> str:
    """Return the RubricAlign prompt text exactly as the source dataset has it."""
    chat = row["prompt"]
    if chat is None or len(chat) == 0:
        raise ValueError("empty prompt column")
    return str(chat[0]["content"])


def _extract_raw_question(row) -> str:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError("extra_info is missing or is not a mapping")
    for key in ("raw_question", "question", "prompt"):
        value = extra_info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("extra_info carries no non-empty raw_question/question/prompt")


def convert_frame(df: pd.DataFrame, prompt_source: str, variant: str, split: str, file_name: str) -> pd.DataFrame:
    if "prompt" not in df.columns:
        raise ValueError(f"{split}/{file_name} has no prompt column")
    if "data_source" not in df.columns:
        raise ValueError(f"{split}/{file_name} has no data_source column")

    extract = _extract_raw_question if prompt_source == "raw_question" else _extract_instruction_prompt

    prompts = []
    extra_infos = []
    ground_truths = []
    for position, (_, row) in enumerate(df.iterrows()):
        try:
            content = extract(row)
        except ValueError as exc:
            raise ValueError(f"{split}/{file_name} row {position}: {exc}") from exc

        data_source = str(row["data_source"])
        if variant == "alphaalign":
            label = PROFILE_SAFETY_LABEL.get(data_source)
            if label is None:
                raise ValueError(
                    f"{split}/{file_name} row {position}: data_source '{data_source}' has no safe/unsafe mapping. "
                    f"Known profiles: {sorted(PROFILE_SAFETY_LABEL)}"
                )
            content = f"{ALPHAALIGN_INSTRUCTION} {content}"
            category = str(row["category"]) if "category" in df.columns else "unknown"
            ground_truths.append(f"{label}::{category}")
        else:
            ground_truths.append("")

        # NSPO's plain loader expects the bare request as a string. AlphaAlign
        # uses verl's native chat-message contract and therefore keeps a list.
        if variant == "plain":
            prompts.append(content)
        else:
            prompts.append([{"role": "user", "content": content}])

        extra_info = canonical_extra_info(row.get("extra_info"))
        extra_info["index"] = position
        extra_info["split"] = split
        extra_info["baseline_prompt_source"] = prompt_source
        extra_info["baseline_variant"] = variant
        extra_infos.append(extra_info)

    if variant == "plain" and not all(isinstance(prompt, str) and prompt.strip() for prompt in prompts):
        raise ValueError(f"{split}/{file_name}: plain NSPO prompts must all be non-empty strings")
    if variant == "alphaalign" and not all(
        isinstance(prompt, list)
        and prompt
        and all(isinstance(message, dict) and isinstance(message.get("content"), str) for message in prompt)
        for prompt in prompts
    ):
        raise ValueError(f"{split}/{file_name}: AlphaAlign prompts must be chat-message lists with string content")

    out = df.drop(columns=[c for c in RUBRIC_COLUMNS if c in df.columns]).copy()
    out["prompt"] = prompts
    out["extra_info"] = extra_infos
    out["ability"] = "safety"
    # verl reads non_tensor_batch["reward_model"]["ground_truth"] unconditionally in
    # both the 0.2 naive manager and the 0.5 batch manager, and the RubricAlign
    # dataset does not carry the column.
    out["reward_model"] = [{"style": "rule", "ground_truth": gt} for gt in ground_truths]
    return out


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    """Write one baseline shard with an explicit nested metadata schema."""
    if "extra_info" not in frame.columns:
        raise ValueError(f"baseline frame is missing extra_info: {path}")

    extra_info_position = list(frame.columns).index("extra_info")
    table = pa.Table.from_pandas(frame.drop(columns=["extra_info"]), preserve_index=False)
    extra_info = pa.array(frame["extra_info"].tolist(), type=BASELINE_EXTRA_INFO_ARROW_TYPE)
    table = table.add_column(
        extra_info_position,
        pa.field("extra_info", BASELINE_EXTRA_INFO_ARROW_TYPE),
        extra_info,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".parquet.tmp")
    pq.write_table(table, temporary_path, compression="snappy")
    temporary_path.replace(path)


def validate_runtime_dataset(split_dir: Path) -> int:
    """Exercise the exact multi-file merge used by AlphaAlign's RL loader."""
    parquet_files = [split_dir / file_name for file_name in DATASET_FILES]
    missing = [path for path in parquet_files if not path.is_file()]
    if missing:
        raise ValueError(f"prepared baseline split is incomplete: {missing}")

    reference_schema = pq.read_schema(parquet_files[0])
    extra_info_index = reference_schema.get_field_index("extra_info")
    if extra_info_index < 0:
        raise ValueError(f"prepared baseline is missing extra_info: {parquet_files[0]}")
    actual_extra_info_type = reference_schema.field(extra_info_index).type
    if not actual_extra_info_type.equals(BASELINE_EXTRA_INFO_ARROW_TYPE):
        raise TypeError(
            "prepared baseline extra_info has the wrong Arrow type: "
            f"file={parquet_files[0]}, actual={actual_extra_info_type}, "
            f"expected={BASELINE_EXTRA_INFO_ARROW_TYPE}"
        )

    for path in parquet_files[1:]:
        schema = pq.read_schema(path)
        if not schema.equals(reference_schema, check_metadata=False):
            raise TypeError(
                "prepared baseline Parquet schemas are not concat-compatible: "
                f"reference={parquet_files[0]}, mismatched={path}, "
                f"reference_schema={reference_schema}, mismatched_schema={schema}"
            )

    # A fresh cache prevents an older conversion at the same path from hiding
    # a regression in Hugging Face feature inference.
    from datasets import load_dataset, concatenate_datasets

    with tempfile.TemporaryDirectory(prefix="baseline-hf-preflight-") as cache_dir:
        shards = [
            load_dataset("parquet", data_files=str(path), split="train", cache_dir=cache_dir)
            for path in parquet_files
        ]
        combined = concatenate_datasets(shards)

    expected_rows = sum(pq.read_metadata(path).num_rows for path in parquet_files)
    if combined.num_rows != expected_rows:
        raise ValueError(
            "prepared baseline row count changed during Hugging Face merge: "
            f"expected={expected_rows}, actual={combined.num_rows}"
        )
    print(
        f"[PREFLIGHT] Baseline runtime merge passed: "
        f"{len(parquet_files)} files, {expected_rows} rows, directory={split_dir}"
    )
    return expected_rows


def prepare_dataset(source_dir: Path, output_dir: Path, prompt_source: str, variant: str) -> dict:
    if prompt_source not in PROMPT_SOURCES:
        raise ValueError(f"unsupported prompt source: {prompt_source}")
    if variant not in VARIANTS:
        raise ValueError(f"unsupported baseline variant: {variant}")
    if variant == "alphaalign" and prompt_source != "raw_question":
        raise ValueError(
            "--variant alphaalign replaces the instruction template with AlphaAlign's own, "
            "so it requires --prompt-source raw_question."
        )
    if source_dir.resolve() == output_dir.resolve():
        raise ValueError("output directory must differ from the source dataset directory")

    manifest = {
        "source_dir": str(source_dir),
        "prompt_source": prompt_source,
        "variant": variant,
        "extra_info_schema": "baseline_v1",
        "splits": {},
    }

    for split in SPLITS:
        split_in = source_dir / split
        split_out = output_dir / split
        if not split_in.is_dir():
            raise ValueError(f"missing split directory: {split_in}")
        split_out.mkdir(parents=True, exist_ok=True)

        for file_name in DATASET_FILES:
            path_in = split_in / file_name
            if not path_in.is_file():
                raise ValueError(f"missing dataset file: {path_in}")
            frame = pd.read_parquet(path_in)
            converted = convert_frame(frame, prompt_source, variant, split, file_name)
            path_out = split_out / file_name
            _write_frame(converted, path_out)
            print(f"[INFO] {split}/{file_name}: {len(converted)} rows -> {path_out}")

        manifest["splits"][split] = validate_runtime_dataset(split_out)

    manifest_path = output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    with open(temporary_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    temporary_manifest.replace(manifest_path)
    print(f"[INFO] Wrote {manifest_path}")
    print(
        f"[INFO] Baseline dataset ready ({variant}): "
        f"train={manifest['splits']['train']}, val={manifest['splits']['val']}"
    )
    return manifest


def main() -> None:
    default_source = Path(__file__).resolve().parents[1] / "Evaluation" / "updated_training_rubric_dataset_independent_v3"
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", default=str(default_source), help="RubricAlign dataset directory containing train/ and val/")
    parser.add_argument("--output-dir", required=True, help="Destination directory for the baseline dataset")
    parser.add_argument(
        "--prompt-source",
        default="raw_question",
        choices=PROMPT_SOURCES,
        help="raw_question uses the plain user request; instruction keeps the RubricAlign template",
    )
    parser.add_argument(
        "--variant",
        default="plain",
        choices=VARIANTS,
        help="plain for NSPO; alphaalign adds AlphaAlign's instruction text and safe/unsafe ground truth",
    )
    args = parser.parse_args()

    try:
        prepare_dataset(Path(args.source_dir), Path(args.output_dir), args.prompt_source, args.variant)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"[ERROR] {exc}") from exc


if __name__ == "__main__":
    main()
