#!/usr/bin/env python3
"""Build a temporary rubric dataset with a selected training prompt template."""

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from scripts.rubric_dataset_schema import (
        EXTRA_INFO_ARROW_TYPE,
        normalize_extra_info_table,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from rubric_dataset_schema import EXTRA_INFO_ARROW_TYPE, normalize_extra_info_table


USER_REQUEST_MARKER = "\n\nUser request:\n"

PROMPT_PREFIXES = {
    "legacy_compact": (
        "Execute the following steps in order within <think></think>: 1. Analyze user intent and assess "
        "risk level (safe/partially unsafe/unsafe); 2. Identify key risks and form a brief answer summary; "
        "3. Based on the answer summary, plan the response strategy: maximize helpfulness within safety "
        "limits,    providing a safer, high-level, lawful response that addresses the closest legitimate "
        "need. Then, provide the final answer in <answer></answer>. Output must be exactly in this format "
        "and contain no extra text outside tags: <think>...</think><answer>...</answer>."
    ),
    "structured_steps": (
        "You are a helpful and safe AI assistant. Follow these steps to respond:\n\n"
        "Step 1 (Thinking): Analyze user intent, assess risk level (safe/partially unsafe/unsafe), identify "
        "key risks, and plan your response strategy \u2014 maximize helpfulness within safety limits.\n\n"
        "Step 2 (Answering): Provide a safe, high-level, lawful response that addresses the closest "
        "legitimate need.\n\n"
        "You must output your response in exactly the following format, with no extra text outside the "
        "tags:\n\n"
        "<think>\n[Your analysis here]\n</think>\n<answer>\n[Your response here]\n</answer>"
    ),
}


def extract_user_request(instruction_prompt: str) -> str:
    if USER_REQUEST_MARKER not in instruction_prompt:
        raise ValueError("instruction_prompt does not contain the expected 'User request:' marker")
    return instruction_prompt.split(USER_REQUEST_MARKER, 1)[1]


def render_prompt(version: str, user_request: str) -> str:
    try:
        prefix = PROMPT_PREFIXES[version]
    except KeyError as exc:
        choices = ", ".join(PROMPT_PREFIXES)
        raise ValueError(f"unsupported prompt version {version!r}; expected one of: {choices}") from exc
    return f"{prefix}{USER_REQUEST_MARKER}{user_request}"


def rewrite_messages(messages: Any, rendered_prompt: str) -> List[Dict[str, Any]]:
    if not isinstance(messages, list):
        raise ValueError("prompt must be a list of chat messages")

    rewritten = []
    replaced = False
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each prompt message must be a mapping")
        item = dict(message)
        if not replaced and item.get("role") == "user":
            item["content"] = rendered_prompt
            replaced = True
        rewritten.append(item)

    if not replaced:
        raise ValueError("prompt does not contain a user message")
    return rewritten


def rewrite_table(table: pa.Table, version: str) -> pa.Table:
    required = {"instruction_prompt", "prompt"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"dataset is missing required columns: {', '.join(sorted(missing))}")

    instructions = table["instruction_prompt"].to_pylist()
    messages = table["prompt"].to_pylist()
    rendered = [render_prompt(version, extract_user_request(value)) for value in instructions]
    rewritten_messages = [rewrite_messages(value, prompt) for value, prompt in zip(messages, rendered)]

    instruction_index = table.schema.get_field_index("instruction_prompt")
    prompt_index = table.schema.get_field_index("prompt")
    table = table.set_column(
        instruction_index,
        "instruction_prompt",
        pa.array(rendered, type=table.schema.field(instruction_index).type),
    )
    return table.set_column(
        prompt_index,
        "prompt",
        pa.array(rewritten_messages, type=table.schema.field(prompt_index).type),
    )


def validate_runtime_dataset(output_dir: Path) -> None:
    """Run the same multi-file Arrow merge that the RL dataset loader uses.

    This intentionally validates the files after they have been written and
    reopened.  Checking only the in-memory tables can miss Parquet/Hugging Face
    feature inference differences such as ``list<null>`` for an all-empty
    nested list.
    """
    parquet_files = sorted(output_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no prepared parquet files found in {output_dir}")

    expected_schema = pq.read_schema(parquet_files[0])
    extra_info_index = expected_schema.get_field_index("extra_info")
    if extra_info_index < 0:
        raise ValueError(f"prepared dataset is missing extra_info: {parquet_files[0]}")
    actual_extra_info_type = expected_schema.field(extra_info_index).type
    if not actual_extra_info_type.equals(EXTRA_INFO_ARROW_TYPE):
        raise TypeError(
            "prepared extra_info does not use the canonical Arrow type: "
            f"file={parquet_files[0]}, actual={actual_extra_info_type}, "
            f"expected={EXTRA_INFO_ARROW_TYPE}"
        )

    for path in parquet_files[1:]:
        schema = pq.read_schema(path)
        if not schema.equals(expected_schema, check_metadata=False):
            raise TypeError(
                "prepared Parquet schemas are not concat-compatible: "
                f"reference={parquet_files[0]}, mismatched={path}, "
                f"reference_schema={expected_schema}, mismatched_schema={schema}"
            )

    # Import lazily so schema-only utilities do not acquire the datasets
    # dependency.  A fresh cache also prevents an older file at the same cache
    # path from hiding a schema regression.
    from datasets import Dataset, concatenate_datasets

    with tempfile.TemporaryDirectory(prefix="rubricalign-hf-preflight-") as cache_dir:
        datasets = [
            Dataset.from_parquet(str(path), cache_dir=cache_dir)
            for path in parquet_files
        ]
        combined = concatenate_datasets(datasets)

    expected_rows = sum(pq.read_metadata(path).num_rows for path in parquet_files)
    if combined.num_rows != expected_rows:
        raise ValueError(
            "prepared dataset row count changed during Hugging Face merge: "
            f"expected={expected_rows}, actual={combined.num_rows}"
        )
    print(
        f"[PREFLIGHT] Runtime dataset merge passed: "
        f"{len(parquet_files)} files, {expected_rows} rows, directory={output_dir}"
    )


def prepare_dataset(input_dir: Path, output_dir: Path, version: str) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("output directory must differ from the source dataset directory")

    parquet_files = sorted(input_dir.glob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"no parquet files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    for source_path in parquet_files:
        table = rewrite_table(pq.read_table(source_path), version)
        # Source validation files intentionally remain byte-identical for the
        # audit. Normalize only this disposable training cache so Hugging Face
        # can concatenate source and generated shards with one nested schema.
        table = normalize_extra_info_table(table)
        output_path = output_dir / source_path.name
        temporary_path = output_path.with_suffix(".parquet.tmp")
        pq.write_table(table, temporary_path, compression="snappy")
        temporary_path.replace(output_path)
        total_rows += table.num_rows

    validate_runtime_dataset(output_dir)

    print(
        f"[INFO] Prepared TRAIN_PROMPT_VERSION={version}: "
        f"{len(parquet_files)} files, {total_rows} rows -> {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", choices=tuple(PROMPT_PREFIXES), required=True)
    args = parser.parse_args()
    prepare_dataset(args.input_dir, args.output_dir, args.version)


if __name__ == "__main__":
    main()
