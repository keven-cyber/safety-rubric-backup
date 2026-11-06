#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write concise MMLU and GSM8K scores.")
    parser.add_argument("--mmlu-results", type=Path, required=True)
    parser.add_argument("--gsm8k-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def newest_result(path: Path) -> Path:
    candidates = sorted(path.rglob("*.json"), key=lambda item: item.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No lm-eval JSON result found under {path}")
    return candidates[-1]


def load_result(path: Path) -> dict[str, Any]:
    result_path = newest_result(path)
    with result_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid lm-eval result object: {result_path}")
    return payload


def metric_value(bucket: dict[str, Any], preferred_keys: list[str]) -> float:
    for key in preferred_keys:
        value = bucket.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    raise ValueError(
        f"None of the expected metrics {preferred_keys} were present; available={sorted(bucket)}"
    )


def mmlu_score(payload: dict[str, Any]) -> float:
    for container_name in ("groups", "results"):
        container = payload.get(container_name, {})
        if isinstance(container, dict) and isinstance(container.get("mmlu_modelscope"), dict):
            return metric_value(container["mmlu_modelscope"], ["acc,none", "acc"])
    raise ValueError("MMLU aggregate result 'mmlu_modelscope' was not found")


def gsm8k_score(payload: dict[str, Any]) -> float:
    results = payload.get("results", {})
    task_result = results.get("gsm8k_cot_modelscope") if isinstance(results, dict) else None
    if not isinstance(task_result, dict):
        raise ValueError("GSM8K result 'gsm8k_cot_modelscope' was not found")
    return metric_value(
        task_result,
        [
            "exact_match,flexible-extract",
            "exact_match,strict-match",
            "exact_match,boxed-extract",
            "exact_match,none",
            "exact_match",
        ],
    )


def main() -> None:
    args = parse_args()
    mmlu = mmlu_score(load_result(args.mmlu_results))
    gsm8k = gsm8k_score(load_result(args.gsm8k_results))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        f"MMLU: {mmlu * 100:.2f}%\nGSM8K: {gsm8k * 100:.2f}%\n",
        encoding="utf-8",
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
