#!/usr/bin/env python3
"""Canonical Arrow schema helpers for RubricAlign Parquet datasets.

The helpers in this module only normalize representation: they do not select,
drop, reorder, or rewrite samples and they do not change rubric text.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pyarrow as pa


CANONICAL_COLUMN_ORDER = (
    "prompt",
    "category",
    "instruction_prompt",
    "outcome_safety_rule",
    "outcome_useful_rule",
    "reasoning_process_rule",
    "extra_info",
    "data_source",
    "refusal_rule",
)

EXTRA_INFO_ARROW_TYPE = pa.struct(
    [
        pa.field("attack_family", pa.string()),
        pa.field("audit_decision", pa.string()),
        pa.field("audit_reasons", pa.list_(pa.string())),
        pa.field("audit_score", pa.int64()),
        pa.field("canonical_category", pa.string()),
        pa.field("critical_safety_rule_count", pa.int64()),
        pa.field("mutation_strategy", pa.string()),
        pa.field("outcome_safety_rule", pa.list_(pa.string())),
        pa.field("outcome_useful_rule", pa.list_(pa.string())),
        pa.field("prompt", pa.string()),
        pa.field("question", pa.string()),
        pa.field("raw_question", pa.string()),
        pa.field("reasoning_process_rule", pa.list_(pa.string())),
        pa.field("refusal_rule", pa.list_(pa.string())),
        pa.field("response_strategy", pa.string()),
        pa.field("reward_profile", pa.string()),
        pa.field("rubric_api_review_notes", pa.list_(pa.string())),
        pa.field("rubric_version", pa.string()),
        pa.field("sample_family", pa.string()),
        pa.field("sample_id", pa.string()),
        pa.field("source_dataset", pa.string()),
        pa.field("source_file", pa.string()),
    ]
)

_TEXT_FIELDS = {
    field.name for field in EXTRA_INFO_ARROW_TYPE if pa.types.is_string(field.type)
}
_LIST_FIELDS = {
    field.name for field in EXTRA_INFO_ARROW_TYPE if pa.types.is_list(field.type)
}
_INT_FIELDS = {
    field.name for field in EXTRA_INFO_ARROW_TYPE if pa.types.is_integer(field.type)
}


def _string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        value = [value]
    return [str(item).strip() for item in value if str(item).strip()]


def canonical_extra_info(value: Any) -> Dict[str, Any]:
    """Fill optional metadata fields with typed, semantically neutral defaults."""
    raw = dict(value) if isinstance(value, dict) else {}
    result: Dict[str, Any] = {}
    for field in EXTRA_INFO_ARROW_TYPE:
        if field.name in _TEXT_FIELDS:
            result[field.name] = str(raw.get(field.name) or "")
        elif field.name in _LIST_FIELDS:
            result[field.name] = _string_list(raw.get(field.name))
        elif field.name in _INT_FIELDS:
            candidate = raw.get(field.name)
            result[field.name] = int(candidate) if candidate is not None else 0
        else:  # pragma: no cover - guarded by the fixed schema above.
            raise TypeError(f"unsupported extra_info field type: {field}")
    return result


def normalize_extra_info_table(table: pa.Table) -> pa.Table:
    """Return ``table`` with one explicit, concat-compatible extra_info type."""
    if set(table.column_names) != set(CANONICAL_COLUMN_ORDER):
        missing = sorted(set(CANONICAL_COLUMN_ORDER) - set(table.column_names))
        extra = sorted(set(table.column_names) - set(CANONICAL_COLUMN_ORDER))
        raise ValueError(f"unexpected dataset columns: missing={missing}, extra={extra}")
    table = table.select(CANONICAL_COLUMN_ORDER)
    index = table.schema.get_field_index("extra_info")
    if index < 0:
        raise ValueError("dataset is missing required column: extra_info")
    values = [canonical_extra_info(value) for value in table["extra_info"].to_pylist()]
    array = pa.array(values, type=EXTRA_INFO_ARROW_TYPE)
    return table.set_column(index, pa.field("extra_info", EXTRA_INFO_ARROW_TYPE), array)
