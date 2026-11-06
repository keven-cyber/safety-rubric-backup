import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.prepare_baseline_prompt_dataset import (
    ALPHAALIGN_INSTRUCTION,
    BASELINE_EXTRA_INFO_ARROW_TYPE,
    DATASET_FILES,
    prepare_dataset,
)


PROFILE_BY_FILE = {
    "benign_sampled.parquet": "benign",
    "sensitive_sampled.parquet": "sensitive_benign",
    "harm_sampled.parquet": "ordinary_harm",
    "adaptive_harm.parquet": "adaptive_harm",
}


def _source_row(profile: str, split: str) -> dict:
    request = f"{split} request for {profile}"
    safety_rule = "Do not provide harmful assistance."
    info = {
        "raw_question": request,
        "question": request,
        "prompt": request,
        "reward_profile": profile,
        "sample_id": f"{split}-{profile}",
        "outcome_safety_rule": [safety_rule],
        "outcome_useful_rule": ["Remain useful within the safe boundary."],
        "reasoning_process_rule": [],
        "refusal_rule": ["Refuse only when required."],
    }
    # Reproduce the production C/D1 mismatch: only adaptive shards carry these
    # optional fields, and an all-empty list must still remain list<string>.
    if profile == "adaptive_harm":
        info["attack_family"] = "role_play"
        info["rubric_api_review_notes"] = []

    return {
        "prompt": [{"role": "user", "content": f"source template: {request}"}],
        "category": "test",
        "instruction_prompt": f"source template: {request}",
        "outcome_safety_rule": [safety_rule],
        "outcome_useful_rule": ["Remain useful within the safe boundary."],
        "reasoning_process_rule": [],
        "refusal_rule": ["Refuse only when required."],
        "extra_info": info,
        "data_source": profile,
    }


def _source_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    for split in ("train", "val"):
        for file_name, profile in PROFILE_BY_FILE.items():
            path = root / split / file_name
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([_source_row(profile, split)]).to_parquet(path, index=False)
    return root


@pytest.mark.parametrize("variant", ["alphaalign", "plain"])
def test_prepared_baseline_has_one_runtime_concat_schema(tmp_path, variant):
    source = _source_dataset(tmp_path)
    output = tmp_path / variant

    manifest = prepare_dataset(source, output, "raw_question", variant)

    assert manifest["splits"] == {"train": 4, "val": 4}
    assert manifest["extra_info_schema"] == "baseline_v1"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == manifest

    for split in ("train", "val"):
        schemas = [pq.read_schema(output / split / name) for name in DATASET_FILES]
        assert all(schema.equals(schemas[0], check_metadata=False) for schema in schemas[1:])
        extra_info = schemas[0].field(schemas[0].get_field_index("extra_info"))
        assert extra_info.type.equals(BASELINE_EXTRA_INFO_ARROW_TYPE)

        benign = pd.read_parquet(output / split / "benign_sampled.parquet").iloc[0]
        adaptive = pd.read_parquet(output / split / "adaptive_harm.parquet").iloc[0]
        assert benign["extra_info"]["attack_family"] == ""
        assert list(benign["extra_info"]["rubric_api_review_notes"]) == []
        assert adaptive["extra_info"]["attack_family"] == "role_play"
        assert list(adaptive["extra_info"]["rubric_api_review_notes"]) == []
        assert benign["extra_info"]["baseline_variant"] == variant

        if variant == "alphaalign":
            assert benign["prompt"][0]["content"].startswith(ALPHAALIGN_INSTRUCTION)
            assert benign["reward_model"]["ground_truth"] == "safe::test"
            assert adaptive["reward_model"]["ground_truth"] == "unsafe::test"
        else:
            assert benign["prompt"] == f"{split} request for benign"
