import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parents[2]
DATASET = ROOT / "Evaluation/updated_training_rubric_dataset_independent_v3"
PROFILES = {
    "benign": "benign_sampled.parquet",
    "sensitive_benign": "sensitive_sampled.parquet",
    "ordinary_harm": "harm_sampled.parquet",
    "adaptive_harm": "adaptive_harm.parquet",
}


def _manifest():
    return json.loads((DATASET / "manifest.json").read_text())


def test_v3_uses_all_2400_rows_for_training():
    manifest = _manifest()
    assert manifest["training_rows"] == 2400
    assert manifest["training_distribution"] == {
        "benign": 600,
        "sensitive_benign": 600,
        "ordinary_harm": 480,
        "adaptive_harm": 720,
    }
    assert sum(len(pd.read_parquet(DATASET / "train" / name)) for name in PROFILES.values()) == 2400


def test_v3_validation_is_balanced_and_independent():
    manifest = _manifest()
    assert manifest["validation_rows"] == 256
    assert manifest["validation_distribution"] == {profile: 64 for profile in PROFILES}
    assert manifest["leakage_audit"] == {
        "normalized_prompt_overlap": 0,
        "sample_family_overlap": 0,
        "near_duplicate_threshold": 0.92,
        "near_duplicate_pairs": 0,
    }


def test_v3_rubrics_are_bounded_and_metadata_aligned():
    manifest = _manifest()
    for split in ("training", "validation"):
        audit = manifest["rubric_audit"][split]
        assert audit["format_or_cot_conflicts"] == 0
        assert audit["safety_usefulness_exact_overlaps"] == 0
        assert audit["empty_rules"] == 0
        assert audit["rows_missing_required_rubric_categories"] == 0
        assert audit["metadata_or_rubric_mismatches"] == 0
    for parquet_dir in (DATASET / "train", DATASET / "val"):
        for path in parquet_dir.glob("*.parquet"):
            frame = pd.read_parquet(path)
            assert frame["outcome_safety_rule"].map(len).max() <= 7
            assert frame["outcome_useful_rule"].map(len).max() <= 7
            assert frame["reasoning_process_rule"].map(len).max() <= 4
            assert frame["refusal_rule"].map(len).max() <= 2


def test_v3_does_not_use_final_benchmarks_for_validation():
    manifest = _manifest()
    assert set(manifest["excluded_final_benchmarks"]) == {
        "AdvBench",
        "GCG",
        "PAIR",
        "StrongREJECT",
        "WildGuardTest",
        "wildjailbreak_adv",
    }
    sources = {
        source
        for profile in manifest["validation_profiles"].values()
        for source in profile["sources"]
    }
    assert not any(benchmark.lower() in source.lower() for benchmark in manifest["excluded_final_benchmarks"] for source in sources)
