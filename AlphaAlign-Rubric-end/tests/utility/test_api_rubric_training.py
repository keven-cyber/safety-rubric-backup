import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.select_final_checkpoints import select
from scripts.audit_api_rubric_artifact import audit
from scripts.safety_data_pipeline import (
    PROFILE_FILES,
    RULE_COLUMNS,
    _read_parquet_with_retry,
    _write_frame,
    create_plan,
    load_dataset,
)

ROOT = Path(__file__).resolve().parents[2]


def checkpoints(root, best):
    for step in (600, 900, 990):
        (root / f"global_step_{step}" / "actor").mkdir(parents=True)
    (root / f"global_step_best_{best}" / "actor").mkdir(parents=True)
    (root / "best_checkpoint_info.json").write_text(json.dumps({"best_metric_step": best}))


@pytest.mark.parametrize("best,expected", [(600, 3), (900, 2), (990, 2)])
def test_exact_trio_deduplicates(tmp_path, best, expected):
    checkpoints(tmp_path, best)
    result = select(tmp_path)
    assert result["unique_checkpoint_count"] == expected
    assert {r["step"] for r in result["checkpoints"]} == {900, 990, best}
    row = next(r for r in result["checkpoints"] if r["step"] == best)
    assert "best" in row["roles"]
    assert "global_step_best_" in row["checkpoint"]


def test_missing_900_fails_not_fallback(tmp_path):
    checkpoints(tmp_path, 990)
    shutil.rmtree(tmp_path / "global_step_900")
    with pytest.raises(ValueError, match="missing"):
        select(tmp_path)


def test_online_five_checkpoint_selection_includes_1080_and_last(tmp_path):
    for step in (900, 990, 1080, 1170):
        (tmp_path / f"global_step_{step}" / "actor").mkdir(parents=True)
    (tmp_path / "global_step_best_1080" / "actor").mkdir(parents=True)
    (tmp_path / "best_checkpoint_info.json").write_text(
        json.dumps({"best_metric_step": 1080})
    )
    result = select(tmp_path, ("900", "990", "1080", "last", "best"), 1170)
    assert result["last_step"] == 1170
    assert result["unique_checkpoint_count"] == 4
    assert {row["step"] for row in result["checkpoints"]} == {900, 990, 1080, 1170}
    best = next(row for row in result["checkpoints"] if row["step"] == 1080)
    assert best["roles"] == ["1080", "best"]
    assert "global_step_best_1080" in best["checkpoint"]


def test_online_incomplete_last_is_rejected(tmp_path):
    checkpoints(tmp_path, 990)
    with pytest.raises(ValueError, match="expected 1170"):
        select(tmp_path, ("900", "990", "1080", "last", "best"), 1170)


def dataset(root):
    for split in ("train", "val"):
        for profile, name in PROFILE_FILES.items():
            info = {"sample_id": f"{split}-{profile}", "reward_profile": profile,
                    "raw_question": f"{split} {profile}", "prompt": f"{split} {profile}",
                    "critical_safety_rule_count": 0,
                    **{k: ["An original rule"] for k in RULE_COLUMNS}}
            path = root / split / name
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"extra_info": info, "data_source": profile,
                           **{k: info[k] for k in RULE_COLUMNS}}]).to_parquet(path, index=False)
    (root / "manifest.json").write_text('{}')


def audit_fixture(tmp_path):
    source, output, work = [tmp_path / n for n in ("source", "output", "work")]
    dataset(source)
    shutil.copytree(source, output)
    work.mkdir()
    (work / "plan.jsonl").write_text(json.dumps({
        "job_id": "x", "kind": "rubric_review", "required_repair": False,
        "split": "train", "profile": "ordinary_harm",
        "sample_id": "train-ordinary_harm",
    }) + "\n")
    (work / "verified.jsonl").write_text(json.dumps({"job_id": "x", "status": "ok", "verification_status": "passed",
                                                    "factuality": {"accepted": True}}) + "\n")
    return source, output, work


def revise(output):
    path = output / "train" / PROFILE_FILES["ordinary_harm"]
    frame = pd.read_parquet(path)
    info = dict(frame.at[0, "extra_info"])
    info["outcome_safety_rule"] = ["A revised concrete rule"]
    frame.at[0, "extra_info"] = info
    frame.at[0, "outcome_safety_rule"] = info["outcome_safety_rule"]
    frame.to_parquet(path, index=False)


def test_zero_adaptive_plan(tmp_path):
    dataset(tmp_path)
    report, jobs = create_plan(tmp_path, adaptive_seed_count=0)
    assert len(jobs) == 4
    assert all(j["kind"] == "rubric_review" for j in jobs)
    assert report["plan"]["planned_new_adaptive_rows"] == 0


def test_audit_archives_actual_changes(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    revise(output)
    report = audit(source, output, work)
    assert report["changed_rows"] == 1
    assert report["changed_verified_rows"] == 1
    assert report["retained_original_after_failed_optional_review"] == 0
    assert report["guard_alignment"] == "not_empirically_validated"
    assert (output / "rubric_changes.jsonl").is_file()


def test_audit_accepts_only_verified_generated_questions(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    adaptive_job = {
        "job_id": "adaptive:val:trigger_roleplay_override:seed",
        "kind": "adaptive_generation",
        "split": "val",
        "profile": "adaptive_harm",
        "sample_id": "seed",
    }
    (work / "plan.jsonl").write_text(json.dumps(adaptive_job) + "\n")
    (work / "verified.jsonl").write_text(json.dumps({
        "job_id": adaptive_job["job_id"], "status": "ok",
        "verification_status": "passed", "factuality": {"accepted": True},
    }) + "\n")

    path = output / "val" / PROFILE_FILES["adaptive_harm"]
    frame = pd.read_parquet(path)
    generated = frame.iloc[0].copy()
    info = dict(generated["extra_info"])
    info.update({
        "sample_id": "generated-val-1",
        "sample_family": "family-val-1",
        "source_dataset": "api_safety_alignment_pipeline",
        "mutation_strategy": "trigger_roleplay_override",
        "raw_question": "A distinct generated validation prompt",
        "prompt": "A distinct generated validation prompt",
    })
    generated["extra_info"] = info
    pd.concat([frame, generated.to_frame().T], ignore_index=True).to_parquet(path, index=False)

    report = audit(source, output, work)
    assert report["changed_rows"] == 0
    assert report["new_questions_by_split"] == {"train": 0, "val": 1}
    assert (output / "generated_questions.jsonl").is_file()


def test_unchanged_dataset_cannot_be_called_new(tmp_path):
    with pytest.raises(ValueError, match="no rubric changes"):
        audit(*audit_fixture(tmp_path))


def test_unverified_replacement_blocks_training(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    revise(output)
    (work / "verified.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="Unverified rubric replacement"):
        audit(source, output, work)


def test_failed_optional_review_retains_identical_source_rubric(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    revise(output)
    optional_job = {
        "job_id": "optional-failed", "kind": "rubric_review",
        "required_repair": False, "split": "train", "profile": "borderline",
        "sample_id": "train-borderline",
    }
    with (work / "plan.jsonl").open("a") as handle:
        handle.write(json.dumps(optional_job) + "\n")
    with (work / "verified.jsonl").open("a") as handle:
        handle.write(json.dumps({
            "job_id": "optional-failed", "status": "review_required",
            "verification_status": "review_required", "factuality": {"accepted": False},
        }) + "\n")

    report = audit(source, output, work)

    assert report["changed_verified_rows"] == 1
    assert report["retained_original_after_failed_optional_review"] == 1


def test_unverified_required_repair_blocks_even_when_source_is_retained(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    plan = json.loads((work / "plan.jsonl").read_text())
    plan["required_repair"] = True
    (work / "plan.jsonl").write_text(json.dumps(plan) + "\n")
    (work / "verified.jsonl").write_text(json.dumps({
        "job_id": "x", "status": "review_required",
        "verification_status": "review_required", "factuality": {"accepted": False},
    }) + "\n")

    with pytest.raises(ValueError, match="Required rubric repair was not verified"):
        audit(source, output, work)


def test_question_change_rejected(tmp_path):
    source, output, work = audit_fixture(tmp_path)
    path = output / "train" / PROFILE_FILES["ordinary_harm"]
    frame = pd.read_parquet(path)
    info = dict(frame.at[0, "extra_info"])
    info["prompt"] = "Different question"
    frame.at[0, "extra_info"] = info
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Question"):
        audit(source, output, work)


def test_parquet_read_retries_transient_arrow_error(tmp_path, monkeypatch):
    path = tmp_path / "data.parquet"
    pd.DataFrame({"value": [1]}).to_parquet(path, index=False)
    original = pd.read_parquet
    calls = {"count": 0}

    def flaky_read(candidate, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise ValueError("Parquet file size is 0 bytes")
        return original(candidate, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", flaky_read)
    monkeypatch.setattr("scripts.safety_data_pipeline.time.sleep", lambda _: None)
    frame = _read_parquet_with_retry(path, attempts=3, initial_delay=0)
    assert frame["value"].tolist() == [1]
    assert calls["count"] == 3


def test_load_dataset_reports_corrupt_file_path(tmp_path, monkeypatch):
    dataset(tmp_path)
    corrupt = tmp_path / "train" / PROFILE_FILES["ordinary_harm"]
    corrupt.write_bytes(b"")
    monkeypatch.setenv("RUBRIC_PARQUET_READ_ATTEMPTS", "1")
    with pytest.raises(RuntimeError, match=str(corrupt)):
        load_dataset(tmp_path)


def test_write_frame_republishes_zero_byte_staging_file(tmp_path, monkeypatch):
    source = tmp_path / "source"
    dataset(source)
    source_path = source / "train" / PROFILE_FILES["ordinary_harm"]
    frame = pd.read_parquet(source_path)
    frame["prompt"] = [[{"role": "user", "content": "test"}]]
    frame["category"] = "test"
    frame["instruction_prompt"] = "test"
    destination = tmp_path / "published.parquet"
    original_copyfile = shutil.copyfile
    calls = {"count": 0}

    def first_copy_is_empty(src, dst, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            Path(dst).write_bytes(b"")
            return dst
        return original_copyfile(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", first_copy_is_empty)
    monkeypatch.setattr("scripts.safety_data_pipeline.time.sleep", lambda _: None)
    monkeypatch.setenv("RUBRIC_PARQUET_READ_ATTEMPTS", "1")
    _write_frame(frame, destination)
    assert calls["count"] == 2
    assert len(pd.read_parquet(destination)) == len(frame)


def test_launcher_and_evaluation_wiring():
    launcher = (ROOT / "scripts/train_api_rubric.sh").read_text()
    assert "RUBRIC_ADAPTIVE_SEEDS=0" in launcher
    assert "TRAIN_SAVE_FREQ=30" in launcher
    assert "FINAL_SAFETY_EVAL_STEPS=900,990,best" in launcher
    assert launcher.index("audit_api_rubric_artifact.py") < launcher.index('exec bash "${PROJECT_ROOT}/train_val.sh"')
    script = (ROOT / "train_val.sh").read_text()
    assert 'GENERAL_EVAL_MODELS=("${MERGED_THIS_RUN[@]}")' in script
    for path in (ROOT / "train_val.sh", ROOT / "scripts/train_api_rubric.sh"):
        subprocess.run(["bash", "-n", str(path)], check=True)
