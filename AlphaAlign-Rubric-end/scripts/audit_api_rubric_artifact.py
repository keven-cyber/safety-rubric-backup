#!/usr/bin/env python3
"""Fail closed before training; archive actual rubric deltas and provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path

try:
    from .safety_data_pipeline import PROFILE_FILES, RULE_COLUMNS, load_dataset, _extra, _request, _canonical_extra_info
except ImportError:
    from safety_data_pipeline import PROFILE_FILES, RULE_COLUMNS, load_dataset, _extra, _request, _canonical_extra_info


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _is_verified(record):
    return (
        record.get("status") == "ok"
        and record.get("verification_status") == "passed"
        and bool((record.get("factuality") or {}).get("accepted"))
    )


def _job_row_key(job):
    required = ("split", "profile", "sample_id")
    missing = [field for field in required if not job.get(field)]
    if missing:
        raise ValueError(
            f"Rubric job is missing row identity {missing}: {job.get('job_id', '<unknown>')}"
        )
    return tuple(str(job[field]) for field in required)


def audit(source, output, work):
    source, output, work = map(Path, (source, output, work))
    before, after = load_dataset(source), load_dataset(output)
    records = [json.loads(line) for line in (work / "verified.jsonl").read_text().splitlines() if line.strip()]
    jobs = [json.loads(line) for line in (work / "plan.jsonl").read_text().splitlines() if line.strip()]
    by_id = {r["job_id"]: r for r in records if r.get("job_id")}
    if not jobs or any(
        j.get("kind") not in {"rubric_review", "adaptive_generation"}
        or j.get("split") not in {"train", "val"}
        or (j.get("kind") == "rubric_review" and j.get("split") != "train")
        for j in jobs
    ):
        raise ValueError("Plan contains an unsupported job kind or split")
    jobs_by_row = {}
    for job in jobs:
        if job["kind"] != "rubric_review":
            continue
        key = _job_row_key(job)
        if key in jobs_by_row:
            raise ValueError(f"Multiple rubric jobs target the same row: {key}")
        jobs_by_row[key] = job

    changes, changed_rows, new_questions, hashes = [], set(), [], {}
    adaptive_jobs = [job for job in jobs if job["kind"] == "adaptive_generation"]
    expected_new_by_split = {
        split: sum(job["split"] == split for job in adaptive_jobs)
        for split in ("train", "val")
    }
    for profile, filename in PROFILE_FILES.items():
        for split in ("train", "val"):
            key = (split, profile)
            a, b = before[key], after[key]
            if len(b) < len(a):
                raise ValueError(f"Rows were removed: {key}")
            added = len(b) - len(a)
            if profile != "adaptive_harm" and added:
                raise ValueError(f"New questions may only extend adaptive_harm: {key}")
            hashes[f"{split}/{filename}"] = {"source": sha(source / split / filename),
                                             "generated": sha(output / split / filename)}
            if split == "val" and not added and hashes[f"{split}/{filename}"]["source"] != hashes[f"{split}/{filename}"]["generated"]:
                raise ValueError("Validation data must be byte-identical")
            for (_, old), (_, new) in zip(a.iterrows(), b.iloc[:len(a)].iterrows()):
                oi, ni = _canonical_extra_info(_extra(old)), _canonical_extra_info(_extra(new))
                identity_fields = ("sample_id", "reward_profile", "question", "raw_question", "prompt", "sample_family")
                if _request(old) != _request(new) or any(oi.get(k) != ni.get(k) for k in identity_fields):
                    raise ValueError("Question, identity or profile changed")
                non_rubric = [c for c in old.index if c not in (*RULE_COLUMNS, "extra_info")]
                if old[non_rubric].to_json() != new[non_rubric].to_json():
                    raise ValueError("Non-rubric training columns changed")
                fields = (*RULE_COLUMNS, "critical_safety_rule_count", "response_strategy")
                prior = {f: oi.get(f) for f in fields}
                revised = {f: ni.get(f) for f in fields}
                if prior != revised:
                    row_key = (split, profile, str(oi.get("sample_id")))
                    changed_rows.add(row_key)
                    changes.append({"split": split, "profile": profile, "sample_id": oi.get("sample_id"),
                                    "before": prior, "after": revised})
            for _, new in b.iloc[len(a):].iterrows():
                info = _canonical_extra_info(_extra(new))
                if info.get("source_dataset") != "api_safety_alignment_pipeline":
                    raise ValueError("New question lacks API generation provenance")
                if not info.get("mutation_strategy") or not info.get("sample_family"):
                    raise ValueError("New question lacks mutation or family provenance")
                new_questions.append({
                    "split": split,
                    "profile": profile,
                    "sample_id": info.get("sample_id"),
                    "sample_family": info.get("sample_family"),
                    "mutation_strategy": info.get("mutation_strategy"),
                })

    actual_new_by_split = {
        split: sum(item["split"] == split for item in new_questions)
        for split in ("train", "val")
    }
    verified_adaptive_by_split = {
        split: sum(
            job["split"] == split and _is_verified(by_id.get(job["job_id"], {}))
            for job in adaptive_jobs
        )
        for split in ("train", "val")
    }
    if actual_new_by_split != verified_adaptive_by_split:
        raise ValueError(
            "Generated question counts do not match verified adaptive jobs: "
            f"actual={actual_new_by_split}, verified={verified_adaptive_by_split}, "
            f"planned={expected_new_by_split}"
        )

    unplanned_changes = sorted(changed_rows - set(jobs_by_row))
    if unplanned_changes:
        raise ValueError(f"Rubric rows changed without a planned API review: {unplanned_changes[:5]}")

    retained_original = 0
    verified_jobs = 0
    changed_verified_rows = 0
    for row_key, job in jobs_by_row.items():
        record = by_id.get(job["job_id"], {})
        verified = _is_verified(record)
        if verified:
            verified_jobs += 1
        if row_key in changed_rows:
            if not verified:
                raise ValueError(
                    f"Unverified rubric replacement changed training data: {job['job_id']}"
                )
            changed_verified_rows += 1
        elif not verified:
            if job.get("required_repair"):
                raise ValueError(
                    f"Required rubric repair was not verified: {job['job_id']}"
                )
            # An optional failed review is safe only because the full row-level
            # comparison above proved that its source rubric was retained.
            retained_original += 1

    if not changes and not new_questions:
        raise ValueError(
            "API accepted no rubric changes and no data changes; "
            "do not retrain the old dataset as new"
        )
    (output / "rubric_changes.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in changes))
    (output / "generated_questions.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in new_questions)
    )
    report = {"source_dataset": str(source), "changed_rows": len(changes), "api_jobs": len(jobs),
              "verified_api_jobs": verified_jobs,
              "changed_verified_rows": changed_verified_rows,
              "retained_original_after_failed_optional_review": retained_original,
              "new_questions": len(new_questions),
              "new_questions_by_split": actual_new_by_split,
              "files": hashes, "generation_model": "qwen3.7-plus",
              "guard_alignment": "not_empirically_validated", "credentials_archived": False}
    report["revision_mode"] = os.getenv("RUBRIC_REVISION_MODE", "general")
    (output / "api_revision_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output, args.work), ensure_ascii=False))
