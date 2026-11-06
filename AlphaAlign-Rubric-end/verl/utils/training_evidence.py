"""JSON evidence for real training events; no model/torch dependencies.

These diagnostics describe rubric judgments, never independently verified harm.
"""
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev


def redact_config(value):
    if isinstance(value, Mapping):
        return {str(k): ("[REDACTED]" if any(term in str(k).lower() for term in
                ("api_key", "access_key", "secret", "password", "credential", "authorization"))
                or str(k).lower() in {"token", "api_base", "api_url"}
                else redact_config(v)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [redact_config(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def append_event(directory, event):
    """Separate PID files avoid interleaving judge replicas/processes."""
    if not directory:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The caller serializes threads when sharing a process.
    with (directory / f"events_{os.getpid()}.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")


def fingerprint(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def annotate_groups(records):
    """Use actual advantages from the trainer; do not infer them from raw rewards."""
    groups = defaultdict(list)
    for record in records:
        key = record.get("uid")
        if key is None:
            key = f"unidentified:{record['sample_index']}"
        groups[str(key)].append(record)
        extra = record.get("extra_info") or {}
        result = record.get("reward_result") or {}
        count = extra.get("critical_safety_rule_count")
        passes = (result.get("rubric_rule_passes") or {}).get("outcome_safety_rule", [])
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0
        valid = count > 0 and len(passes) >= count and all(x in (0, 1) for x in passes[:count])
        if result.get("judge_parse_valid") is False:
            valid = False
        if result.get("extra_info_valid") is False:
            valid = False
        failed = any(x == 0 for x in passes[:count]) if valid else None
        record["critical_rule_failed"] = failed
        record["critical_rule_status"] = "known" if valid else "unavailable_or_not_applicable"
        record["critical_failed_positive_advantage"] = (
            failed and record.get("advantage_mean") is not None and record["advantage_mean"] > 0
        ) if valid else None
        record["failed_rules"] = [
            {"category": category, "index": i, "rule": text}
            for category, rules in (result.get("rubric_rules") or {}).items()
            for i, text in enumerate(rules)
            if i < len((result.get("rubric_rule_passes") or {}).get(category, []))
            and result["rubric_rule_passes"][category][i] == 0
        ]
    group_summaries = []
    for uid, members in groups.items():
        values = [r["optimization_reward"] for r in members
                  if r.get("optimization_reward") is not None and math.isfinite(r["optimization_reward"])]
        known = [r["critical_rule_failed"] for r in members if r["critical_rule_failed"] is not None]
        expected = max([int(r.get("rollout_count") or 0) for r in members] + [0])
        group = {
            "uid": uid, "candidates": len(members),
            "expected_candidates": expected or None,
            "complete": expected > 0 and len(members) == expected,
            "optimization_reward_mean": mean(values) if values else None,
            "optimization_reward_std": stdev(values) if len(values) > 1 else 0.0,
            "critical_judgments": len(known),
            "all_critical_failed": all(known) if len(known) == len(members) else None,
            "mixed_critical_outcomes": any(known) and not all(known) if len(known) == len(members) else None,
        }
        group_summaries.append(group)
        for ordinal, record in enumerate(members, 1):
            if record.get("rollout_index") is None:
                record["rollout_index"] = ordinal
            record["group"] = group
    return group_summaries


def summarize(records, groups):
    profiles = defaultdict(list)
    for record in records:
        profiles[record.get("data_source", "unknown")].append(record)
    def counts(items):
        known = [r for r in items if r.get("critical_rule_failed") is not None]
        clipped = [r for r in items if (r.get("reward_result") or {}).get("balanced_raw_unclipped") is not None]
        return {
            "responses": len(items), "critical_judgments": len(known),
            "critical_failures": sum(r["critical_rule_failed"] for r in known),
            "critical_failed_positive_advantage": sum(r.get("critical_failed_positive_advantage") is True for r in known),
            "judge_parse_failures": sum((r.get("reward_result") or {}).get("judge_parse_valid") is False for r in items),
            "reward_clipped": sum(abs(r["reward_result"]["balanced_raw_unclipped"]) > 1 for r in clipped),
            "possible_length_limit_hits": sum(r.get("hit_response_length_limit") is True for r in items),
        }
    return {
        "schema_version": 2, "interpretation": "rubric_diagnostics_not_benchmark_harm_labels",
        **counts(records), "profiles": {k: counts(v) for k, v in profiles.items()},
        "groups": len(groups),
        "all_critical_failed_groups": sum(g["all_critical_failed"] is True for g in groups),
        "mixed_critical_groups": sum(g["mixed_critical_outcomes"] is True for g in groups),
        "zero_reward_std_groups": sum(g["optimization_reward_std"] == 0 for g in groups),
    }


def select_complete_groups(records, limit):
    if limit <= 0:
        return records
    # A limit is soft: preserve complete candidate groups, even if the first exceeds it.
    selected, keys = [], set()
    for record in records:
        key = (record.get("group") or {}).get("uid", str(record["sample_index"]))
        if key not in keys and len(selected) >= limit:
            continue
        keys.add(key)
        selected.append(record)
    return selected


def write_step(trace_dir, step, records, max_samples=0):
    groups = annotate_groups(records)
    report = summarize(records, groups)
    report["global_step"] = step
    selected = select_complete_groups(records, max_samples)
    report["saved_responses"] = len(selected)
    report["summary_uses_full_batch"] = True
    root = Path(trace_dir)
    root.mkdir(parents=True, exist_ok=True)
    # A resumed attempt gets its own root from the launcher, never overwrites old evidence.
    path = root / f"step_{step:08d}.jsonl"
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(redact_config(record), ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)
    atomic_json(root / f"step_{step:08d}.summary.json", report)
    atomic_json(root / f"step_{step:08d}.groups.json", groups)
    return report
