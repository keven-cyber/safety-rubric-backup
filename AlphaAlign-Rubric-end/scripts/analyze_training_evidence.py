#!/usr/bin/env python3
"""Summarize actual run evidence into diagnostics and traceable example cases."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def analyze(run_dir, max_cases=100):
    run_dir = Path(run_dir)
    summary_files = sorted((run_dir / "rollouts").glob("step_*.summary.json"))
    totals, profiles, failed_rules = Counter(), defaultdict(Counter), Counter()
    keys = ("responses", "critical_judgments", "critical_failures", "critical_failed_positive_advantage",
            "judge_parse_failures", "reward_clipped", "possible_length_limit_hits", "groups",
            "all_critical_failed_groups", "mixed_critical_groups", "zero_reward_std_groups")
    steps = []
    for path in summary_files:
        report = json.loads(path.read_text())
        steps.append(report["global_step"])
        totals.update({k: report.get(k, 0) for k in keys})
        for profile, values in report.get("profiles", {}).items():
            profiles[profile].update(values)
    case_counts, cases = Counter(), []
    for path in sorted((run_dir / "rollouts").glob("step_*.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                record = json.loads(line)
                for rule in record.get("failed_rules", []):
                    failed_rules[(rule["category"], rule["rule"])] += 1
                tags = []
                if record.get("critical_failed_positive_advantage"):
                    tags.append("critical_failed_positive_advantage")
                if record.get("critical_rule_failed"):
                    tags.append("critical_rule_failure")
                if (record.get("group") or {}).get("all_critical_failed"):
                    tags.append("all_candidates_failed_critical_rules")
                if (record.get("reward_result") or {}).get("judge_parse_valid") is False:
                    tags.append("judge_parse_failure")
                if record.get("hit_response_length_limit"):
                    tags.append("possible_truncation")
                if record.get("critical_rule_failed") is False and record.get("advantage_mean", 0):
                    if record["advantage_mean"] > 0:
                        tags.append("critical_pass_positive_advantage")
                for tag in tags:
                    # Bound disk/memory while retaining both useful and failed examples.
                    if case_counts[tag] < max_cases:
                        cases.append({"diagnostic": tag, "source_file": str(path.relative_to(run_dir)),
                                      "source_line": line_number, "record": record})
                        case_counts[tag] += 1
    report = {
        "schema_version": 1, "steps": steps, "totals": dict(totals),
        "profiles": {k: dict(v) for k, v in profiles.items()},
        "frequent_failed_rules_in_saved_traces": [
            {"category": category, "rule": rule, "count": count}
            for (category, rule), count in failed_rules.most_common(50)
        ],
        "case_counts": dict(case_counts),
        "interpretation": "Training rubric diagnostics, not final safety benchmarks or proven causes. Counts are over sampled responses, not unique questions.",
        "evidence_available": bool(summary_files),
    }
    output = run_dir / "analysis"
    output.mkdir(exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "casebook.jsonl").open("w", encoding="utf-8") as stream:
        for case in cases:
            stream.write(json.dumps(case, ensure_ascii=False) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-cases-per-type", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(analyze(args.run_dir, args.max_cases_per_type), ensure_ascii=False, indent=2))
