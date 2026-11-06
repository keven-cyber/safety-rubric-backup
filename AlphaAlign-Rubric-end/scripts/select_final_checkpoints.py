#!/usr/bin/env python3
"""Select required numbered, last and best checkpoints without substitution."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def select(root, requested=(900, 990, "best"), expected_last_step=0):
    root = Path(root)
    if (root / 'validation_replay_completed.json').exists():
        from verl.utils.final_stage_resume import verify_replay_selection
        return verify_replay_selection(root, requested, expected_last_step)
    if (root / 'new_reevaluation_completed.json').exists():
        from verl.utils.frozen_validation_review import selection
        return selection(root, requested, expected_last_step)
    if "best" in requested:
        from verl.utils.validation_recovery import require_complete_best
        require_complete_best(root)
    info = json.loads((root / "best_checkpoint_info.json").read_text())
    best = info.get("best_metric_step")
    if type(best) is not int or best <= 0:
        raise ValueError("Missing valid best_metric_step in best_checkpoint_info.json")
    numbered = [
        int(path.name.removeprefix("global_step_"))
        for path in root.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    ]
    if not numbered:
        raise ValueError(f"No numbered checkpoints found under {root}")
    last = max(numbered)
    if expected_last_step and last != expected_last_step:
        raise ValueError(
            f"Last checkpoint is step {last}, expected {expected_last_step}; "
            "refusing to label an incomplete run as last"
        )
    resolved = [best if role == "best" else last if role == "last" else int(role)
                for role in requested]
    rows = []
    for step in sorted(set(resolved)):
        # Prefer the actual best artifact when a numbered copy also exists.
        if step == best:
            from verl.utils.best_checkpoint_reference import resolve
            path = resolve(root, info)
        else:
            path = root / f"global_step_{step}"
        if not (path / "actor").is_dir():
            raise ValueError(f"Required checkpoint is missing: {path}/actor")
        roles = [str(role) for role, value in zip(requested, resolved) if value == step]
        rows.append({"step": step, "checkpoint": str(path), "roles": roles})
    return {"requested": list(requested), "last_step": last, "best_metric": info,
            "unique_checkpoint_count": len(rows), "checkpoints": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requested", default="900,990,best",
                        help="Comma-separated exact steps and/or last,best")
    parser.add_argument("--expected-last-step", type=int, default=0)
    args = parser.parse_args()
    requested = tuple(part.strip() for part in args.requested.split(",") if part.strip())
    if not requested or any(part not in {"last", "best"} and not part.isdigit()
                            for part in requested):
        parser.error("--requested must contain steps and/or last,best")
    result = select(args.root, requested, args.expected_last_step)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["checkpoints"]:
        print(row["checkpoint"])


if __name__ == "__main__":
    main()
