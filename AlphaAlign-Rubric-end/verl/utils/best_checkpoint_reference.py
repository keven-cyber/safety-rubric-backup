"""Name an immutable, complete numbered checkpoint without copying its tensors.

This changes storage only. Metric selection and checkpoint contents are unchanged.
Legacy physical best copies remain readable and are never deleted by this path.
"""
import os
from pathlib import Path


MODE = "numbered_checkpoint_reference_v1"


def enabled():
    return os.getenv("RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE", "0") == "1"


def reference(root, step):
    from verl.utils.checkpoint_recovery import require_checkpoint
    root = Path(root).resolve()
    if type(step) is not int or step <= 0:
        raise ValueError("Best reference requires a positive checkpoint step")
    path = root / f"global_step_{step}"
    if require_checkpoint(path, 4) != step:
        raise ValueError("Best reference is not a complete aligned checkpoint")
    if int((root / "latest_checkpointed_iteration.txt").read_text()) < step:
        raise ValueError("Best reference cannot precede checkpoint commit")
    return dict(checkpoint_storage=MODE, best_checkpoint_relative_path=path.name)


def resolve(root, info):
    """Resolve only the explicit storage version; never silently change a best."""
    root = Path(root)
    step = info.get("best_metric_step")
    mode = info.get("checkpoint_storage")
    if mode is None:
        return root / f"global_step_best_{step}"
    if mode != MODE or info.get("best_checkpoint_relative_path") != f"global_step_{step}":
        raise ValueError("Unknown or unsafe best-checkpoint reference")
    reference(root, step)
    return root / info["best_checkpoint_relative_path"]
