"""Exclude complete GRPO groups after exhausted Judge failures.

Reward diagnostics are collected from heterogeneous rubric profiles.  Some
fields are therefore absent for part of a batch.  Keep those diagnostics
batch-aligned so a GRPO group can be removed without reusing full-batch
indices against a shorter list.
"""


def _missing_reward_extra(key):
    # Missing Judge failures must not make an otherwise valid rollout fail.
    # Other diagnostics use NaN to preserve the distinction between "zero"
    # and "not defined for this rubric profile".
    return False if key == "judge_failed" else float("nan")


def append_dense_reward_extras(storage, row_values, row_index):
    """Append one row while keeping every numeric diagnostic batch-aligned."""
    if row_index < 0:
        raise ValueError("row_index must be non-negative")

    existing = set(storage)
    for key in existing:
        storage[key].append(row_values.get(key, _missing_reward_extra(key)))

    for key, value in row_values.items():
        if key in existing:
            continue
        values = storage.setdefault(key, [])
        values.extend([_missing_reward_extra(key)] * row_index)
        values.append(value)


def select_aligned_reward_extras(values_by_key, keep, batch_size):
    """Select retained rows and quarantine malformed sparse diagnostics.

    The required ``judge_failed`` field is validated separately by
    :func:`valid_group_indices`.  Optional diagnostics must never crash the
    training update merely because a custom reward function returned them for
    only a subset of samples.
    """
    keep = list(keep)
    if batch_size < 0:
        raise ValueError("batch_size must be non-negative")
    if any(not isinstance(index, int) or index < 0 or index >= batch_size for index in keep):
        raise ValueError("Reward extra selection indices must be within the rollout batch")

    selected = {}
    skipped = {}
    for key, values in values_by_key.items():
        try:
            value_count = len(values)
        except TypeError:
            skipped[key] = None
            continue
        if value_count != batch_size:
            skipped[key] = value_count
            continue
        selected[key] = [values[index] for index in keep]
    return selected, skipped


def valid_group_indices(uids, failed, world_size=1):
    if len(uids) != len(failed):
        raise ValueError("Judge failure flags must align with rollout UIDs")
    bad = {uid for uid, invalid in zip(uids, failed) if invalid}
    keep = [i for i, uid in enumerate(uids) if uid not in bad]
    if not keep:
        raise RuntimeError("All GRPO groups failed Judge scoring; refusing an empty update")
    if len(keep) % world_size:
        raise RuntimeError("Valid GRPO groups do not divide across actor ranks; refusing a partial group")
    return keep, len(bad)
