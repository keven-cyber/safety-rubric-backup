"""Read-only first-update gate for the authorized fresh baseline comparison.

It archives the native batch and checks it without changing reward/advantages.
An incomplete or nonfinite batch fails; no missing value is imputed.
"""
from collections import Counter
import os
from pathlib import Path

from verl.utils.first_batch_snapshot import capture, verify, _publish


def snapshot(trainer, batch, first_step):
    if os.getenv('BASELINE_FIRST_BATCH_GATE') != '1' or trainer.global_steps != first_step:
        return None
    root = os.environ['BASELINE_RUN_DIR']
    return capture(root, Path(root).name, first_step, first_step - 1,
                   batch.batch, batch.non_tensor_batch, batch.meta_info,
                   trainer.tokenizer, trainer.actor_rollout_wg.world_size)


def gate(trainer, batch, first_step, reference):
    if os.getenv('BASELINE_FIRST_BATCH_GATE') != '1' or trainer.global_steps != first_step:
        return
    import torch
    root = os.environ['BASELINE_RUN_DIR']
    manifest = verify(root, reference, Path(root).name, first_step, first_step - 1,
                      batch.non_tensor_batch['uid'])
    counts = Counter(batch.non_tensor_batch['uid'])
    if len(counts) != 8 or set(counts.values()) != {8}:
        raise ValueError('Baseline requires eight aligned groups of eight')
    if len(trainer.train_dataset) != 3120 or len(trainer.val_dataset) != 304:
        raise ValueError('Baseline dataset count differs from shared comparison')
    finite = {}
    for name in ('token_level_scores', 'token_level_rewards', 'advantages', 'old_log_probs'):
        finite[name] = bool(torch.isfinite(batch.batch[name]).all())
        if not finite[name]:
            raise ValueError('Nonfinite native baseline tensor: ' + name)
    mask = batch.batch['response_mask']
    if not bool(((mask == 0) | (mask == 1)).all()) or not bool((mask.sum(-1) > 0).all()):
        raise ValueError('Invalid native baseline response mask')
    _publish(Path(root) / 'first_batch' / 'before_actor_update.json',
             dict(status='passed', global_step=first_step, optimizer_updates=0,
                  original_snapshot=manifest, finite=finite, native_algorithm_unchanged=True))


def actor_result(trainer, first_step, metrics):
    if os.getenv('BASELINE_FIRST_BATCH_GATE') != '1' or trainer.global_steps != first_step:
        return
    from verl.utils.first_batch_snapshot import native
    result = native(metrics)  # strict finite serialization; errors propagate
    if not result:
        raise ValueError('Missing first native actor-update metrics')
    _publish(Path(os.environ['BASELINE_RUN_DIR']) / 'first_batch' / 'actor_update.json',
             dict(status='finite_native_update_returned', global_step=first_step, metrics=result))
