"""Preserve the full native304 validation before its first reward call.

This is generation/scoring evidence only, not a frozen-policy recovery claim.
The native reward, validation ranking and optimizer remain unchanged.
"""
import os
from pathlib import Path
from verl.utils.first_batch_snapshot import native, _publish


def original(trainer, batch):
    if os.getenv('BASELINE_VALIDATION_ARCHIVE') != '1':
        return None
    if len(batch.batch['responses']) != 304:
        raise ValueError('Archive requires the entire original304 before scoring')
    keys = ['prompts', 'responses', 'attention_mask', 'position_ids']
    tensors = {name: native(batch.batch[name]) for name in keys}
    root = Path(os.environ['BASELINE_RUN_DIR']) / 'validation_originals' / ('step_%08d' % trainer.global_steps)
    root.mkdir(parents=True, exist_ok=False)
    identity = _publish(root / 'original304.json', native(dict(global_step=trainer.global_steps,
        count=304, tensors=tensors, non_tensor_batch=batch.non_tensor_batch,
        meta_info=batch.meta_info, stage='before_native_reward',
        policy_origin=os.environ.get('POLICY_MODEL_PATH'), frozen_policy_copy_verified=False)))
    _publish(root / 'original304.integrity.json', identity)
    return root


def scored(root, scores):
    if root is None:
        return
    values = native(scores)  # strict finite conversion before any aggregate
    if len(values) != 304 or not all(type(v) in (int, float) for v in values):
        raise ValueError('Native validation scoring must complete all304')
    _publish(root / 'scored304.json', dict(count=304, scores=values, status='finite_complete'))
