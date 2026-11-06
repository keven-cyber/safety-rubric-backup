"""Immutable first-batch generation evidence for explicitly gated deployments.

This module reads generation tensors before scoring/filtering. It does not score,
sample, alter a batch, recover old responses, or claim an optimizer update.
"""
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path


def native(value):
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('Nonfinite first-batch evidence')
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise TypeError('Evidence mapping keys must already be strings')
        return {k: native(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [native(v) for v in value]
    package = type(value).__module__.split('.')[0]
    if package == 'torch' and hasattr(value, 'detach'):
        return native(value.detach().cpu().tolist())
    if package == 'numpy' and hasattr(value, 'tolist'):
        return native(value.tolist())
    raise TypeError('Unsupported evidence value: ' + type(value).__name__)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _root(run_dir):
    p = Path(run_dir)
    if not p.is_absolute() or p != p.resolve() or not p.is_dir():
        raise ValueError('Use the existing, absolute, nonsymlink run directory')
    target = p / 'first_batch'
    if target.is_symlink():
        raise ValueError('Snapshot directory cannot be a link')
    target.mkdir(mode=0o700, exist_ok=True)
    return p, target


def _publish(path, value):
    raw = (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode()
    if path.exists() or path.is_symlink():
        raise FileExistsError(str(path))
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    directory = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return {'sha256': _digest(raw), 'bytes': len(raw)}


def capture(run_dir, run_id, step, restored_step, tensors, non_tensors, meta_info,
            tokenizer, actor_world_size):
    """Capture all original64 rows, after balancing and BEFORE base scoring.

    Only the future first-batch gate calls this once. Detailed rollout tracing may
    remain off. A disk/serialization failure propagates; no ready marker is forged.
    """
    if (type(step) is not int or type(restored_step) is not int or restored_step < 0
            or step != restored_step + 1 or type(actor_world_size) is not int
            or actor_world_size <= 0):
        raise ValueError('Not the first fresh/restored training batch')
    if not isinstance(run_id, str) or not run_id:
        raise ValueError('Run identity required')
    keys = ('prompts', 'responses', 'attention_mask', 'response_mask', 'position_ids')
    if not all(k in tensors for k in keys):
        raise ValueError('Original generation tensors incomplete')
    tensor_copy = {k: native(tensors[k]) for k in keys}
    metadata = native(non_tensors)
    uids = metadata.get('uid', [])
    if (len(uids) != 64 or not all(isinstance(v, str) and v for v in uids)
            or len(Counter(uids)) != 8 or set(Counter(uids).values()) != {8}
            or 64 % actor_world_size):
        raise ValueError('Original first batch must contain eight complete eight-row groups')
    if any(len(tensor_copy[k]) != 64 for k in keys):
        raise ValueError('Generation rows are misaligned')
    records = []
    for i in range(64):
        prompt, response = tensor_copy['prompts'][i], tensor_copy['responses'][i]
        attention, mask = tensor_copy['attention_mask'][i], tensor_copy['response_mask'][i]
        if (len(attention) != len(prompt) + len(response) or len(mask) != len(response)
                or any(v not in (0, 1) for v in attention + mask)
                or any(type(v) is not int for v in prompt + response)):
            raise ValueError('Original token or mask layout invalid')
        prompt_ids = [v for v, keep in zip(prompt, attention[:len(prompt)]) if keep]
        response_ids = [v for v, keep in zip(response, mask) if keep]
        if not prompt_ids or not response_ids:
            raise ValueError('Missing original prompt/response token evidence')
        records.append(dict(sample_index=i, uid=uids[i],
            prompt=tokenizer.decode(prompt_ids, skip_special_tokens=True),
            response=tokenizer.decode(response_ids, skip_special_tokens=True)))
    payload = native(dict(schema_version=1, run_id=run_id, global_step=step,
        restored_global_step=restored_step, stage='before_base_reward_and_group_filter',
        records=records, tensors=tensor_copy, non_tensor_batch=metadata,
        meta_info=meta_info,
        tensor_layout={k: {'dtype': str(tensors[k].dtype), 'shape': list(tensors[k].shape)}
                       for k in keys},
        interpretation='Original generation snapshot only; not verdicts or optimizer-update proof'))
    _, folder = _root(run_dir)
    # Keep a claim even on partial failure: never overwrite previous evidence.
    with (folder / 'capture.claim').open('xb') as claim:
        claim.write(b'first-generation snapshot claim\n')
        claim.flush()
        os.fsync(claim.fileno())
    snapshot = folder / ('step_%08d.original.json' % step)
    identity = _publish(snapshot, payload)
    _publish(snapshot.with_name(snapshot.name + '.integrity.json'), identity)
    manifest = dict(schema_version=1, status='ready', run_id=run_id,
        global_step=step, restored_global_step=restored_step,
        actor_world_size=actor_world_size, response_count=64, prompt_count=8,
        rollout_counts=[8] * 8, snapshot=snapshot.name, **identity)
    ready = folder / ('step_%08d.ready.json' % step)
    ready_identity = _publish(ready, manifest)
    return dict(relative_path='first_batch/' + ready.name,
                manifest_sha256=ready_identity['sha256'])


def verify(run_dir, reference, run_id, step, restored_step, original_uids):
    """Re-read immutable bytes and original UID order before allowing an update."""
    run, folder = _root(run_dir)
    relative = reference.get('relative_path', '')
    if relative != 'first_batch/step_%08d.ready.json' % step:
        raise ValueError('Unexpected snapshot reference')
    p = run / relative
    if p.is_symlink():
        raise ValueError('Ready marker cannot be a link')
    raw = p.read_bytes()
    if _digest(raw) != reference.get('manifest_sha256'):
        raise ValueError('First-batch manifest changed')
    m = json.loads(raw)
    if (m.get('status') != 'ready' or m.get('run_id') != run_id
            or m.get('global_step') != step or m.get('restored_global_step') != restored_step
            or m.get('snapshot') != 'step_%08d.original.json' % step):
        raise ValueError('Snapshot identity mismatch')
    p = folder / m['snapshot']
    if p.is_symlink():
        raise ValueError('Snapshot cannot be a link')
    raw = p.read_bytes()
    if _digest(raw) != m['sha256'] or len(raw) != m['bytes']:
        raise ValueError('Original generation evidence changed')
    x = json.loads(raw)
    uids = native(original_uids)
    if (x['non_tensor_batch']['uid'] != uids or len(uids) != 64
            or [r['sample_index'] for r in x['records']] != list(range(64))
            or [r['uid'] for r in x['records']] != uids
            or len(Counter(uids)) != 8 or set(Counter(uids).values()) != {8}):
        raise ValueError('Original UID/order/group evidence does not align')
    return m
