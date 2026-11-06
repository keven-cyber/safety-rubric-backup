import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.utils import first_batch_snapshot as s


class Tokenizer:
    def decode(self, ids, skip_special_tokens):
        assert skip_special_tokens is True
        return '|'.join(map(str, ids))


def batch():
    # Synthetic CPU tensors, not real GPU policy responses.
    tensors = dict(prompts=torch.tensor([[0, 3, i + 4] for i in range(64)]),
        responses=torch.tensor([[i + 9, 2, 0] for i in range(64)]),
        attention_mask=torch.tensor([[0, 1, 1, 1, 1, 0]] * 64),
        response_mask=torch.tensor([[1, 1, 0]] * 64),
        position_ids=torch.tensor([[0, 0, 1, 2, 3, 4]] * 64))
    metadata = dict(uid=np.array(['uid-%d' % (i // 8) for i in range(64)], dtype=object),
        raw_prompt=np.array(['original question\nunchanged'] * 64, dtype=object),
        extra_info=np.array([{'sample_id': 'item-%d' % i, 'original_rule': '原规则'}
                             for i in range(64)], dtype=object))
    return tensors, metadata


def capture(path, restored=90):
    t, n = batch()
    return s.capture(path, 'synthetic-test', restored + 1, restored, t, n,
                     {'original_meta': True}, Tokenizer(), 4), t, n


@pytest.mark.parametrize('restored', [0, 30, 90, 120])
def test_complete_originals_identity_tokens_rng_and_no_mutation(tmp_path, restored):
    run = tmp_path.resolve()
    t, n = batch()
    originals = {k: v.clone() for k, v in t.items()}
    old_meta = copy.deepcopy(n)
    torch_rng, numpy_rng = torch.get_rng_state().clone(), copy.deepcopy(np.random.get_state())
    ref = s.capture(run, 'synthetic-test', restored + 1, restored, t, n,
                    {'untouched': ['x']}, Tokenizer(), 4)
    m = s.verify(run, ref, 'synthetic-test', restored + 1, restored, n['uid'])
    p = run / 'first_batch' / m['snapshot']
    x = json.loads(p.read_text())
    assert len(x['records']) == 64 and x['records'][0]['response'] == '9|2'
    assert x['tensors']['responses'] == t['responses'].tolist()
    assert x['non_tensor_batch']['extra_info'][0]['original_rule'] == '原规则'
    assert m['rollout_counts'] == [8] * 8
    assert torch.equal(torch_rng, torch.get_rng_state())
    current = np.random.get_state()
    assert current[0] == numpy_rng[0] and np.array_equal(current[1], numpy_rng[1])
    assert current[2:] == numpy_rng[2:]
    assert all(torch.equal(v, originals[k]) for k, v in t.items())
    assert all(np.array_equal(n[k], old_meta[k]) for k in n)
    assert json.loads(p.with_name(p.name + '.integrity.json').read_text()) == {
        'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'bytes': p.stat().st_size}


def test_immutable_no_second_capture(tmp_path):
    capture(tmp_path.resolve())
    with pytest.raises(FileExistsError):
        capture(tmp_path.resolve())


@pytest.mark.parametrize('target', ['manifest', 'snapshot', 'uid_order', 'wrong_step', 'wrong_run'])
def test_tamper_or_misalignment_is_rejected(tmp_path, target):
    run = tmp_path.resolve()
    ref, t, n = capture(run)
    args = [run, ref, 'synthetic-test', 91, 90, n['uid']]
    if target == 'manifest':
        p = run / ref['relative_path'];p.write_bytes(p.read_bytes() + b' ')
    elif target == 'snapshot':
        p = run / 'first_batch/step_00000091.original.json';p.write_bytes(p.read_bytes() + b' ')
    elif target == 'uid_order':
        args[-1] = n['uid'][::-1]
    elif target == 'wrong_step':
        args[3] = 92
    else:
        args[2] = 'another-run'
    with pytest.raises(ValueError):
        s.verify(*args)


@pytest.mark.parametrize('invalid', ['groups', 'tokens', 'unknown', 'nan', 'first_step'])
def test_invalid_input_never_publishes_ready(tmp_path, invalid):
    run = tmp_path.resolve();t, n = batch();step = 91
    if invalid == 'groups':n['uid'][0] = 'another-uid'
    elif invalid == 'tokens':t['response_mask'][0, 0] = 2
    elif invalid == 'unknown':n['extra_info'][0] = {'object': object()}
    elif invalid == 'nan':n['extra_info'][0] = {'value': float('nan')}
    else:step = 92
    with pytest.raises((ValueError, TypeError)):
        s.capture(run, 'synthetic-test', step, 90, t, n, {}, Tokenizer(), 4)
    assert not list(run.glob('first_batch/*.ready.json'))


def test_disk_failure_cannot_mark_ready(tmp_path, monkeypatch):
    run = tmp_path.resolve()
    def disk_error(_):raise OSError('injected fsync error')
    monkeypatch.setattr(s.os, 'fsync', disk_error)
    with pytest.raises(OSError):capture(run)
    assert not list(run.glob('first_batch/*.ready.json'))


def test_snapshot_symlink_refused(tmp_path):
    run = (tmp_path / 'run');run.mkdir();outside = tmp_path / 'outside';outside.mkdir()
    (run / 'first_batch').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):capture(run.resolve())
    assert not list(outside.iterdir())
