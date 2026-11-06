"""Native baseline evidence hooks, synthetic CPU batches only."""
import ast
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from verl.utils import first_batch_snapshot

ROOT=Path(__file__).resolve().parents[2]

def gate_module(prefix):
    # The helper is identical in the two independent native trainer packages.
    path=ROOT/prefix/'utils/baseline_first_batch.py'
    scope={'__name__':'baseline_gate_fixture'}
    exec(compile(path.read_text(),str(path),'exec'),scope)
    return SimpleNamespace(**scope)

def fixture(tmp_path):
    from tests.utility.test_first_batch_snapshot import batch,Tokenizer
    tensors,metadata=batch()
    for name in ('token_level_scores','token_level_rewards','advantages','old_log_probs'):
        tensors[name]=torch.ones((64,3))
    data=SimpleNamespace(batch=tensors,non_tensor_batch=metadata,meta_info={})
    trainer=SimpleNamespace(global_steps=1,tokenizer=Tokenizer(),actor_rollout_wg=SimpleNamespace(world_size=4),train_dataset=range(3120),val_dataset=range(304))
    return trainer,data

@pytest.mark.parametrize('prefix',['AlphaAlign/verl','NSPO/verl/verl'])
def test_snapshot_before_reward_and_gate_do_not_change_native_math(monkeypatch,tmp_path,prefix):
    gate=gate_module(prefix);monkeypatch.setenv('BASELINE_RUN_DIR',str(tmp_path.resolve()));monkeypatch.setenv('BASELINE_FIRST_BATCH_GATE','1')
    trainer,data=fixture(tmp_path);before={k:v.clone() for k,v in data.batch.items()};rng=torch.get_rng_state().clone()
    ref=gate.snapshot(trainer,data,1);gate.gate(trainer,data,1,ref);gate.actor_result(trainer,1,{'actor/pg_loss':.1,'actor/grad_norm':.2})
    assert all(torch.equal(data.batch[k],v) for k,v in before.items());assert torch.equal(rng,torch.get_rng_state())
    assert (tmp_path/'first_batch/actor_update.json').is_file()

@pytest.mark.parametrize('invalid',['nan','group','count','mask','tampered'])
def test_gate_rejects_incomplete_nonfinite_or_tampered_batch(monkeypatch,tmp_path,invalid):
    gate=gate_module('AlphaAlign/verl');monkeypatch.setenv('BASELINE_RUN_DIR',str(tmp_path.resolve()));monkeypatch.setenv('BASELINE_FIRST_BATCH_GATE','1')
    trainer,data=fixture(tmp_path);ref=gate.snapshot(trainer,data,1)
    if invalid=='nan':data.batch['advantages'][0,0]=float('nan')
    if invalid=='group':data.non_tensor_batch['uid'][0]='changed'
    if invalid=='count':trainer.train_dataset=range(3119)
    if invalid=='mask':data.batch['response_mask'][0,:]=0
    if invalid=='tampered':(tmp_path/'first_batch/step_00000001.original.json').write_text('{}')
    with pytest.raises(ValueError):gate.gate(trainer,data,1,ref)
    assert not (tmp_path/'first_batch/before_actor_update.json').exists()

def test_native_hook_order_is_before_reward_and_actor():
    for prefix in ['AlphaAlign/verl','NSPO/verl/verl']:
        text=(ROOT/prefix/'trainer/ppo/ray_trainer.py').read_text();start=text.index('baseline_original_snapshot =')
        fit=text[start:]
        assert fit.index('baseline_gate(self') < fit.index('self.actor_rollout_wg.update_actor(batch)') < fit.index('baseline_actor_result(self')
        assert 'baseline_first_step = self.global_steps + 1' in text
    assert (ROOT/'AlphaAlign/verl/utils/first_batch_snapshot.py').read_bytes()==(ROOT/'verl/utils/first_batch_snapshot.py').read_bytes()
    assert (ROOT/'NSPO/verl/verl/utils/first_batch_snapshot.py').read_bytes()==(ROOT/'verl/utils/first_batch_snapshot.py').read_bytes()
