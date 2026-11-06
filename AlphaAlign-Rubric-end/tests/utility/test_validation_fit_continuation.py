"""Fault injection through the actual two-iteration fit/validate/save methods.

Only GPU RPC, telemetry and timing wrappers are replaced. GRPO, the reward manager,
validation containment and checkpoint/best control flow execute their real code.
This is CPU integration evidence, never GPU or benchmark accuracy evidence.
"""
import ast
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from enum import Enum
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import time
from types import SimpleNamespace
from typing import Dict, Optional
import uuid

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import (compute_data_metrics, compute_timing_metrics,
    compute_throughout_metrics, reduce_metrics, process_validation_metrics, bootstrap_metric)
from verl.utils.training_evidence import atomic_json, redact_config
from verl.utils import tracking, validation_recovery as vr
from verl.workers.reward_manager.batch import BatchRewardManager

ROOT = Path(__file__).resolve().parents[2]


@contextmanager
def _timer(name, timing_raw):
    start = time.perf_counter()
    yield
    timing_raw[name] = time.perf_counter() - start


def controller():
    path = ROOT / 'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    names = {'fit', '_validate', '_validate_impl', '_rollout_trace_json_value', '_save_checkpoint',
             '_init_best_checkpoint_state', '_get_checkpoint_root', '_load_best_checkpoint_info',
             '_save_best_checkpoint_info', '_pick_best_metric_key', '_maybe_save_best_checkpoint',
             '_save_best_checkpoint_copy'}
    node = ast.ClassDef(name='Trainer', bases=[], keywords=[], decorator_list=[],
                       body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names])
    functions = [n for n in tree.body if
                 (isinstance(n, ast.FunctionDef) and n.name in
                  {'compute_response_mask', 'compute_advantage', '_call_reward_once'})
                 or (isinstance(n, ast.ClassDef) and n.name == 'AdvantageEstimator')]
    env = dict(globals(), pprint=lambda *a, **kw: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions + [node], type_ignores=[])),
                 str(path), 'exec'), env)
    env['RayPPOTrainer'] = env['Trainer']
    return env['Trainer']()


class Loader:
    def __init__(self):
        self.cursor = 29 * 8
    def __len__(self):
        return 390
    def __iter__(self):
        for _ in range(2):
            self.cursor += 8
            yield dict(input_ids=torch.ones(8, 2, dtype=torch.long),
                attention_mask=torch.ones(8, 2, dtype=torch.long),
                position_ids=torch.zeros(8, 2, dtype=torch.long),
                raw_prompt_ids=np.array([[1, 1]] * 8, dtype=object),
                data_source=np.array(['benign'] * 8, dtype=object))
    def state_dict(self):
        return dict(cursor=self.cursor, epoch=1)


class CPUWorker:
    world_size = 4
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(.125))
        self.optimizer = torch.optim.Adam([self.weight], lr=1e-6)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1000, gamma=1.)
        self.updates = []
    def generate_sequences(self, batch):
        n = 1 if batch.meta_info.get('validate') else 8
        count = len(batch) * n
        return DataProto.from_dict(tensors=dict(prompts=batch.batch['input_ids'].repeat_interleave(n, 0),
            responses=torch.ones(count, 2, dtype=torch.long),
            attention_mask=torch.ones(count, 4, dtype=torch.long)))
    def compute_log_prob(self, batch):
        return DataProto.from_dict(tensors=dict(old_log_probs=torch.zeros(len(batch), 2)))
    def update_actor(self, batch):
        self.optimizer.zero_grad()
        features = torch.arange(len(batch), dtype=torch.float32).unsqueeze(1)
        loss = -(self.weight * features * batch.batch['advantages']).mean()
        loss.backward()
        norm = float(self.weight.grad.abs())
        self.optimizer.step(); self.scheduler.step()
        self.updates.append(dict(rewards=batch.batch['token_level_rewards'].clone(),
                                 advantages=batch.batch['advantages'].clone(), weight=self.weight.detach().clone()))
        return DataProto(meta_info=dict(metrics={'actor/grad_norm': [norm], 'actor/lr': [1e-6]}))
    def save_checkpoint(self, local, remote, step, **kw):
        local = Path(local); local.mkdir(parents=True)
        for rank in range(4):
            torch.save(dict(weight=self.weight.detach()), local / f'model_world_size_4_rank_{rank}.pt')
            torch.save(self.optimizer.state_dict(), local / f'optim_world_size_4_rank_{rank}.pt')
            torch.save(dict(lr_scheduler=self.scheduler.state_dict(), rng=dict(cpu=torch.get_rng_state(),
                cuda=torch.get_rng_state(), numpy=np.random.get_state(), random=random.getstate())),
                local / f'extra_state_world_size_4_rank_{rank}.pt')


def run(monkeypatch, root, unavailable):
    for key in ['RUBRIC_PRESENTATION_DAMPING_ENABLE', 'RUBRIC_TARGETED_REFUSAL_ENABLE',
                'RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE', 'RUBRIC_PRESENTATION_FIRST_BATCH_GATE',
                'RUBRIC_TRAIN_BATCH_PREFLIGHT_ONLY', 'RUBRIC_SELECTION_V2_ENABLE',
                'RUBRIC_SELECTION_ORDER_V5_ENABLE', 'RUBRIC_JUDGE_DROP_FAILED']:
        monkeypatch.setenv(key, '0')
    for key, value in dict(RUBRIC_VALIDATION_QUARANTINE_ENABLE='1',
        RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE='1', RUBRIC_RUN_DIR=str(root / 'run'),
        RUBRIC_EVOLVE='1', RUBRIC_EVOLVE_SAVE_DIR=str(root / 'rubric')).items():
        monkeypatch.setenv(key, value)
    (root / 'rubric').mkdir(parents=True)
    (root / 'rubric/runtime_status.json').write_text('{}')
    monkeypatch.setattr(tracking, 'Tracking', lambda **kw: SimpleNamespace(log=lambda **kw: None))
    obj = controller()
    obj.config = OmegaConf.create(dict(trainer=dict(project_name='CPU_test', experiment_name='fault_injection',
        logger=['console'], resume_mode='disable', total_epochs=1, val_before_train=False, balance_batch=False,
        critic_warmup=0, test_freq=30, save_freq=30, default_local_dir=str(root / 'checkpoints'),
        default_hdfs_dir=None, save_best_ckpt=True, best_metric_mode='max', max_actor_ckpt_to_keep=-1),
        algorithm=dict(adv_estimator='grpo', gamma=1., lam=1., use_kl_in_reward=False, kl_penalty='kl'),
        actor_rollout_ref=dict(rollout=dict(n=8, val_kwargs=dict(n=1, do_sample=False))),
        reward_model=dict(enable=False)))
    obj.total_training_steps = 31
    obj._load_checkpoint = lambda: setattr(obj, 'global_steps', 29)
    obj._dump_rollout_traces = lambda *a, **kw: None
    obj._maybe_log_val_generations = lambda **kw: None
    obj._write_light_validation_diagnostics = lambda *a: None
    obj.train_dataloader = Loader()
    obj.actor_rollout_wg = CPUWorker()
    obj.use_reference_policy = obj.use_critic = obj.use_rm = False
    obj.resource_pool_manager = SimpleNamespace(get_n_gpus=lambda: 4)
    obj._init_best_checkpoint_state()
    def reward(batch, return_dict=True):
        values = torch.zeros(len(batch), 2); values[:, -1] = torch.arange(len(batch)).float() / 64
        return dict(reward_tensor=values, reward_extra_info={}, reward_trace_info=[])
    obj.reward_fn = reward
    obj.tokenizer = SimpleNamespace(decode=lambda ids, **kw: 'Original text', eos_token_id=0, pad_token_id=0)
    obj.val_dataloader = [dict(input_ids=torch.ones(2, 2, dtype=torch.long),
        attention_mask=torch.ones(2, 2, dtype=torch.long), position_ids=torch.zeros(2, 2, dtype=torch.long),
        raw_prompt_ids=np.array([[1, 1]] * 2, dtype=object), data_source=np.array(['benign'] * 2, dtype=object))]
    manager = BatchRewardManager(obj.tokenizer, 0, None)
    manager.verify = lambda batch: ([dict(score=.25, judge_parse_valid=True,
        judge_failed=bool(unavailable and obj.global_steps == 30)) for _ in range(len(batch))], [True] * len(batch))
    obj.val_reward_fn = manager
    obj.fit()
    return obj


def test_actual_fit_validation_failure_still_saves30_and_updates31_identically(monkeypatch, tmp_path):
    bad = run(monkeypatch, tmp_path / 'unavailable', True)
    good = run(monkeypatch, tmp_path / 'control', False)
    assert len(bad.actor_rollout_wg.updates) == len(good.actor_rollout_wg.updates) == 2
    for a, b in zip(bad.actor_rollout_wg.updates, good.actor_rollout_wg.updates):
        assert all(torch.equal(a[key], b[key]) for key in ['rewards', 'advantages', 'weight'])
    assert bad.train_dataloader.state_dict() == good.train_dataloader.state_dict()
    for step in [30, 31]:
        p = tmp_path / f'unavailable/checkpoints/global_step_{step}'
        assert (p / 'data.pt').is_file()
        assert len(list((p / 'actor').glob('*.pt'))) == 12
    pending = list((tmp_path / 'unavailable/checkpoints/validation_pending').glob('*.json'))
    assert len(pending) == 1
    assert json.loads(pending[0].read_text())['best_eligible'] is False
    with pytest.raises(ValueError, match='unresolved frozen'):
        vr.require_complete_best(tmp_path / 'unavailable/checkpoints')
    assert json.loads((tmp_path / 'unavailable/run/metrics/step_00000031.json').read_text())['global_step'] == 31


def test_real_runtime_contract_is_explicit_fault_injection_not_model_evidence(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    report = vr.check_runtime_contract(tmp_path)
    assert report['passed'] and report['fault_injection_only']
    assert report['model_calls'] == report['dataset_rows_used'] == report['optimizer_updates'] == 0
    assert report['real_scheduled_validation_recovery_not_yet_exercised']
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '0')
    with pytest.raises(ValueError, match='not enabled'):
        vr.check_runtime_contract(tmp_path / 'disabled')


def test_actual_fit_first_generation_snapshot_precedes_gate_and_actor(monkeypatch,tmp_path):
    # Real DataProto/Torch/fit/reward/GRPO/Adam; GPU RPC and external telemetry
    # remain the same CPU test replacements as this module's existing harness.
    original_controller=controller
    captured=[]
    from verl.utils import first_batch_snapshot as snapshots
    real_capture=snapshots.capture
    def capture(*args,**kwargs):
        ref=real_capture(*args,**kwargs);captured.append(ref);return ref
    monkeypatch.setattr(snapshots,'capture',capture)
    def fresh_controller():
        obj=original_controller();actual_fit=obj.fit
        tree=ast.parse((ROOT/'verl/trainer/ppo/ray_trainer.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_write_training_batch_gate')
        env=dict(globals(),__file__=str(ROOT/'verl/trainer/ppo/ray_trainer.py'))
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_gate','exec'),env)
        actual_fit.__func__.__globals__['_write_training_batch_gate']=env['_write_training_batch_gate']
        def fit():
            obj._load_checkpoint=lambda:setattr(obj,'global_steps',0)
            (tmp_path/'snapshot/run').mkdir(parents=True,exist_ok=True)
            generate=obj.actor_rollout_wg.generate_sequences
            def complete_generate(batch):
                result=generate(batch)
                result.batch['position_ids']=torch.arange(4).repeat(len(result),1)
                return result
            obj.actor_rollout_wg.generate_sequences=complete_generate
            obj.train_dataset=range(3120);obj.val_dataset=range(304)
            monkeypatch.setenv('RUBRIC_FIRST_BATCH_SNAPSHOT_ENABLE','1')
            monkeypatch.setenv('RUBRIC_PRESENTATION_FIRST_BATCH_GATE','1')
            monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED','1')
            old_reward=obj.reward_fn
            def reward(batch,return_dict=True):
                value=old_reward(batch,return_dict)
                value['reward_trace_info']=[dict(judge_parse_valid=True,judge_failed=False) for _ in range(len(batch))]
                return value
            obj.reward_fn=reward
            before=obj.actor_rollout_wg.update_actor
            def update(batch):
                report=json.loads((tmp_path/'snapshot/run/preflight/training_batch_gate.json').read_text())
                assert report['passed'] and report['original_generation_manifest']['response_count']==64
                assert report['original_generation_manifest']['actor_world_size']==4
                assert report['original_generation_snapshot']==captured[0]
                return before(batch)
            obj.actor_rollout_wg.update_actor=update
            return actual_fit()
        obj.fit=fit;return obj
    monkeypatch.setitem(globals(),'controller',fresh_controller)
    obj=run(monkeypatch,tmp_path/'snapshot',False)
    assert len(captured)==1 and len(obj.actor_rollout_wg.updates)==2
    report=json.loads((tmp_path/'snapshot/run/preflight/training_batch_gate.json').read_text())
    manifest=report['original_generation_manifest'];raw=json.loads((tmp_path/'snapshot/run/first_batch'/manifest['snapshot']).read_text())
    assert len(raw['records'])==64 and len(set(row['uid'] for row in raw['records']))==8
    assert raw['tensors']['responses']==[[1,1]]*64
