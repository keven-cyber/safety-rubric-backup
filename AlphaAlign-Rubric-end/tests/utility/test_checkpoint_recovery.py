import ast
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import random
from types import SimpleNamespace

from omegaconf import OmegaConf
import numpy as np
import pytest
import torch
from verl import DataProto
from verl.utils import checkpoint_recovery as cr
from verl.utils.training_evidence import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def original(root, step=30):
    checkpoint = root / f'global_step_{step}'
    actor = checkpoint / 'actor'
    actor.mkdir(parents=True)
    for rank in range(4):
        torch.save({'weight': torch.tensor([rank + .125])}, actor / f'model_world_size_4_rank_{rank}.pt')
        torch.save({'step': step, 'exp_avg': torch.tensor([rank + .25]), 'lr': 1e-6}, actor / f'optim_world_size_4_rank_{rank}.pt')
        torch.save({'lr_scheduler': {'last_epoch': step}, 'rng': {'cpu': torch.get_rng_state(), 'cuda': torch.get_rng_state(),
                       'numpy': np.random.get_state(), 'random': random.getstate()}}, actor / f'extra_state_world_size_4_rank_{rank}.pt')
    torch.save({'cursor': step * 8, 'epoch': 1}, checkpoint / 'data.pt')
    rubric = checkpoint / 'rubric_evolution'
    rubric.mkdir()
    (rubric / 'runtime_status.json').write_text(json.dumps({'last_step': step}))
    (rubric / 'manifest.json').write_text(json.dumps(dict(schema_version=1, global_step=step, epoch=1,
        files={'runtime_status.json': cr.digest(rubric / 'runtime_status.json')})))
    shutil.copytree(checkpoint, root / f'global_step_best_{step}')
    (root / 'best_checkpoint_info.json').write_text(json.dumps(dict(best_metric_step=step,
        best_metric_key='val-core/balanced/constraint_rank_score', best_metric_mode='max', best_metric_value=.75)))
    (root / 'best_checkpointed_iteration.txt').write_text(str(step))
    (root / 'latest_checkpointed_iteration.txt').write_text(str(step))
    return checkpoint


def tree_bytes(root):
    return {str(f.relative_to(root)): f.read_bytes() for f in root.rglob('*') if f.is_file()}


def prepared(tmp_path):
    source = original(tmp_path / 'old')
    before = tree_bytes(source.parent)
    report = cr.prepare(source, tmp_path / 'new', tmp_path)
    assert tree_bytes(source.parent) == before
    return source, report


def test_copy_keeps_actual_tensor_optimizer_rng_data_and_best_bytes(tmp_path):
    source, report = prepared(tmp_path)
    dest = tmp_path / 'new'
    assert report['status'] == 'ready' and report['restored_step'] == 30
    assert report['source_and_copy_stream_sha256_equal'] and not report['links_used']
    assert (dest / 'latest_checkpointed_iteration.txt').read_text().strip() == '30'
    for name, data in tree_bytes(source).items():
        copied = dest / source.name / name
        assert copied.read_bytes() == data and not copied.is_symlink()
        assert report['files'][str(Path(source.name) / name)]['sha256'] == hashlib.sha256(data).hexdigest()
    state = torch.load(dest / 'global_step_30/actor/optim_world_size_4_rank_0.pt', weights_only=False)
    extra = torch.load(dest / 'global_step_30/actor/extra_state_world_size_4_rank_0.pt', weights_only=False)
    assert state['step'] == extra['lr_scheduler']['last_epoch'] == 30 and state['lr'] == 1e-6
    assert torch.equal(extra['rng']['cpu'], torch.get_rng_state())
    assert (dest / 'best_checkpoint_info.json').read_bytes() == (source.parent / 'best_checkpoint_info.json').read_bytes()


@pytest.mark.parametrize('bad', ['optimizer', 'rng', 'data', 'rubric', 'best', 'latest', 'pending', 'missing_rng'])
def test_incomplete_or_misaligned_source_cannot_fall_back_to_fresh(tmp_path, bad):
    source = original(tmp_path / 'old')
    if bad == 'optimizer': (source / 'actor/optim_world_size_4_rank_3.pt').unlink()
    if bad == 'rng': (source / 'actor/extra_state_world_size_4_rank_2.pt').write_bytes(b'')
    if bad == 'missing_rng': torch.save({'lr_scheduler': {'last_epoch': 30}}, source / 'actor/extra_state_world_size_4_rank_2.pt')
    if bad == 'data': (source / 'data.pt').unlink()
    if bad == 'rubric': (source / 'rubric_evolution/runtime_status.json').write_text('{}')
    if bad == 'best': (source.parent / 'best_checkpoint_info.json').write_text('{"best_metric_step":60,"best_metric_value":1}')
    if bad == 'latest': (source.parent / 'latest_checkpointed_iteration.txt').write_text('60')
    if bad == 'pending':
        p = source.parent / 'validation_pending'; p.mkdir(); (p / 'step30.json').write_text('{}')
    before = tree_bytes(source.parent)
    with pytest.raises((ValueError, FileNotFoundError)):
        cr.prepare(source, tmp_path / 'new', tmp_path)
    assert tree_bytes(source.parent) == before and not (tmp_path / 'new').exists()


def test_existing_destination_and_path_escape_are_rejected(tmp_path):
    source = original(tmp_path / 'old')
    (tmp_path / 'new').mkdir()
    for destination in [tmp_path / 'new', tmp_path / 'old/new', tmp_path / '../outside']:
        with pytest.raises(ValueError): cr.prepare(source, destination, tmp_path)


def test_copied_source_must_match_loaded_step_and_not_change_after_copy(tmp_path):
    source, _ = prepared(tmp_path)
    with pytest.raises(ValueError): cr.copied_context(tmp_path / 'new', 60)
    context = cr.copied_context(tmp_path / 'new', 30)
    assert context['status'] == 'copied'
    assert not cr.verified_context(context)  # copy description is not actual restoration
    (tmp_path / 'new/global_step_30/actor/model_world_size_4_rank_0.pt').write_bytes(b'tampered')
    with pytest.raises(ValueError): cr.copied_context(tmp_path / 'new', 30)


def trainer_methods(names):
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(os=os, json=json, torch=torch, atomic_json=atomic_json,
               find_latest_ckpt_path=lambda root: str(Path(root) / ('global_step_' + (Path(root) / 'latest_checkpointed_iteration.txt').read_text().strip())))
    node = ast.ClassDef(name='Trainer', bases=[], keywords=[], body=methods, decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'real_recovery_trainer', 'exec'), env)
    return env['Trainer']


@pytest.mark.parametrize('total_steps', [1170, 30])
def test_actual_trainer_loads_full_state_before_emitting_restored_context(monkeypatch, tmp_path, total_steps):
    source, _ = prepared(tmp_path)
    monkeypatch.setenv('RUBRIC_CHECKPOINT_RECOVERY_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(tmp_path / 'new_run/rubric'))
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'new_run'))
    obj = trainer_methods({'_load_checkpoint','_init_best_checkpoint_state','_get_checkpoint_root','_load_best_checkpoint_info'})()
    obj.config = OmegaConf.create(dict(trainer=dict(resume_mode='auto', default_hdfs_dir=None,
        default_local_dir=str(tmp_path / 'new'), del_local_ckpt_after_load=False, save_best_ckpt=True,
        best_metric_key='val-core/balanced/constraint_rank_score', best_metric_mode='max')))
    obj._init_best_checkpoint_state()
    obj.total_training_steps = total_steps; obj.use_critic = False
    loaded = {}
    def load(path, del_local_after_load):
        assert not del_local_after_load
        loaded['model'] = torch.load(Path(path) / 'model_world_size_4_rank_0.pt', weights_only=False)
        loaded['optimizer'] = torch.load(Path(path) / 'optim_world_size_4_rank_0.pt', weights_only=False)
        loaded['extra'] = torch.load(Path(path) / 'extra_state_world_size_4_rank_0.pt', weights_only=False)
    obj.actor_rollout_wg = SimpleNamespace(load_checkpoint=load)
    obj.train_dataloader = SimpleNamespace(load_state_dict=lambda state: loaded.update(data=state))
    obj._load_checkpoint()
    assert obj.global_steps == 30 and loaded['data']['cursor'] == 240
    assert loaded['optimizer']['step'] == loaded['extra']['lr_scheduler']['last_epoch'] == 30
    assert obj._restored_rubric_epoch == 1 and cr.verified_context(obj._checkpoint_recovery)
    report = json.loads((tmp_path / 'new_run/preflight/checkpoint_restore.json').read_text())
    assert report['status'] == 'restored' and report['restored_step'] == 30
    assert json.loads((tmp_path / 'new_run/rubric/runtime_status.json').read_text()) == {'last_step': 30}


def test_resumed_first_batch_gate_is31_and_fresh_gate_stays1(monkeypatch, tmp_path):
    source, _ = prepared(tmp_path)
    context = cr.copied_context(tmp_path / 'new', 30)
    context.update(status='restored', actor_optimizer_scheduler_rng_load_returned=True,
                   data_loader_loaded=True, rubric_state_loaded=True, best_selection_loaded=True)
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_training_batch_gate')
    env = dict(os=os, np=np, torch=torch, atomic_json=atomic_json, __file__=str(ROOT / 'verl/trainer/ppo/ray_trainer.py'))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'real_batch_gate', 'exec'), env)
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '1')
    batch = DataProto.from_dict(tensors={k: torch.zeros(64, 1) for k in ('token_level_scores', 'token_level_rewards', 'advantages')},
        non_tensors=dict(uid=np.array([str(i // 8) for i in range(64)], dtype=object),
                         presentation_advantage_contribution=np.zeros(64)))
    traces = [dict(judge_parse_valid=True, judge_failed=False)] * 64
    report = env['_write_training_batch_gate'](batch, traces, 31, dict(train=3120, val=304), checkpoint_recovery=context)
    assert report['passed'] and report['global_step'] == 31 and report['optimizer_updates'] == 0
    assert report['restored_global_step'] == 30 and report['policy_origin'] == 'restored_checkpoint'
    with pytest.raises(ValueError): env['_write_training_batch_gate'](batch, traces, 31, dict(train=3120, val=304))
    assert env['_write_training_batch_gate'](batch, traces, 1, dict(train=3120, val=304))['passed']


def test_explicit_recovery_flag_cannot_be_ignored_by_disabled_resume(monkeypatch):
    monkeypatch.setenv('RUBRIC_CHECKPOINT_RECOVERY_ENABLE', '1')
    obj = trainer_methods({'_load_checkpoint'})()
    obj.config = OmegaConf.create(dict(trainer=dict(resume_mode='disable')))
    with pytest.raises(ValueError, match='silently start fresh'):
        obj._load_checkpoint()


@pytest.mark.parametrize('recover', [False, True])
def test_actual_fast_launcher_default_fresh_or_explicit_verified_resume(tmp_path, recover):
    import subprocess
    source = original(tmp_path / 'persistent/old')
    before = tree_bytes(source.parent)
    bindir = tmp_path / 'bin'; bindir.mkdir()
    capture = tmp_path / 'capture.json'
    child = bindir / 'bash'
    child.write_text('''#!/usr/bin/env python3
import os,json
from pathlib import Path
p=Path(os.environ['CKPT_PATH'])
Path(os.environ['CAPTURE']).write_text(json.dumps(dict(mode=os.environ['TRAIN_RESUME_MODE'],
 enabled=os.environ['RUBRIC_CHECKPOINT_RECOVERY_ENABLE'],checkpoint=str(p),
 manifest=(p/'checkpoint_recovery.json').is_file(),profile=os.environ['RUBRIC_JUDGE_SERVING_PROFILE'],
 api=os.environ['RUBRIC_JUDGE_API_CONCURRENCY'])))
''')
    child.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir)+os.pathsep+os.environ['PATH'],
               PRIMUS_SAVE_CHECKPOINT_DIR=str(tmp_path / 'persistent'),
               PRIMUS_JOB_SUBMIT_TIMESTAMP='same_platform_timestamp', CAPTURE=str(capture))
    env.pop('RUBRIC_RECOVERY_SOURCE_CHECKPOINT', None)
    if recover: env['RUBRIC_RECOVERY_SOURCE_CHECKPOINT'] = str(source)
    result = subprocess.run(['/bin/bash', str(ROOT / 'scripts/train_rubricevolve_presentation_v8_fast.sh'), 'g1'],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    obj = json.loads(capture.read_text())
    assert obj['mode'] == ('auto' if recover else 'disable')
    assert obj['enabled'] == ('1' if recover else '0') and obj['manifest'] == recover
    assert obj['profile'] == 'batched_v8' and obj['api'] == '48'
    assert tree_bytes(source.parent) == before
    if recover: assert cr.copied_context(obj['checkpoint'], 30)['restored_step'] == 30


def test_explicit_gap_preserving_recovery_copies_frozen_policy_generation_and_unknown_marker(tmp_path, monkeypatch):
    source = original(tmp_path / 'old', 60)
    frozen = original(tmp_path / 'temporary', 30)
    shutil.copytree(frozen, source.parent / 'global_step_30')
    generation = tmp_path / 'run/validation_pending/generations/step30.json'
    generation.parent.mkdir(parents=True)
    generation.write_text(json.dumps(dict(global_step=30, batch_index=1,
        snapshot=dict(inputs=['q'], outputs=['answer'], tensors={'responses': [[1]]}, non_tensor_batch={'index': [1]}))))
    pending = source.parent / 'validation_pending/step30.json'
    pending.parent.mkdir()
    pending.write_text(json.dumps(dict(global_step=30, batch_index=1, status='incomplete', aggregate_reward=None,
        best_eligible=False, final_selection_blocked=True, checkpoint=str(source.parent/'global_step_30'),
        generated_batch=str(generation))))
    before = tree_bytes(source.parent)
    monkeypatch.setenv('RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING', '1')
    report = cr.prepare(source, tmp_path / 'new', tmp_path)
    assert tree_bytes(source.parent) == before
    assert report['preserved_validation_gaps'][0]['unresolved']
    assert (tmp_path/'new/validation_pending/step30.json').read_bytes() == pending.read_bytes()
    assert (tmp_path/'new/validation_pending/preserved_generations/step30.json').read_bytes() == generation.read_bytes()
    assert (tmp_path/'new/global_step_30/data.pt').read_bytes() == frozen.joinpath('data.pt').read_bytes()
    from verl.utils.validation_recovery import require_complete_best
    with pytest.raises(ValueError, match='incomplete'):
        require_complete_best(tmp_path/'new')
    cr.copied_context(tmp_path/'new',60)
    (tmp_path/'new/validation_pending/preserved_generations/step30.json').write_text('{}')
    with pytest.raises(ValueError, match='changed'):
        cr.copied_context(tmp_path/'new',60)


def test_second_and_third_hop_preserve_original_gap_bytes_and_policies(monkeypatch,tmp_path):
    source=original(tmp_path/'original',30)
    run=tmp_path/'run';run.mkdir();g=run/'validation_pending/generations/event.json';g.parent.mkdir(parents=True)
    g.write_text(json.dumps(dict(global_step=30,batch_index=1,snapshot=dict(inputs=['p'],outputs=['a'],tensors={},non_tensor_batch={}))))
    pending=source.parent/'validation_pending/event.json';pending.parent.mkdir()
    pending.write_text(json.dumps(dict(global_step=30,batch_index=1,status='incomplete',aggregate_reward=None,best_eligible=False,
        final_selection_blocked=True,checkpoint=str(source),generated_batch=str(g))))
    monkeypatch.setenv('RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING','1')
    previous=source
    for i in range(3):
        dest=tmp_path/f'copy{i}';report=cr.prepare(previous,dest,tmp_path)
        assert (dest/'validation_pending/event.json').read_bytes()==pending.read_bytes()
        assert report['preserved_validation_gaps'][0]['original_pending']==str(pending)
        assert (dest/'global_step_30/data.pt').read_bytes()==(source/'data.pt').read_bytes()
        previous=dest/'global_step_30'


def test_protocol_transition_history_cannot_disappear_on_resume(monkeypatch,tmp_path):
    source=original(tmp_path/'original',30);marker=source.parent/'scoring_protocol_transition.json'
    marker.write_text(json.dumps(dict(status='requires_same_protocol_review',final_best_blocked=True)))
    with pytest.raises(ValueError,match='Scoring protocol review'):cr.prepare(source,tmp_path/'refuse',tmp_path)
    monkeypatch.setenv('RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING','1')
    dest=tmp_path/'next';report=cr.prepare(source,dest,tmp_path)
    history=list((dest/'protocol_history').glob('*.json'));assert len(history)==1 and history[0].read_bytes()==marker.read_bytes()
    from verl.utils.validation_recovery import require_complete_best
    with pytest.raises(ValueError,match='same-protocol'):require_complete_best(dest)
