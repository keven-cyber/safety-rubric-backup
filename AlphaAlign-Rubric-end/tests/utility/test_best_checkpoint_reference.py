import ast
import json
import math
import os
from pathlib import Path
import shutil
from typing import Optional

from omegaconf import OmegaConf
import pytest
import torch

from verl.utils import best_checkpoint_reference as br
from verl.utils import checkpoint_recovery as cr
from verl.utils.training_evidence import atomic_json
from scripts.select_final_checkpoints import select
from test_checkpoint_recovery import original, tree_bytes

ROOT = Path(__file__).resolve().parents[2]


def trainer(root):
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    names = {'_init_best_checkpoint_state', '_get_checkpoint_root', '_load_best_checkpoint_info',
             '_save_best_checkpoint_info', '_pick_best_metric_key', '_maybe_save_best_checkpoint',
             '_save_best_checkpoint_copy'}
    node = ast.ClassDef(name='Trainer', bases=[], keywords=[], decorator_list=[],
                       body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names])
    env = dict(os=os, math=math, json=json, shutil=shutil, Optional=Optional, atomic_json=atomic_json)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), 'real_best_methods', 'exec'), env)
    obj = env['Trainer']()
    obj.config = OmegaConf.create(dict(trainer=dict(default_local_dir=str(root), default_hdfs_dir=None,
        save_best_ckpt=True, best_metric_key='val-core/balanced/constraint_rank_score',
        best_metric_mode='max', max_actor_ckpt_to_keep=-1)))
    obj._init_best_checkpoint_state()
    return obj


def test_real_best_selection_reuses_exact_numbered_bytes_and_retains_old_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE', '1')
    source30 = original(tmp_path, 30)
    before30 = tree_bytes(source30)
    old_best = tree_bytes(tmp_path / 'global_step_best_30')
    old_info = (tmp_path / 'best_checkpoint_info.json').read_bytes()
    source60 = original(tmp_path, 60)
    (tmp_path / 'best_checkpoint_info.json').write_bytes(old_info)
    (tmp_path / 'best_checkpointed_iteration.txt').write_text('30')
    obj = trainer(tmp_path)
    obj._latest_val_step = 60
    obj._latest_val_metrics = {'val-core/balanced/constraint_rank_score': .8}
    monkeypatch.setattr(shutil, 'copytree', lambda *a, **kw: pytest.fail('No duplicate tensor copy'))
    obj._maybe_save_best_checkpoint(60)
    info = json.loads((tmp_path / 'best_checkpoint_info.json').read_text())
    assert info['best_metric_step'] == 60 and info['best_metric_value'] == .8
    assert info['checkpoint_storage'] == br.MODE
    assert br.resolve(tmp_path, info) == source60
    assert tree_bytes(source30) == before30 and tree_bytes(tmp_path / 'global_step_best_30') == old_best
    assert tree_bytes(source60) == tree_bytes(tmp_path / 'global_step_best_60')
    result = select(tmp_path, ('60', 'best'), 60)
    assert result['unique_checkpoint_count'] == 1
    assert result['checkpoints'][0]['checkpoint'] == str(source60)
    assert result['checkpoints'][0]['roles'] == ['60', 'best']
    io = json.loads((tmp_path / 'checkpoint_io/step_00000060.json').read_text())
    assert io['additional_best_tensor_bytes_copied'] == 0 and not io['original_files_modified']


def test_full_recovery_of_referenced_best_preserves_originals_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE', '1')
    source = original(tmp_path / 'old')
    info = json.loads((source.parent / 'best_checkpoint_info.json').read_text())
    info.update(br.reference(source.parent, 30))
    atomic_json(source.parent / 'best_checkpoint_info.json', info)
    before = tree_bytes(source.parent)
    report = cr.prepare(source, tmp_path / 'new', tmp_path)
    assert report['status'] == 'ready' and tree_bytes(source.parent) == before
    assert not (tmp_path / 'new/global_step_best_30').exists()
    assert tree_bytes(tmp_path / 'new/global_step_30') == tree_bytes(source)
    assert json.loads((tmp_path / 'new/best_checkpoint_info.json').read_text()) == info
    restored = trainer(tmp_path / 'new')
    assert restored.best_metric_step == 30 and restored.best_metric_value == .75
    assert cr.copied_context(tmp_path / 'new', 30)['original_best_step'] == 30


@pytest.mark.parametrize('problem', ['missing_optimizer', 'path_escape', 'unknown_storage', 'link'])
def test_reference_cannot_substitute_incomplete_or_different_best(tmp_path, problem):
    source = original(tmp_path)
    info = json.loads((tmp_path / 'best_checkpoint_info.json').read_text())
    info.update(br.reference(tmp_path, 30))
    if problem == 'missing_optimizer':
        (source / 'actor/optim_world_size_4_rank_2.pt').unlink()
    elif problem == 'path_escape':
        info['best_checkpoint_relative_path'] = '../global_step_30'
    elif problem == 'unknown_storage':
        info['checkpoint_storage'] = 'unknown'
    else:
        source.rename(tmp_path / 'actual')
        source.symlink_to(tmp_path / 'actual', target_is_directory=True)
    with pytest.raises(ValueError):
        br.resolve(tmp_path, info)


def test_reference_storage_requires_unpruned_numbered_checkpoints(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE', '1')
    original(tmp_path)
    obj = trainer(tmp_path)
    obj.config.trainer.max_actor_ckpt_to_keep = 2
    with pytest.raises(ValueError, match='retained local'):
        obj._save_best_checkpoint_copy(30, None)


def test_reference_metadata_storage_errors_propagate(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE', '1')
    original(tmp_path)
    obj = trainer(tmp_path)
    obj._best_storage_fields = br.reference(tmp_path, 30)
    def fail(*a, **kw):
        raise OSError('disk failure')
    monkeypatch.setitem(obj._save_best_checkpoint_info.__func__.__globals__, 'atomic_json', fail)
    with pytest.raises(OSError, match='disk failure'):
        obj._save_best_checkpoint_info()


@pytest.mark.parametrize('mode', ['min', 'max'])
def test_real_best_ranking_sequence_is_identical_to_legacy_storage(monkeypatch, tmp_path, mode):
    observed = []
    for reference_enabled in ['0', '1']:
        monkeypatch.setenv('RUBRIC_BEST_CHECKPOINT_REFERENCE_ENABLE', reference_enabled)
        root = tmp_path / reference_enabled
        root.mkdir()
        obj = trainer(root)
        obj.best_metric_mode = mode
        selected = []
        previous_info = previous_tracker = None
        for step, value in zip([30, 60, 90, 120, 150, 180], [.2, .1, .3, float('nan'), .3, .25]):
            original(root, step)
            if previous_info is not None:
                (root / 'best_checkpoint_info.json').write_bytes(previous_info)
                (root / 'best_checkpointed_iteration.txt').write_bytes(previous_tracker)
            obj._latest_val_step = step
            obj._latest_val_metrics = {'val-core/balanced/constraint_rank_score': value}
            obj._maybe_save_best_checkpoint(step)
            selected.append((obj.best_metric_step, obj.best_metric_value))
            previous_info = (root / 'best_checkpoint_info.json').read_bytes()
            previous_tracker = (root / 'best_checkpointed_iteration.txt').read_bytes()
        observed.append(selected)
    assert observed[0] == observed[1]
