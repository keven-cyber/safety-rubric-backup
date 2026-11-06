"""CPU contracts using actual trainer validation + actual reward manager.

Generation and distributed checkpoint loading are intercepted, not GPU proof.
"""
import importlib.util
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
from verl.utils import frozen_validation_review as review
from verl.utils.validation_recovery import JudgeScoringUnavailable, require_complete_best, write, check_runtime_contract, score_batch
from verl.workers.reward_manager.batch import BatchRewardManager
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.utils.training_evidence import atomic_json
from scripts.select_final_checkpoints import select

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('review_actual_validation_reference', Path(__file__).with_name('test_validation_recovery.py'))
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
reference._env.update(process_validation_metrics=process_validation_metrics, atomic_json=atomic_json)


def checkpoint(root, step, content=None):
    path = root / f'global_step_{step}'
    (path / 'actor').mkdir(parents=True)
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(4):
            (path / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt').write_bytes(content or f'{step}-{kind}-{rank}'.encode())
    (path / 'data.pt').write_bytes(b'preserved fake data state, not a real GPU checkpoint')
    return path


def test_actual_inventory_traces_lineage_deduplicates_exact_copies_and_retains_distinct_policies(tmp_path):
    ancestor, current = tmp_path / 'old', tmp_path / 'current'
    ancestor.mkdir(); current.mkdir()
    for step in range(30, 1171, 30):
        checkpoint(ancestor if step <= 150 else current, step)
    checkpoint(current, 150)
    changed = checkpoint(current, 120, b'distinct actual model bytes')
    write(current / 'checkpoint_recovery.json', dict(status='ready', destination=str(current), source_checkpoint=str(ancestor / 'global_step_150')))
    marker = ancestor / 'scoring_protocol_transition.json'
    write(marker, dict(status='requires_same_protocol_review', final_best_blocked=True))
    before = marker.read_bytes()
    plan = review.inventory(current, tmp_path)
    assert len(plan['policies']) == 40
    assert len(next(p for p in plan['policies'] if p['step'] == 150)['aliases']) == 2
    assert len([p for p in plan['policies'] if p['step'] == 120]) == 2
    assert any(p['checkpoint'] == str(changed) for p in plan['policies'])
    assert marker.read_bytes() == before
    (current / 'global_step_1170/actor/model_world_size_4_rank_3.pt').unlink()
    with pytest.raises(ValueError, match='Incomplete frozen actor'):
        review.inventory(current, tmp_path)


def test_missing_old_step_cannot_be_silently_excluded(tmp_path):
    for step in range(60, 1171, 30):
        checkpoint(tmp_path, step)
    with pytest.raises(ValueError, match='coverage is incomplete'):
        review.inventory(tmp_path, tmp_path)


def make_trainer(tmp_path, monkeypatch, unavailable=False, error=None):
    source = tmp_path / 'source'
    source.mkdir()
    for step in range(30,1171,30):
        checkpoint(source, step)
    obj = reference.trainer(tmp_path)
    Path(obj.config.trainer.default_local_dir).mkdir()
    obj.val_dataset = range(304)
    obj.config.trainer.del_local_ckpt_after_load = False
    obj.config.trainer.log_val_generations = 10
    obj.config.data = dict(val_files=[str(tmp_path / 'validation.parquet')])
    (tmp_path / 'validation.parquet').write_bytes(b'metadata-only fixture, not production data')
    obj.use_critic = False
    data = obj.val_dataloader[0]
    data['data_source'] = np.array(['ordinary_harm']*76 + ['adaptive_harm']*76 + ['benign']*76 + ['sensitive_benign']*76, dtype=object)
    obj.tokenizer.decode = lambda ids, **kw: 'decoded text ' + str(int(ids[0]))
    obj.actor_rollout_wg.generate_sequences = lambda batch: DataProto.from_dict(tensors=dict(
        prompts=torch.ones(304, 2, dtype=torch.long), responses=torch.ones(304, 2, dtype=torch.long),
        input_ids=torch.ones(304, 4, dtype=torch.long), attention_mask=torch.ones(304, 4, dtype=torch.long),
        position_ids=torch.zeros(304, 4, dtype=torch.long)))
    obj._maybe_log_val_generations = lambda **kw: None
    obj._write_light_validation_diagnostics = lambda *args: None
    loads = []
    obj.actor_rollout_wg.load_checkpoint = lambda path, del_local_after_load: loads.append((path, del_local_after_load))
    obj.actor_rollout_wg.update_actor = lambda *args: pytest.fail('Frozen review must never update actor')
    manager = BatchRewardManager(obj.tokenizer, 0, None)

    def verify(batch):
        # All original tokens must already be on disk before ANY Judge score.
        files = list((Path(os.environ['RUBRIC_RUN_DIR']) / 'validation_pending/generations').glob('*.json'))
        assert len(files) == 1 and len(json.loads(files[0].read_text())['snapshot']['tensors']['responses']) == 304
        # Intercepted Judge transport: explicitly synthetic CPU request evidence.
        trace = Path(os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'])
        trace.mkdir()
        (trace / 'cpu_fixture.jsonl').write_text('\n'.join(json.dumps(dict(
            fixture_only=True, event='base_judge_attempt', error_type=None,
            request_audit=dict(response_id=f'cpu-fixture-{i}'), raw_output='synthetic-test-only')) for i in range(304))+'\n')
        if error:
            raise error
        if unavailable and obj.global_steps == 900:
            raise JudgeScoringUnavailable('typed unavailable; no deterministic rerun')
        return [dict(score=.25, judge_parse_valid=True, judge_failed=False,
            balanced_safety=1., balanced_critical_min=1., balanced_boundary_or_over_refusal=0.,
            balanced_critical_gate_applied=0., balanced_useful=1.) for _ in range(304)], [True]*304

    manager.verify = verify
    obj.val_reward_fn = manager
    for k,v in dict(RUBRIC_FROZEN_REVIEW_ENABLE='1',RUBRIC_FROZEN_REVIEW_AUTHORIZATION=review.AUTHORIZATION,
        RUBRIC_VALIDATION_QUARANTINE_ENABLE='1', RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE='1',
        RUBRIC_FROZEN_REVIEW_SOURCE_ROOT=str(source),PRIMUS_SAVE_CHECKPOINT_DIR=str(tmp_path),
        RUBRIC_RUN_DIR=str(tmp_path/'run'), REWARD_MODE='rubric_safe_helpful_balanced',
        RUBRIC_SELECTION_CONSTRAINT_ENABLE='1', RUBRIC_SELECTION_V2_ENABLE='0').items():
        monkeypatch.setenv(k,v)
    return obj, loads


def test_actual_validation_manager_archives_before_score_and_final_selection_is_complete_new_review(tmp_path, monkeypatch):
    obj, loads = make_trainer(tmp_path, monkeypatch)
    review.run(obj)
    assert len(loads) == 39 and all(delete is False for _,delete in loads)
    root = Path(obj.config.trainer.default_local_dir) / 'new_reevaluation/selected_checkpoints'
    report = review.verify_completion(root)
    assert report['best_metric']['best_metric_step'] == 30  # chronological strict-improvement tie
    assert report['optimizer_updates'] == 0 and report['label'] == review.LABEL
    result = select(root, ('900','990','1080','last','best'),1170)
    assert len(result['checkpoints']) == 5
    assert result['original_validation_gaps_remain_unknown']
    original = Path(report['results'][0]['path'])
    raw = original.read_bytes()
    original.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='changed'):
        review.verify_completion(root)


def test_contained_unknown_keeps_all304_original_and_other_policy_reviews_but_blocks_final_best(tmp_path, monkeypatch):
    obj, loads = make_trainer(tmp_path, monkeypatch, unavailable=True)
    with pytest.raises(ValueError, match='Incomplete/nonfinite'):
        review.run(obj)
    root = Path(obj.config.trainer.default_local_dir)
    assert len(loads) == 39  # independent policies continue; no retry of failed900
    assert not (root / 'new_reevaluation/selected_checkpoints').exists()
    event = json.loads(next((root / 'validation_pending').glob('*.json')).read_text())
    assert event['checkpoint'].endswith('source/global_step_900')
    assert event['aggregate_reward'] is None and event['best_eligible'] is False
    assert len(json.loads(Path(event['generated_batch']).read_text())['snapshot']['outputs']) == 304
    with pytest.raises(ValueError, match='unresolved'):
        require_complete_best(root)


@pytest.mark.parametrize('error',[ValueError('bad shape'),OSError('storage'),RuntimeError('unknown failure')])
def test_program_storage_unknown_failure_propagates_without_completed_review(tmp_path, monkeypatch, error):
    obj, loads = make_trainer(tmp_path, monkeypatch, error=error)
    with pytest.raises(type(error), match=str(error)):
        review.run(obj)
    assert len(loads) == 1 and not list(Path(obj.config.trainer.default_local_dir).rglob('new_reevaluation_completed.json'))


def test_ordinary_guard_still_blocks_original_protocol_history(tmp_path):
    write(tmp_path / 'scoring_protocol_transition.json', dict(final_best_blocked=True))
    with pytest.raises(ValueError, match='chronological'):
        require_complete_best(tmp_path)


def test_mandatory_real_manager_contract_passes_before_frozen_policy_load_in_review_mode(tmp_path, monkeypatch):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE','1')
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_ENABLE','1')
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_AUTHORIZATION',review.AUTHORIZATION)
    monkeypatch.delenv('RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT',raising=False)
    report=check_runtime_contract(tmp_path)
    assert report['passed'] and report['model_calls']==report['optimizer_updates']==report['dataset_rows_used']==0
    assert report['incomplete_best_rejected'] and report['unavailable_result_contained']
    assert os.environ['RUBRIC_FROZEN_REVIEW_ENABLE']=='1'


def test_real_review_snapshot_still_requires_frozen_policy_and_cannot_impersonate_diagnostic(tmp_path, monkeypatch):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE','1')
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_ENABLE','1')
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_AUTHORIZATION',review.AUTHORIZATION)
    monkeypatch.delenv('RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT',raising=False)
    called=[]
    snapshot=dict(inputs=['original'],outputs=['original'],tensors={'responses':[[1]]},non_tensor_batch={})
    with pytest.raises(KeyError,match='RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT'):
        score_batch(lambda:called.append(1),snapshot,tmp_path/'run',tmp_path/'root',30,1)
    with pytest.raises(ValueError,match='exact zero-model-call'):
        score_batch(lambda:called.append(1),snapshot,tmp_path/'run',tmp_path/'root',30,1,diagnostic_only=True)
    assert not called


@pytest.mark.parametrize('arm',['g1','g3'])
def test_actual_full_launcher_chain_copies_complete1170_and_enters_only_authorized_review(tmp_path, arm):
    spec = importlib.util.spec_from_file_location('review_checkpoint_fixture',Path(__file__).with_name('test_checkpoint_recovery.py'))
    fixture = importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    for step in (900,990,1080,1170):
        source = fixture.original(tmp_path/'original',step)
    data = tmp_path/'data';data.mkdir()
    write(data/'manifest.json',dict(version='updated_training_rubric_dataset_independent_v4_targeted_hard_negative',training_rows=3120,validation_rows=304))
    for split in ('train','val'):
        (data/split).mkdir()
        for name in ('benign_sampled.parquet','sensitive_sampled.parquet','harm_sampled.parquet','adaptive_harm.parquet'):
            (data/split/name).write_bytes(b'metadata-only CPU launcher fixture')
    auth=tmp_path/'auth.json';auth.write_text('{}')
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    stub=bin_dir/'bash'
    stub.write_text('''#!/usr/bin/python3
import os,sys,json
from pathlib import Path
if len(sys.argv)>1 and Path(sys.argv[1]).name=='train_val.sh':
 r=json.loads((Path(os.environ['CKPT_PATH'])/'checkpoint_recovery.json').read_text())
 assert r['status']=='ready' and r['restored_step']==1170 and r['source_files_modified'] is False
 keys=['RUBRIC_FROZEN_REVIEW_ENABLE','RUBRIC_FROZEN_REVIEW_AUTHORIZATION','RUBRIC_EVAL_ONLY',
 'TRAIN_RESUME_MODE','RUBRIC_CHECKPOINT_RECOVERY_ENABLE','RUBRIC_VALIDATION_QUARANTINE_ENABLE',
 'RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR','RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE',
 'FINAL_SAFETY_EVAL_STEPS','FINAL_EVAL_EXPECTED_LAST_STEP','RUBRIC_SELECTION_V2_ENABLE']
 Path(os.environ['TEST_CAPTURE']).write_text(json.dumps({k:os.environ[k] for k in keys}));sys.exit(0)
os.execv('/bin/bash',['/bin/bash']+sys.argv[1:])
''')
    stub.chmod(0o755)
    env=dict(os.environ, PATH=str(bin_dir)+os.pathsep+os.environ['PATH'],
        FILE_PATH=str(data), PRIMUS_TRAIN_DATASET_DIR=str(data), PRIMUS_SAVE_CHECKPOINT_DIR=str(tmp_path),
        PRIMUS_OSS_WORKSPACE_DIR=str(tmp_path),PRIMUS_JOB_SUBMIT_TIMESTAMP='cpu-launcher-fixture',
        RUBRIC_FROZEN_REVIEW_SOURCE_ROOT=str(source.parent),RUBRIC_RECOVERY_SOURCE_CHECKPOINT=str(source),
        RUBRIC_EVOLVE_API_AUTH_FILE=str(auth),TEST_CAPTURE=str(tmp_path/'captured.json'))
    p=subprocess.run(['/bin/bash',str(ROOT/'scripts/reevaluate_frozen_validation_v8.sh'),arm],env=env,capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stdout+p.stderr
    actual=json.loads((tmp_path/'captured.json').read_text())
    assert actual==dict(RUBRIC_FROZEN_REVIEW_ENABLE='1',RUBRIC_FROZEN_REVIEW_AUTHORIZATION=review.AUTHORIZATION,
        RUBRIC_EVAL_ONLY='0',TRAIN_RESUME_MODE='auto',RUBRIC_CHECKPOINT_RECOVERY_ENABLE='1',
        RUBRIC_VALIDATION_QUARANTINE_ENABLE='1',RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR='0' if arm=='g1' else '0.25',
        RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='0' if arm=='g1' else '1',FINAL_SAFETY_EVAL_STEPS='900,990,1080,last,best',
        FINAL_EVAL_EXPECTED_LAST_STEP='1170',RUBRIC_SELECTION_V2_ENABLE='0')


def set_designated_scope(obj, monkeypatch, best=900, distinct=False):
    import shutil
    source=Path(os.environ['RUBRIC_FROZEN_REVIEW_SOURCE_ROOT'])
    shutil.copytree(source/f'global_step_{best}',source/f'global_step_best_{best}')
    if distinct:
        (source/f'global_step_best_{best}/actor/model_world_size_4_rank_0.pt').write_bytes(b'distinct preserved original best model')
    write(source/'best_checkpoint_info.json',dict(best_metric_step=best,best_metric_key=review.KEY,best_metric_mode='max',best_metric_value=-1.))
    (source/'best_checkpointed_iteration.txt').write_text(str(best))
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_SCOPE','final_checkpoints')
    return source


def test_designated_review_hashes_only_requested_late_policies_and_original_best(tmp_path,monkeypatch):
    obj,loads=make_trainer(tmp_path,monkeypatch)
    source=set_designated_scope(obj,monkeypatch)
    # An irrelevant early partial checkpoint must neither be hashed nor evaluated.
    (source/'global_step_30/actor/model_world_size_4_rank_3.pt').unlink()
    plan=review.inventory(source,tmp_path,scope='final_checkpoints')
    review.validate_scope(plan)
    assert [p['step'] for p in plan['policies']]==[900,990,1080,1170]
    assert next(p for p in plan['policies'] if 'best' in p['roles'])['step']==900
    with pytest.raises(ValueError,match='Unknown frozen review scope'):
        review.inventory(source,tmp_path,scope='skip_unknown')
    review.run(obj)
    assert len(loads)==4 and all(not delete for _,delete in loads)
    selected=Path(obj.config.trainer.default_local_dir)/'new_reevaluation/selected_checkpoints'
    report=review.verify_completion(selected)
    assert report['best_metric']['best_metric_step']==900
    assert report['full_history_best_reselected'] is False
    assert report['best_metric']['selection_basis']=='original_training_best_and_designated_final_checkpoints'
    actual=select(selected,('900','990','1080','last','best'),1170)
    assert actual['unique_checkpoint_count']==4 and not actual['full_history_best_reselected']
    assert next(p for p in actual['checkpoints'] if p['step']==900)['roles']==['900','best']
    report['full_history_best_reselected']=True
    write(selected/'new_reevaluation_completed.json',report)
    with pytest.raises(ValueError,match='cannot claim a full-history best'):
        review.verify_completion(selected)


@pytest.mark.parametrize('best',[900,1170])
def test_designated_review_keeps_distinct_same_step_best_and_standard_policy(tmp_path,monkeypatch,best):
    obj,loads=make_trainer(tmp_path,monkeypatch)
    source=set_designated_scope(obj,monkeypatch,best=best,distinct=True)
    review.run(obj)
    assert len(loads)==5
    selected=Path(obj.config.trainer.default_local_dir)/'new_reevaluation/selected_checkpoints'
    report=review.verify_completion(selected)
    assert f'global_step_{best}' in report['selected_records'] and f'global_step_best_{best}' in report['selected_records']
    assert report['selected_records'][f'global_step_{best}']!=report['selected_records'][f'global_step_best_{best}']
    del report['selected_records'][f'global_step_{best}']
    write(selected/'new_reevaluation_completed.json',report)
    with pytest.raises(ValueError,match='omitted or changed a designated role'):
        review.verify_completion(selected)


def test_designated_review_does_not_replace_original_best_with_late_subset_winner(tmp_path,monkeypatch):
    obj,loads=make_trainer(tmp_path,monkeypatch)
    set_designated_scope(obj,monkeypatch,best=1080)
    review.run(obj)
    selected=Path(obj.config.trainer.default_local_dir)/'new_reevaluation/selected_checkpoints'
    report=review.verify_completion(selected)
    assert len(loads)==4 and report['best_metric']['best_metric_step']==1080
    assert review.best_record([review.read(x['path']) for x in report['results']])['step']==900


def test_designated_review_still_blocks_whole_unknown_and_preserves_old_markers(tmp_path,monkeypatch):
    obj,loads=make_trainer(tmp_path,monkeypatch,unavailable=True)
    source=set_designated_scope(obj,monkeypatch)
    marker=source/'scoring_protocol_transition.json';write(marker,dict(original=True,unknown=True));before=marker.read_bytes()
    with pytest.raises(ValueError,match='Incomplete/nonfinite'):
        review.run(obj)
    assert len(loads)==4 and marker.read_bytes()==before
    assert not list(Path(obj.config.trainer.default_local_dir).rglob('new_reevaluation_completed.json'))


def test_designated_inventory_requires_original_best_and_all_fixed_final_steps(tmp_path,monkeypatch):
    obj,loads=make_trainer(tmp_path,monkeypatch)
    source=set_designated_scope(obj,monkeypatch)
    (source/'global_step_990/actor/model_world_size_4_rank_1.pt').unlink()
    with pytest.raises(ValueError,match='Incomplete frozen actor'):
        review.inventory(source,tmp_path,scope='final_checkpoints')
    (source/'best_checkpointed_iteration.txt').write_text('1080')
    with pytest.raises(ValueError,match='tracker disagrees'):
        review.inventory(source,tmp_path,scope='final_checkpoints')


def actual_numbered_best_fixture(tmp_path,monkeypatch,best):
    import shutil
    obj,loads=make_trainer(tmp_path,monkeypatch)
    source=Path(os.environ['RUBRIC_FROZEN_REVIEW_SOURCE_ROOT'])
    spec=importlib.util.spec_from_file_location('bestref_complete_fixture',Path(__file__).with_name('test_checkpoint_recovery.py'))
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    for step in (900,990,1080,1170):
        shutil.rmtree(source/f'global_step_{step}')
        fixture.original(source,step)
        shutil.rmtree(source/f'global_step_best_{step}')
    info=dict(best_metric_step=best,best_metric_key=review.KEY,best_metric_mode='max',best_metric_value=-1.,
              checkpoint_storage='numbered_checkpoint_reference_v1',best_checkpoint_relative_path=f'global_step_{best}')
    write(source/'best_checkpoint_info.json',info)
    (source/'best_checkpointed_iteration.txt').write_text(str(best))
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_SCOPE','final_checkpoints')
    return obj,loads,source


@pytest.mark.parametrize('best',[900,1080])
def test_designated_review_uses_actual_numbered_best_reference_without_legacy_alias(tmp_path,monkeypatch,best):
    obj,loads,source=actual_numbered_best_fixture(tmp_path,monkeypatch,best)
    assert not (source/f'global_step_best_{best}').exists()
    before=(source/'best_checkpoint_info.json').read_bytes()
    plan=review.inventory(source,tmp_path,scope='final_checkpoints')
    review.validate_scope(plan)
    assert plan['original_training_best']['checkpoint']==str(source/f'global_step_{best}')
    assert len(plan['policies'])==4
    review.run(obj)
    report=review.verify_completion(Path(obj.config.trainer.default_local_dir)/'new_reevaluation/selected_checkpoints')
    assert len(loads)==4 and report['best_metric']['best_metric_step']==best
    assert report['full_history_best_reselected'] is False
    assert (source/'best_checkpoint_info.json').read_bytes()==before
    assert not (source/f'global_step_best_{best}').exists()


@pytest.mark.parametrize('changes',[
    {'checkpoint_storage':'unknown_storage'},
    {'best_checkpoint_relative_path':'../global_step_900'},
    {'best_checkpoint_relative_path':'global_step_990'}])
def test_designated_numbered_best_cannot_guess_or_retarget_unsafe_reference(tmp_path,monkeypatch,changes):
    obj,loads,source=actual_numbered_best_fixture(tmp_path,monkeypatch,900)
    p=source/'best_checkpoint_info.json';v=review.read(p);v.update(changes);write(p,v)
    with pytest.raises(ValueError,match='Unknown or unsafe'):
        review.inventory(source,tmp_path,scope='final_checkpoints')
    assert not loads
