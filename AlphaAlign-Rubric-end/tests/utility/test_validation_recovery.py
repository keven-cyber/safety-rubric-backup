import json
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from verl import DataProto
import ast
from collections import defaultdict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

# Execute the real trainer methods without importing Ray/GPU worker services.
_trainer_path = Path(__file__).resolve().parents[2] / "verl/trainer/ppo/ray_trainer.py"
_tree = ast.parse(_trainer_path.read_text())
_class = next(n for n in _tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
_methods = [n for n in _class.body if isinstance(n, ast.FunctionDef) and n.name in
            {'_validate', '_validate_impl', '_rollout_trace_json_value', '_maybe_save_best_checkpoint'}]
_env = dict(os=os, np=np, torch=torch, defaultdict=defaultdict, DataProto=DataProto,
            pad_dataproto_to_divisor=pad_dataproto_to_divisor, unpad_dataproto=unpad_dataproto)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='RayPPOTrainer', bases=[], keywords=[],
     body=_methods, decorator_list=[])], type_ignores=[])), str(_trainer_path), 'exec'), _env)
RayPPOTrainer = _env['RayPPOTrainer']
from verl.utils import validation_recovery as vr
from verl.utils.reward_score import rubric_judge as rj
from scripts.select_final_checkpoints import select

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests/fixtures/validation_duplicate_evidence_20261031.json"


def test_actual_four_raw_attempts_remain_invalid_and_unedited():
    events = json.loads(FIXTURE.read_text())
    assert [e['attempt'] for e in events] == [1, 2, 3, 4]
    assert len({e['request_audit']['response_id'] for e in events}) == 4
    for event in events:
        obj = json.loads(event['raw_output'])
        rules = rj.RubricRules(**event['rules'])
        evidence = rj._split_think_answer(event['scoped_response'], strict_final=True)[1]
        assert obj['evidence_line_ids'] == [3, 5, 7, 9, 5]
        with pytest.raises(ValueError):
            rj._grounded_output(obj, rules, evidence)
        assert json.loads(event['raw_output']) == obj


def test_actual_api_path_exhausts_four_without_accepting_duplicate_ids(monkeypatch, tmp_path):
    events = json.loads(FIXTURE.read_text())
    calls = []
    def create(**request):
        event = events[len(calls)]
        calls.append(request)
        return SimpleNamespace(id=event['request_audit']['response_id'], model='judge', choices=[
            SimpleNamespace(message=SimpleNamespace(content=event['raw_output']), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    for key, value in dict(RUBRIC_VALIDATION_QUARANTINE_ENABLE='1', RUBRIC_JUDGE_DROP_FAILED='1',
        RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE='1', RUBRIC_JUDGE_API_LOG_ENABLE='0',
        RUBRIC_RUN_DIR=str(tmp_path)).items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(rj, '_api_max_retries', lambda: 3)
    monkeypatch.setattr(rj, '_api_retry_backoff', lambda: 0)
    api = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge')
    event = events[0]
    with pytest.raises(vr.JudgeScoringUnavailable):
        api.judge(event['prompt'], event['response'], rj.RubricRules(**event['rules']))
    assert len(calls) == 4
    saved = [json.loads(line) for file in (tmp_path / 'judge_errors').glob('*.jsonl')
             for line in file.read_text().splitlines()]
    assert [x['raw_output'] for x in saved] == [e['raw_output'] for e in events]


def trainer(tmp_path, count=304):
    obj = RayPPOTrainer.__new__(RayPPOTrainer)
    obj.config = OmegaConf.create(dict(actor_rollout_ref=dict(rollout=dict(val_kwargs=dict(n=1, do_sample=False))),
        reward_model=dict(enable=False), trainer=dict(default_local_dir=str(tmp_path / 'checkpoints'))))
    obj.global_steps = 30
    obj.val_dataloader = [dict(input_ids=torch.ones(count, 2, dtype=torch.long),
        attention_mask=torch.ones(count, 2, dtype=torch.long), position_ids=torch.zeros(count, 2, dtype=torch.long),
        raw_prompt_ids=np.array([[1, 1]] * count, dtype=object),
        extra_info=np.array([dict(sample_id=str(i), outcome_useful_rule=['Original rule.']) for i in range(count)], dtype=object),
        data_source=np.array(['unchanged'] * count, dtype=object))]
    obj.tokenizer = SimpleNamespace(decode=lambda ids, **kw: 'Original decoded text.', eos_token_id=0, pad_token_id=0)
    obj.actor_rollout_wg = SimpleNamespace(world_size=1, generate_sequences=lambda data: DataProto.from_dict(tensors=dict(
        prompts=torch.ones(count, 2, dtype=torch.long), responses=torch.ones(count, 2, dtype=torch.long),
        attention_mask=torch.ones(count, 4, dtype=torch.long))))
    obj.val_reward_fn = lambda *a, **kw: (_ for _ in ()).throw(vr.JudgeScoringUnavailable('finite retries exhausted'))
    obj._maybe_log_val_generations = lambda **kw: pytest.fail('No partial table or aggregate may be emitted')
    return obj


def test_real_validate_archives_all304_tokens_restores_environment_and_cannot_rank_best(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EPOCH', '3')
    obj = trainer(tmp_path)
    result = obj._validate()
    assert result == {}
    assert os.environ['RUBRIC_EVOLVE'] == '1' and os.environ['RUBRIC_EPOCH'] == '3'
    snapshots = list((tmp_path / 'run/validation_pending/generations').glob('*.json'))
    data = json.loads(snapshots[0].read_text())['snapshot']
    assert len(data['outputs']) == len(data['tensors']['responses']) == len(data['non_tensor_batch']['extra_info']) == 304
    event = json.loads(next((tmp_path / 'checkpoints/validation_pending').glob('*.json')).read_text())
    assert event['aggregate_reward'] is None and event['dataset_modified'] is False
    assert event['best_eligible'] is False and event['checkpoint_save_still_required'] is True
    obj._latest_val_metrics, obj._latest_val_step = result, 30
    obj.save_best_ckpt = True
    obj._maybe_save_best_checkpoint(30)  # returns without any model-copy/ranking call
    with pytest.raises(ValueError, match='unresolved frozen validation'):
        select(tmp_path / 'checkpoints', ('900', 'last', 'best'), 1170)


@pytest.mark.parametrize('error', [ValueError('shape'), OSError('storage'), RuntimeError('not Judge unavailable')])
def test_unknown_failures_propagate_and_validation_env_is_restored(monkeypatch, tmp_path, error):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'run'))
    monkeypatch.delenv('RUBRIC_EVOLVE', raising=False)
    monkeypatch.setenv('RUBRIC_EPOCH', '2')
    obj = trainer(tmp_path, 2)
    obj.val_reward_fn = lambda *a, **kw: (_ for _ in ()).throw(error)
    with pytest.raises(type(error), match=str(error)):
        obj._validate()
    assert 'RUBRIC_EVOLVE' not in os.environ and os.environ['RUBRIC_EPOCH'] == '2'
    assert not list((tmp_path / 'checkpoints/validation_pending').glob('*.json'))


def test_complete_scoring_is_the_exact_original_object_and_no_pending_gap(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    original = dict(reward_tensor=torch.tensor([[.595]]), reward_trace_info=[dict(score=.595)])
    assert vr.score_batch(lambda: original, dict(outputs=['original']), tmp_path / 'run', tmp_path / 'ckpt', 30, 1) is original
    vr.require_complete_best(tmp_path / 'ckpt')


def test_default_disabled_does_not_contain_or_archive(monkeypatch, tmp_path):
    monkeypatch.delenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', raising=False)
    with pytest.raises(vr.JudgeScoringUnavailable):
        vr.score_batch(lambda: (_ for _ in ()).throw(vr.JudgeScoringUnavailable('unknown')), {}, None, None, 1, 1)
    assert not list(tmp_path.iterdir())


def test_archive_failure_cannot_be_quarantined(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    called = []
    with pytest.raises(ValueError):
        vr.score_batch(lambda: called.append(1), dict(value=float('nan')), tmp_path / 'run', tmp_path / 'ckpt', 1, 1)
    assert not called


def test_none_from_broken_reward_function_is_not_a_recoverable_judge_result(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'run'))
    obj = trainer(tmp_path, 2)
    obj.val_reward_fn = lambda *a, **kw: None
    with pytest.raises(TypeError):
        obj._validate()
    assert not list((tmp_path / 'checkpoints/validation_pending').glob('*.json'))


def test_actual_reward_manager_retains_original_scores_in_typed_validation_failure(monkeypatch):
    from verl.workers.reward_manager.batch import BatchRewardManager
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE', '1')
    original = [dict(score=-.532, judge_failed=True, judge_parse_valid=True,
                     rubric_rule_passes=dict(outcome_useful_rule=[1]))]
    data = DataProto.from_dict(tensors=dict(prompts=torch.ones(1, 2, dtype=torch.long),
        responses=torch.ones(1, 2, dtype=torch.long), attention_mask=torch.ones(1, 4, dtype=torch.long)),
        non_tensors=dict(data_source=np.array(['original'], dtype=object)))
    manager = BatchRewardManager(SimpleNamespace(decode=lambda *a, **kw: 'original'), 0, None)
    manager.verify = lambda batch: (original, [True])
    with pytest.raises(vr.JudgeScoringUnavailable) as failure:
        manager(data, return_dict=True)
    assert failure.value.evidence['reward_traces'] is original
    assert original[0]['score'] == -.532 and original[0]['rubric_rule_passes']['outcome_useful_rule'] == [1]
    data.meta_info['allow_judge_group_drop'] = True
    result = manager(data, return_dict=True)
    assert result['reward_trace_info'][0]['judge_failed'] is True
    assert result['reward_tensor'].sum().item() == pytest.approx(-.532)


@pytest.mark.parametrize('index', [0, 1])
def test_real_a800_validation_conflicts_have_two_independent_unchanged_raws(index):
    from verl.utils import consistency_quarantine as cq
    rows = json.loads((ROOT / 'tests/fixtures/a800_validation_conflicts_20261031.json').read_text())
    event = rows[index]
    assert cq.verified_audit_event(event)
    pair = event['request_audit']['consistency']
    assert pair['first']['response_id'] != pair['current']['response_id']
    assert pair['first']['score'] != pair['current']['score']
    for part in ['first', 'current']:
        raw = json.loads(pair[part]['raw_output'])
        assert raw['score'] == pair[part]['score']
        assert raw['predicate_truth'] == ('TRUE' if raw['score'] else 'FALSE')
