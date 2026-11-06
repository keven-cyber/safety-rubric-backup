"""CPU synthetic8x8-group boundaries; no GPU/accuracy/throughput claim."""
import itertools
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from concurrent.futures import Future

import numpy as np
import pytest
import torch
from verl import DataProto
from verl.utils import presentation_reward as pr
from verl.utils.judge_filter import valid_group_indices
from verl.utils.reward_score import rubric_judge as rj
from verl.utils.validation_recovery import JudgeScoringUnavailable
from verl.workers.reward_manager.batch import BatchRewardManager


@pytest.mark.parametrize('concise',[False,True])
@pytest.mark.parametrize('mode,floor',[('legacy','0'),('bounded_residual','.25')])
def test_real_online_wrapper_manager_and_grpo_keep_unrelated_complete_groups(monkeypatch,tmp_path,mode,floor,concise):
    from scripts.replay_presentation_rewards import actual_trainer
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as evolve
    env=dict(REWARD_MODE='rubric_safe_helpful_balanced',RUBRIC_GUARD_ENABLE='0',
        RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE='1',RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE='0',
        RUBRIC_JUDGE_DROP_FAILED='1',RUBRIC_VALIDATION_QUARANTINE_ENABLE='1',
        RUBRIC_JUDGE_API_CONCURRENCY='4',RUBRIC_JUDGE_API_MAX_RETRIES='3',
        RUBRIC_JUDGE_API_RETRY_BACKOFF='0',RUBRIC_JUDGE_API_LOG_ENABLE='0',
        RUBRIC_BASE_JUDGE_TRACE_DIR=str(tmp_path/'attempts'),RUBRIC_EVOLVE='1',
        RUBRIC_GLOBAL_STEP='144',RUBRIC_EPOCH='1',RUBRIC_EVOLVE_START_STEP='360',
        RUBRIC_EVOLVE_CACHE_DIR=str(tmp_path/'cache'),RUBRIC_EVOLVE_LOG_DIR=str(tmp_path/'logs'),
        RUBRIC_EVOLVE_SAVE_DIR=str(tmp_path/'evolve'),RUBRIC_EVOLVE_REPLAY_ENABLE='0',
        RUBRIC_EVOLVE_ALLOWED_PROFILES='benign',RUBRIC_EVOLVE_REWARD_MODE=mode,
        RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',
        RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='0',RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',
        RUBRIC_EVOLVE_GROUP_RANK_GUARD='0',RUBRIC_TARGETED_REFUSAL_ENABLE='0',
        RUBRIC_PRESENTATION_DAMPING_ENABLE='1',RUBRIC_PRESENTATION_REWARD_FACTOR='.2',
        RUBRIC_PRESENTATION_ADV_CAP='.05',GLOBAL_REASONING_RULE_ENABLE='0',RUBRIC_FORMAT_WEIGHT='.1',
        RUBRIC_FORMAT_REQUIRE_THINK='0',RUBRIC_LENGTH_PENALTY_WEIGHT='0',RUBRIC_CRITICAL_FAILURE_GATE='off')
    env.update(RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE='1' if concise else '0',
        RUBRIC_JUDGE_GRANULARITY='all_rules' if concise else 'single_rule',
        RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE='1')
    for k,v in env.items():monkeypatch.setenv(k,v)
    calls=[];ids=itertools.count()
    def create(**request):
        rid=next(ids);calls.append(request)
        if concise:
            content=request['messages'][1]['content'];short='short' in content;bad='bad0' in content
            value=dict(outcome_safety_rule=[1],outcome_useful_rule=[0 if short else 1]*(1 if bad else 2),reasoning_process_rule=[],refusal_rule=[0])
            return SimpleNamespace(id=f'original-{rid}',model='judge',choices=[SimpleNamespace(
                message=SimpleNamespace(content=json.dumps(value)),finish_reason='stop')])
        user=json.loads(request['messages'][1]['content'])
        text=''.join(row['text'] for row in user['assistant_evidence_lines']);rule=user['rule_to_evaluate']
        bad='bad0' in text and 'relevant' in rule
        score=0 if 'refusal' in rule or 'short' in text else 1
        raw=json.dumps(dict(evidence_line_ids=[1,1] if bad else [1],reason='Original independent observation',
                           predicate_truth='TRUE' if score else 'FALSE',score=score))
        return SimpleNamespace(id=f'original-{rid}',model='judge',choices=[SimpleNamespace(
            message=SimpleNamespace(content=raw),finish_reason='stop')])
    mod=ModuleType('openai');mod.OpenAI=lambda **kw:SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)));monkeypatch.setitem(sys.modules,'openai',mod)
    api=rj.ApiJudge('http://localhost:8000/v1','unused','judge')
    monkeypatch.setattr(rj,'_get_judge_backend',lambda:api)
    monkeypatch.setattr(evolve,'_get_base_guard_impl',lambda:rj)
    rules=rj.RubricRules(['The answer is safe.'],['First original predicate.',
        'The answer gives relevant information.'],[],['The answer is a refusal.'])
    responses=['<answer>bad0</answer>']+['<answer>helpful</answer>']*7+[
        '<answer>helpful</answer>' if i%2 else '<answer>short</answer>' for i in range(8)]
    uid=['unavailable-group']*8+['valid-group']*8
    infos=[dict(rules.all_rules,prompt=q,uid=q) for q in uid]
    scores=rj.compute_score_batch(['benign']*16,responses,[None]*16,infos)
    assert len(scores)==16 and scores[0]['judge_failed'] and not scores[0]['judge_parse_valid']
    assert not any(s['judge_failed'] for s in scores[1:])
    if not concise:
        assert scores[0]['rubric_rule_passes']['_scoring_unavailable'][0]['no_valid_verdict']
        assert scores[0]['rubric_rule_passes']['_scoring_unavailable'][0]['rule_index']==1
        assert scores[0]['rubric_rule_passes']['outcome_useful_rule'][0]==1
    data=DataProto.from_dict(tensors=dict(prompts=torch.ones(16,2,dtype=torch.long),
        responses=torch.ones(16,2,dtype=torch.long),attention_mask=torch.ones(16,4,dtype=torch.long)),
        non_tensors=dict(data_source=np.array(['benign']*16,dtype=object),uid=np.array(uid,dtype=object)))
    data.meta_info['allow_judge_group_drop']=True
    manager=BatchRewardManager(SimpleNamespace(decode=lambda *a,**kw:'synthetic'),0,None)
    manager.verify=lambda batch:(scores,[True]*16)
    observed=manager(data,return_dict=True);traces=observed['reward_trace_info']
    keep,dropped=valid_group_indices(uid,[s['judge_failed'] for s in traces],8)
    assert keep==list(range(8,16)) and dropped==1 and len(traces)==16
    control=rj.compute_score_batch(['benign']*8,responses[8:],[None]*8,infos[8:])
    def actual_grpo(values):
        tensor=torch.tensor([[s['score']] for s in values],dtype=torch.float32)
        nt={k:np.array([s[k] for s in values]) for k in (pr.COMPONENT,pr.SCORED,pr.CONTENT)}
        nt.update(uid=np.array(['valid-group']*8),targeted_refusal_adv_delta=np.zeros(8))
        batch=SimpleNamespace(batch=dict(token_level_scores=tensor.clone(),token_level_rewards=tensor,
                            response_mask=torch.ones_like(tensor)),non_tensor_batch=nt)
        actual_trainer()(batch,'GRPO');return batch
    kept=[traces[i] for i in keep]
    assert [s['score'] for s in kept]==[s['score'] for s in control]
    a,b=actual_grpo(kept),actual_grpo(control)
    assert torch.equal(a.batch['advantages'],b.batch['advantages']) and torch.isfinite(a.batch['advantages']).all()
    assert np.max(np.abs(a.non_tensor_batch['presentation_advantage_contribution']))<=.050001
    events=[json.loads(line) for p in (tmp_path/'attempts').glob('*.jsonl') for line in p.read_text().splitlines()]
    failed=[e for e in events if e['error_type']]
    assert len(failed)==4
    if concise:
        assert all(len(json.loads(e['raw_output'])['outcome_useful_rule'])==1 for e in failed)
    else:
        assert all(json.loads(e['raw_output'])['evidence_line_ids']==[1,1] for e in failed)
    assert len({e['request_audit']['response_id'] for e in failed})==4
    data.meta_info['allow_judge_group_drop']=False
    with pytest.raises(JudgeScoringUnavailable):manager(data,return_dict=True)


@pytest.mark.parametrize('error',[ValueError('unknown'),TypeError('storage'),RuntimeError('uncontrolled')])
def test_unknown_request_errors_still_propagate(monkeypatch,error):
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED','1');f=Future();f.set_exception(error)
    with pytest.raises(type(error),match=str(error)):
        rj._resolve_base_judge_future(f,rj.RubricRules([],['original'],[],[]),None)


@pytest.mark.parametrize('drop,targeted',[('0',False),('1',True)])
def test_unavailability_does_not_override_disabled_filter_or_correction(monkeypatch,drop,targeted):
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED',drop);f=Future();f.set_exception(JudgeScoringUnavailable('unknown verdict'))
    with pytest.raises(JudgeScoringUnavailable):rj._resolve_base_judge_future(f,
        rj.RubricRules([],['original'],[],[]),{'targeted_refusal_reference_only':targeted})
