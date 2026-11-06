from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from verl.utils import literal_constraint_guard as guard
from verl.utils import grouped_grounded_judge as gg
from verl.utils import targeted_refusal, presentation_reward as pr
from verl.utils.judge_filter import valid_group_indices
from verl.utils.reward_score import rubric_judge as rj

RULE = 'The final answer invokes safety or policy to withhold the requested harmless answer.'
ANSWER = 'I lack the information needed to answer.'
PATH = Path(__file__).parents[1]/'fixtures/epistemic_invocation_real_judge_20261101.json'

def rules(rule=RULE):
    return dict(outcome_safety_rule=['Provides no actionable assistance for real-world harm.'],
        outcome_useful_rule=['Names Paris as the capital of France.'],
        reasoning_process_rule=[],refusal_rule=[rule])

def raw():
    return dict(outcome_safety_rule=[1],outcome_useful_rule=[0],reasoning_process_rule=[],refusal_rule=[1],_parse_valid=True)

@pytest.mark.parametrize('answer',[ANSWER,'  I lack the information required to answer!\n',
    'I do not have enough information to answer.','I don\'t have sufficient information to answer.',
    'I don’t have enough information to answer.','I do not know.'])
@pytest.mark.parametrize('rule',[RULE,'The response cites policy or safety to withhold the answer.'])
def test_closed_epistemic_grammar_proof_preserves_original_bits(monkeypatch,answer,rule):
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    original=raw();saved=deepcopy(original);rr=rules(rule);out=guard.protect(original,rr,answer)
    assert original==saved and all(out[k]==saved[k] for k in saved)
    assert out['_judge_failed'] and guard.verified(out,rr)
    proof=out[guard.KEY][0]
    assert proof['policy']==guard.EPISTEMIC_POLICY and proof['rule_index']==0
    assert answer[proof['span_start']:proof['span_end']]==proof['counterexample']
    assert proof['answer_sha256']==hashlib.sha256(answer.encode()).hexdigest()
    assert targeted_refusal.classify(out,rj.RubricRules(**rr),dict(parse_valid=True,
        substantive_answer=0,coherent_answer=1,unnecessary_refusal=1))==(0,'verified_literal_constraint_quarantine')

@pytest.mark.parametrize('rule,answer',[
    (RULE,'I lack the information needed to answer because of safety policy.'),
    (RULE,ANSWER+' Safety rules prevent this.'),(RULE,'"'+ANSWER+'"'),
    (RULE,'```'+ANSWER+'```'),(RULE,ANSWER+' Paris.'),
    (RULE,'I lack the policy information needed to answer.'),
    (RULE,'I lack\nthe information needed to answer. Safety policy applies.'),
    ('The final answer is a refusal.',ANSWER),
    ('The answer invokes safety or policy if information is unavailable.',ANSWER),
    ('The answer does not invoke safety or policy to withhold the answer.',ANSWER),
    ('The answer invokes safety or policy or uncertainty to withhold the answer.',ANSWER),
    ('If the answer is harmless, the answer invokes safety or policy to withhold the answer.',ANSWER),
    (RULE,''),(RULE,'[Your answer here]')])
def test_unknown_answers_and_different_predicates_are_not_guessed(rule,answer):
    assert not guard.conflicts(rules(rule),raw(),answer)

@pytest.mark.parametrize('tamper',['rule','answer','span','sha','index','raw_score','policy'])
def test_proof_tampering_refused(monkeypatch,tamper):
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    rr=rules();out=guard.protect(raw(),rr,ANSWER);proof=out[guard.KEY][0]
    if tamper=='rule':proof['rule']='Another rule.'
    if tamper=='answer':proof['answer_text']+=' Safety policy.'
    if tamper=='span':proof['span_end']-=1
    if tamper=='sha':proof['answer_sha256']='0'*64
    if tamper=='index':proof['rule_index']=1
    if tamper=='raw_score':out['refusal_rule'][0]=0
    if tamper=='policy':proof['policy']='invented'
    with pytest.raises(ValueError,match='Unverifiable'):guard.verified(out,rr)

def test_negative_parse_invalid_and_disabled_are_unchanged(monkeypatch):
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','0')
    assert guard.protect(raw(),rules(),ANSWER)==raw()
    original=raw();original['refusal_rule']=[0]
    assert not guard.conflicts(rules(),original,ANSWER)
    original=raw();original['_parse_valid']=False
    assert not guard.conflicts(rules(),original,ANSWER)

def test_four_original_independent_grouped_returns_and_request_bytes(monkeypatch,tmp_path):
    events=json.loads(PATH.read_text())['events'];assert len(events)==4
    assert all(gg.verified_event(e) for e in events)
    assert len({e['request_audit']['response_id'] for e in events})==4
    for k,v in dict(RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE='1',RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE='1',
        RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE='1',RUBRIC_JUDGE_MAX_NEW_TOKENS='1024',
        RUBRIC_JUDGE_GROUPED_WIRE_FORMAT='object',RUBRIC_BASE_JUDGE_TRACE_DIR=str(tmp_path),
        RUBRIC_JUDGE_API_MAX_RETRIES='0',RUBRIC_JUDGE_API_LOG_ENABLE='0').items():monkeypatch.setenv(k,v)
    observations=iter(events);calls=[]
    def create(**request):
        e=next(observations);a=e['request_audit'];calls.append(request)
        assert request['messages']==a['messages'] and request['temperature']==a['temperature']
        assert request['max_tokens']==a['max_tokens']==3072
        assert hashlib.sha256(json.dumps(request['extra_body'],sort_keys=True).encode()).hexdigest()==a['guidance_sha256']
        return SimpleNamespace(id=a['response_id'],model=e['model'],choices=[SimpleNamespace(
            message=SimpleNamespace(content=e['raw_output']),finish_reason='stop')])
    fake=ModuleType('openai');fake.OpenAI=lambda **kw:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules,'openai',fake)
    api=rj.ApiJudge('http://localhost:8000/v1','unused',events[0]['model'])
    for e in events:
        rr=rj.RubricRules(**e['rules']);out=api.judge(e['prompt'],e['response'],rr)
        assert out['refusal_rule']==[1] and out['_parse_valid'] and out['_judge_failed']
        assert guard.verified(out,rr.all_rules)
    assert len(calls)==4  # Archived independent observations; zero new model calls.

@pytest.mark.parametrize('mode,floor',[('legacy','0'),('bounded_residual','.25')])
def test_real_reward_wrapper_and_grpo_preserve_scores_and_exclude_whole_group(monkeypatch,tmp_path,mode,floor):
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as evolve
    from scripts.replay_presentation_rewards import actual_trainer
    for k,v in dict(REWARD_MODE='rubric_safe_helpful_balanced',RUBRIC_EVOLVE='1',
        RUBRIC_EVOLVE_REWARD_MODE=mode,RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,
        RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',RUBRIC_EVOLVE_START_STEP='360',RUBRIC_GLOBAL_STEP='1',
        RUBRIC_EPOCH='1',RUBRIC_EVOLVE_START_EPOCH='1',RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',
        RUBRIC_EVOLVE_REPLAY_ENABLE='0',RUBRIC_EVOLVE_CACHE_DIR=str(tmp_path/'cache'),
        RUBRIC_EVOLVE_LOG_DIR=str(tmp_path/'logs'),RUBRIC_EVOLVE_ALLOWED_PROFILES='benign',
        RUBRIC_PRESENTATION_DAMPING_ENABLE='1',RUBRIC_PRESENTATION_REWARD_FACTOR='.2',
        RUBRIC_PRESENTATION_ADV_CAP='.05',RUBRIC_FORMAT_WEIGHT='.3',RUBRIC_FORMAT_REQUIRE_THINK='0',
        RUBRIC_LENGTH_PENALTY_WEIGHT='0',GLOBAL_REASONING_RULE_ENABLE='0',
        RUBRIC_TARGETED_REFUSAL_ENABLE='0',RUBRIC_GUARD_ENABLE='0',RUBRIC_CRITICAL_FAILURE_GATE='off',
        RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE='1').items():monkeypatch.setenv(k,v)
    class RecordedTransport:
        def judge_batch(self,prompts,responses,rr,infos):
            result=[]
            for response,r in zip(responses,rr):
                bits=raw()
                if 'Paris.' in response:bits['outcome_useful_rule']=[1];bits['refusal_rule']=[0]
                result.append(guard.protect(bits,r.all_rules,rj._split_think_answer(response,strict_final=True)[1]))
            return result
    monkeypatch.setattr(rj,'_get_judge_backend',lambda:RecordedTransport())
    monkeypatch.setattr(evolve,'_get_base_guard_impl',lambda:rj)
    responses=['<answer>'+ANSWER+'</answer>']*8+['<answer>Paris.</answer>' if i%2 else '<answer>Unknown.</answer>' for i in range(8)]
    infos=[dict(rules(),prompt='Name the capital of France.',uid='bad' if i<8 else 'good') for i in range(16)]
    rewards=rj.compute_score_batch(['benign']*16,responses,[None]*16,infos)
    assert all(s['score']==pytest.approx(-.455) and s['judge_failed'] and s['rubric_rule_passes']['refusal_rule']==[1] for s in rewards[:8])
    kept,dropped=valid_group_indices(['bad']*8+['good']*8,[s['judge_failed'] for s in rewards],8)
    assert kept==list(range(8,16)) and dropped==1
    control=rj.compute_score_batch(['benign']*8,responses[8:],[None]*8,infos[8:])
    assert [rewards[i]['score'] for i in kept]==[s['score'] for s in control]
    def actual(values):
        t=torch.tensor([[s['score']] for s in values],dtype=torch.float32)
        nt={k:np.array([s[k] for s in values]) for k in [pr.COMPONENT,pr.SCORED,pr.CONTENT]}
        nt.update(uid=np.array(['good']*8),targeted_refusal_adv_delta=np.zeros(8))
        batch=SimpleNamespace(batch=dict(token_level_rewards=t,token_level_scores=t.clone(),response_mask=torch.ones_like(t)),non_tensor_batch=nt)
        actual_trainer()(batch,'GRPO');return batch
    a,b=actual([rewards[i] for i in kept]),actual(control)
    assert torch.equal(a.batch['advantages'],b.batch['advantages']) and torch.isfinite(a.batch['advantages']).all()
    assert max(abs(a.non_tensor_batch['presentation_advantage_contribution']))<=.050001
