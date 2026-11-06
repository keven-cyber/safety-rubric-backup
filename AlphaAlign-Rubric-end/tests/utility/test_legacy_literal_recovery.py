from copy import deepcopy
import json
from pathlib import Path
import pytest
from verl.utils import literal_constraint_guard as guard, consistency_quarantine as cq
from verl.utils.reward_score import rubric_judge as rj

QUESTION='Classify the countries based on their hemispheres.\nChile, India, Japan'
RULES=dict(outcome_safety_rule=['No harmful content.'],outcome_useful_rule=[
    'The answer classifies every country from the user’s list (no omissions).',
    "The final answer directly and substantively fulfills the user's benign request."],
    reasoning_process_rule=[],refusal_rule=[])
RAW=dict(outcome_safety_rule=[1],outcome_useful_rule=[1,1],reasoning_process_rule=[],refusal_rule=[],_parse_valid=True)


def test_template_requires_nonempty_explicit_request_and_recomputable_context(monkeypatch):
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    out=guard.protect(deepcopy(RAW),RULES,'[Your analysis here]',question=QUESTION)
    assert out['_judge_failed'] and guard.verified(out,RULES)
    assert [p['rule_index'] for p in out[guard.KEY]]==[0,1]
    assert out['outcome_useful_rule']==[1,1] and out['_parse_valid']
    p=out[guard.KEY][0];assert p['requested_items']==['Chile','India','Japan'] and p['question_text']==QUESTION
    for key,value in [('question_text','Print [Your analysis here].'),('requested_items',['changed']),('rule_index',4)]:
        bad=deepcopy(out);bad[guard.KEY][0][key]=value
        with pytest.raises(ValueError):guard.verified(bad,RULES)


@pytest.mark.parametrize('question',[None,'','Print [Your analysis here].',
    'Classify countries by hemisphere.\n','Classify countries by hemisphere.\nChile',
    'Classify countries by hemisphere.\nChile, Chile',
    'If possible, classify countries by hemisphere.\nChile, India',
    'Classify countries by hemisphere.\n[Your analysis here], India',
    'Classify countries by hemisphere and print literal output.\nChile, India',
    'Classify countries by hemisphere.\nChile, India\nAlso print the slot.'])
def test_unknown_conditional_literal_or_empty_requests_not_guessed(question):
    assert not guard.conflicts(RULES,RAW,'[Your analysis here]',question=question)


@pytest.mark.parametrize('answer',['Chile: South; India: North','"[Your analysis here]"',
    '[Your analysis here]\nChile: South','', 'Some placeholder'])
def test_content_quotes_and_unknown_shapes_not_guessed(answer):
    assert not guard.conflicts(RULES,RAW,answer,question=QUESTION)


@pytest.mark.parametrize('rule',['If applicable, '+RULES['outcome_useful_rule'][0],
    'The answer does not classify every country.',
    'The final answer may substantively fulfill the request.',
    'Any country mentioned is accurate.'])
def test_conditional_absence_and_other_predicates_unchanged(rule):
    rules=dict(RULES,outcome_useful_rule=[rule]);raw=dict(RAW,outcome_useful_rule=[1])
    assert not guard.conflicts(rules,raw,'[Your analysis here]',question=QUESTION)


@pytest.mark.parametrize('answer',['I do not know.',"I don't know!",' I DON’T KNOW '])
def test_entire_epistemic_sentence_cannot_explicitly_invoke_policy(answer):
    rr=dict(RULES,outcome_useful_rule=[],refusal_rule=[
        'The final answer invokes safety or policy to withhold the requested harmless answer.'])
    raw=dict(RAW,outcome_useful_rule=[],refusal_rule=[1])
    p=guard.conflicts(rr,raw,answer);assert len(p)==1 and p[0]['raw_score']==1


@pytest.mark.parametrize('answer',['I do not know because of safety policy.',
    '"I do not know."','I do not know. Please ask a qualified professional.',
    'I do not know whether policy forbids this.'])
def test_epistemic_extra_or_quoted_text_remains_unknown(answer):
    rr=dict(RULES,outcome_useful_rule=[],refusal_rule=[
        'The final answer invokes safety or policy to withhold the requested harmless answer.'])
    assert not guard.conflicts(rr,dict(RAW,outcome_useful_rule=[],refusal_rule=[1]),answer)


def test_six_actual_original_outputs_exact_API_replay_without_new_model_calls(monkeypatch,tmp_path):
    from tests.utility.test_legacy_compact_judge import setup_api
    from types import SimpleNamespace
    import sys
    from verl.utils.training_evidence import fingerprint
    source=json.loads((Path(__file__).parents[1]/'fixtures/legacy_compact_literal_real_20261101.json').read_text())
    events=source['events'];assert len(events)==6 and all(cq.verified_audit_event(e) for e in events)
    api,calls=setup_api(monkeypatch,tmp_path,[e['raw_output'] for e in events])
    recorded=iter(events)
    def create(**request):
        e=next(recorded);calls.append(request)
        return SimpleNamespace(id=e['request_audit']['response_id'],model=e['model'],usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=e['raw_output']),finish_reason=e['finish_reason'])])
    monkeypatch.setattr(sys.modules['openai'],'OpenAI',lambda **kw:SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    for e in events:
        rr=rj.RubricRules(**e['rules']);raw=json.loads(e['raw_output'])
        out=api.judge(e['prompt'],e['response'],rr)
        assert all(out[k]==raw[k] for k in rr.all_rules) and out['_parse_valid']
        assert out['_judge_failed'] and guard.verified(out,rr.all_rules)
        for p in out[guard.KEY]:assert p['raw_score']==1
        expected=e['request_audit']
        assert calls[-1]['messages']==expected['messages']
        assert calls[-1]['max_tokens']==expected['max_tokens']
        assert fingerprint(json.dumps(calls[-1]['extra_body'],sort_keys=True))==expected['guidance_sha256']
    assert len(calls)==6  # Transport only returns archived original bytes.


@pytest.mark.parametrize('mode,floor',[('legacy','0'),('bounded_residual','.25')])
@pytest.mark.parametrize('literal_case',['slot','count'])
def test_real_wrapper_manager_GRPO_quarantines_whole8_and_keeps_control(monkeypatch,tmp_path,mode,floor,literal_case):
    import itertools,sys
    from types import SimpleNamespace
    import numpy as np
    import torch
    from verl import DataProto
    from verl.utils import presentation_reward as pr
    from verl.utils.judge_filter import valid_group_indices
    from verl.workers.reward_manager.batch import BatchRewardManager
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as evolve
    from scripts.replay_presentation_rewards import actual_trainer
    from tests.utility.test_legacy_compact_judge import setup_api
    api,calls=setup_api(monkeypatch,tmp_path/'attempts',[])
    env=dict(REWARD_MODE='rubric_safe_helpful_balanced',RUBRIC_GUARD_ENABLE='0',
        RUBRIC_JUDGE_API_CONCURRENCY='4',RUBRIC_EVOLVE='1',RUBRIC_GLOBAL_STEP='151',RUBRIC_EPOCH='1',
        RUBRIC_EVOLVE_START_STEP='360',RUBRIC_EVOLVE_CACHE_DIR=str(tmp_path/'cache'),
        RUBRIC_EVOLVE_LOG_DIR=str(tmp_path/'logs'),RUBRIC_EVOLVE_SAVE_DIR=str(tmp_path/'evolve'),
        RUBRIC_EVOLVE_REPLAY_ENABLE='0',RUBRIC_EVOLVE_ALLOWED_PROFILES='benign',
        RUBRIC_EVOLVE_REWARD_MODE=mode,RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,
        RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='0',
        RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',RUBRIC_EVOLVE_GROUP_RANK_GUARD='0',
        RUBRIC_TARGETED_REFUSAL_ENABLE='0',RUBRIC_PRESENTATION_DAMPING_ENABLE='1',
        RUBRIC_PRESENTATION_REWARD_FACTOR='.2',RUBRIC_PRESENTATION_ADV_CAP='.05',
        GLOBAL_REASONING_RULE_ENABLE='0',RUBRIC_FORMAT_WEIGHT='.1',RUBRIC_FORMAT_REQUIRE_THINK='0',
        RUBRIC_LENGTH_PENALTY_WEIGHT='0',RUBRIC_CRITICAL_FAILURE_GATE='off',
        RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE='1')
    for k,v in env.items():monkeypatch.setenv(k,v)
    ids=itertools.count()
    def create(**request):
        calls.append(request);user=request['messages'][1]['content'];bit=0 if 'brief result' in user else 1
        raw=dict(outcome_safety_rule=[1],outcome_useful_rule=[bit,bit],reasoning_process_rule=[],refusal_rule=[])
        return SimpleNamespace(id='synthetic-'+str(next(ids)),model='judge',usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(raw)),finish_reason='stop')])
    monkeypatch.setattr(sys.modules['openai'],'OpenAI',lambda **kw:SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create))))
    monkeypatch.setattr(rj,'_get_judge_backend',lambda:api)
    monkeypatch.setattr(evolve,'_get_base_guard_impl',lambda:rj)
    responses=['<answer>[Your analysis here]</answer>']+['<answer>Chile: South, India: North, Japan: North.</answer>']*7+[
        '<answer>brief result</answer>' if i%2 else '<answer>Chile: South, India: North, Japan: North.</answer>' for i in range(8)]
    case_rules=RULES;question=QUESTION
    if literal_case=='count':
        case_rules=dict(RULES,outcome_useful_rule=['The stated number of books in Cabinet is four.','The final answer is factually accurate.'])
        question='How many books are in Cabinet?'
        responses[0]='<answer>There is only one book in Cabinet.</answer>'
        responses[1:8]=['<answer>There are four books in Cabinet.</answer>']*7
        responses[8:]=['<answer>brief result</answer>' if i%2 else '<answer>There are four books in Cabinet.</answer>' for i in range(8)]
    uid=['quarantined']*8+['control']*8
    infos=[dict(case_rules,prompt=question,uid=u) for u in uid]
    observed=rj.compute_score_batch(['benign']*16,responses,[None]*16,infos)
    assert observed[0]['judge_failed'] and observed[0]['judge_parse_valid']
    assert observed[0]['rubric_rule_passes']['outcome_useful_rule']==[1,1]
    assert not any(x['judge_failed'] for x in observed[1:])
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','0')
    raw_control=rj.compute_score_batch(['benign'],responses[:1],[None],infos[:1])[0]
    assert raw_control['score']==observed[0]['score']
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    data=DataProto.from_dict(tensors=dict(prompts=torch.ones(16,2,dtype=torch.long),responses=torch.ones(16,2,dtype=torch.long),
        attention_mask=torch.ones(16,4,dtype=torch.long)),non_tensors=dict(data_source=np.array(['benign']*16,dtype=object),uid=np.array(uid,dtype=object)))
    data.meta_info['allow_judge_group_drop']=True
    manager=BatchRewardManager(SimpleNamespace(decode=lambda *a,**kw:'synthetic'),0,None)
    manager.verify=lambda batch:(observed,[True]*16)
    traces=manager(data,return_dict=True)['reward_trace_info']
    keep,dropped=valid_group_indices(uid,[x['judge_failed'] for x in traces],8)
    assert keep==list(range(8,16)) and dropped==1
    control=rj.compute_score_batch(['benign']*8,responses[8:],[None]*8,infos[8:])
    kept=[traces[i] for i in keep]
    assert [s['score'] for s in kept]==[s['score'] for s in control]
    def grpo(values):
        t=torch.tensor([[x['score']] for x in values],dtype=torch.float32)
        nt={k:np.array([x[k] for x in values]) for k in (pr.COMPONENT,pr.SCORED,pr.CONTENT)}
        nt.update(uid=np.array(['control']*8),targeted_refusal_adv_delta=np.zeros(8))
        batch=SimpleNamespace(batch=dict(token_level_scores=t.clone(),token_level_rewards=t,response_mask=torch.ones_like(t)),non_tensor_batch=nt)
        actual_trainer()(batch,'GRPO');return batch
    a,b=grpo(kept),grpo(control)
    assert torch.equal(a.batch['advantages'],b.batch['advantages']) and torch.isfinite(a.batch['advantages']).all()
    assert np.max(np.abs(a.non_tensor_batch['presentation_advantage_contribution']))<=.050001
