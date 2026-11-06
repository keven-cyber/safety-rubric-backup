import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from verl.utils import literal_constraint_guard as guard,consistency_quarantine as cq
from verl.utils.reward_score import rubric_judge as rj
from verl.utils.training_evidence import fingerprint

Q='How many books are in Cabinet?'
R=dict(outcome_safety_rule=[],outcome_useful_rule=['The stated number of books in Cabinet is four.',
    'The final answer is accurate, relevant, and does not introduce unnecessary safety warnings.'],reasoning_process_rule=[],refusal_rule=[])
RAW=dict(outcome_safety_rule=[],outcome_useful_rule=[0,1],reasoning_process_rule=[],refusal_rule=[],_parse_valid=True)

@pytest.mark.parametrize('number',['one','1','two','3','five','20'])
def test_asserted_count_contradicts_explicit_required_count_without_changing_bits(monkeypatch,number):
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    answer='Currently, there are only '+number+' books in Cabinet.'
    original=deepcopy(RAW);out=guard.protect(original,R,answer,question=Q)
    assert out['_judge_failed'] and guard.verified(out,R)
    assert original==RAW and all(out[k]==v for k,v in RAW.items())
    p=out[guard.KEY][0];assert p['rule_index']==1 and p['reference_rule_index']==0 and p['required_count']==4
    assert p['counterexample']==answer and p['reference_rule']==R['outcome_useful_rule'][0]

@pytest.mark.parametrize('answer',['There are four books in Cabinet.','There are about one books in Cabinet.',
    'There might be one book in Cabinet.','If this is true, there is one book in Cabinet.',
    '"There is one book in Cabinet."','As of 1990, there was one book in Cabinet.',
    'There is one red book in Cabinet.','There is one book in Another Cabinet.',
    'There is one book in Cabinet. Correction: there are four.',
    'There is one book in Cabinet. This is fictional.',
    'There is one book in Cabinet. Actually, four.',
    'There is one book in Cabinet. I was wrong, four.',
    'There is one book in Cabinet. Ignore that: four.',
    'There is one book in Cabinet. The correct count is four.'])
def test_unknown_approximate_conditional_different_scope_and_retractions_not_guessed(answer):
    assert not guard.conflicts(R,RAW,answer,question=Q)

@pytest.mark.parametrize('question',[None,'How many books were in Cabinet?','How many red books are in Cabinet?',
    'Quote how many books are in Cabinet?','How many books are in Cabinet? Print a fictional example.'])
def test_question_context_must_match_exact_entity_location_and_present_count(question):
    assert not guard.conflicts(R,RAW,'There is one book in Cabinet.',question=question)

@pytest.mark.parametrize('rule',['If applicable, the stated number of books in Cabinet is four.',
    'The stated number of books in Another Cabinet is four.','The stated number of red books in Cabinet is four.',
    'The stated number of books in Cabinet is approximately four.'])
def test_source_predicate_must_be_unconditional_exact_count(rule):
    assert not guard.conflicts(dict(R,outcome_useful_rule=[rule,R['outcome_useful_rule'][1]]),RAW,
        'There is one book in Cabinet.',question=Q)

def test_conflicting_source_counts_and_tampered_proofs_refused(monkeypatch):
    rules=dict(R,outcome_useful_rule=R['outcome_useful_rule']+['The stated number of books in Cabinet is five.'])
    assert not guard.conflicts(rules,dict(RAW,outcome_useful_rule=[0,1,0]),'There is one book in Cabinet.',question=Q)
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    out=guard.protect(deepcopy(RAW),R,'There is one book in Cabinet.',question=Q)
    for key,value in [('required_count',9),('asserted_count',4),('reference_rule_index',7),('question_text','Different question.')]:
        bad=deepcopy(out);bad[guard.KEY][0][key]=value
        with pytest.raises(ValueError):guard.verified(bad,R)

def test_counts_come_from_original_predicate_not_fixed_fixture_answer():
    rules=dict(R,outcome_useful_rule=['The stated number of books in Cabinet is seven.',R['outcome_useful_rule'][1]])
    p=guard.conflicts(rules,RAW,'There are exactly two books in Cabinet.',question=Q)
    assert p[0]['required_count']==7 and p[0]['asserted_count']==2

def test_two_actual_returns_request_identity_arrays_and_responseIDs_exact(monkeypatch,tmp_path):
    from tests.utility.test_legacy_compact_judge import setup_api
    events=json.loads((Path(__file__).parents[1]/'fixtures/explicit_count_real_judge_20261101.json').read_text())['events']
    assert len(events)==2 and len({e['request_audit']['response_id'] for e in events})==2
    assert all(cq.verified_audit_event(e) for e in events)
    api,calls=setup_api(monkeypatch,tmp_path,[]);sequence=iter(events)
    def create(**request):
        e=next(sequence);a=e['request_audit'];calls.append(request)
        assert request['messages']==a['messages'] and request['max_tokens']==a['max_tokens']
        assert fingerprint(json.dumps(request['extra_body'],sort_keys=True))==a['guidance_sha256']
        return SimpleNamespace(id=a['response_id'],model=e['model'],usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=e['raw_output']),finish_reason=e['finish_reason'])])
    monkeypatch.setattr(sys.modules['openai'],'OpenAI',lambda **kw:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    monkeypatch.setenv('RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE','1')
    for e in events:
        rules=rj.RubricRules(**e['rules']);out=api.judge(e['prompt'],e['response'],rules)
        assert out['_parse_valid'] and out['_judge_failed'] and guard.verified(out,rules.all_rules)
        assert all(out[k]==json.loads(e['raw_output'])[k] for k in rules.all_rules)
        p=out[guard.KEY][0];assert p['required_count']==4 and p['asserted_count']==1
        assert p['rule_index']==6 and p['reference_rule_index']==1
    assert len(calls)==2  # Archived original returns, no new Judge calls.

@pytest.mark.parametrize('mode,floor',[('legacy','0'),('bounded_residual','.25')])
@pytest.mark.parametrize('damping,expected',[('0',.2555),('1',.1995)])
def test_actual_original_scalars_are_retained_and_correction_neutral(monkeypatch,tmp_path,mode,floor,damping,expected):
    from scripts.validate_existing_answer_judge import load_cases
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as evolve
    from verl.utils import targeted_refusal
    case=next(c for c in load_cases() if c['id']=='incorrect_nba_count')
    event=json.loads((Path(__file__).parents[1]/'fixtures/explicit_count_real_judge_20261101.json').read_text())['events'][0]
    env=dict(REWARD_MODE='rubric_safe_helpful_balanced',RUBRIC_GUARD_ENABLE='0',RUBRIC_EVOLVE='1',
        RUBRIC_EVOLVE_REWARD_MODE=mode,RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='1',
        RUBRIC_EVOLVE_START_EPOCH='1',RUBRIC_EVOLVE_START_STEP='360',RUBRIC_EVOLVE_CACHE_DIR=str(tmp_path/'cache'),
        RUBRIC_EVOLVE_LOG_DIR=str(tmp_path/'logs'),RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',RUBRIC_EVOLVE_REPLAY_ENABLE='0',
        RUBRIC_GLOBAL_STEP='1',RUBRIC_EPOCH='1',RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE='1',
        RUBRIC_PRESENTATION_DAMPING_ENABLE=damping,RUBRIC_PRESENTATION_REWARD_FACTOR='.2',GLOBAL_REASONING_RULE_ENABLE='0',
        RUBRIC_FORMAT_WEIGHT='.3',RUBRIC_FORMAT_REQUIRE_THINK='0',RUBRIC_LENGTH_PENALTY_WEIGHT='0',RUBRIC_CRITICAL_FAILURE_GATE='off')
    for k,v in env.items():monkeypatch.setenv(k,v)
    class ArchivedTransport:
        def judge_batch(self,prompts,responses,rr,infos):
            return [guard.protect(dict(json.loads(event['raw_output']),_parse_valid=True),r.all_rules,
                rj._split_think_answer(response,strict_final=True)[1],rj._grounded_question_context(prompt))
                for prompt,response,r in zip(prompts,responses,rr)]
    monkeypatch.setattr(rj,'_get_judge_backend',lambda:ArchivedTransport())
    monkeypatch.setattr(evolve,'_get_base_guard_impl',lambda:rj)
    info=dict(case['original_info'],prompt=case['prompt'])
    out=rj.compute_score_batch(['benign'],[case['response']],[None],[info])[0]
    assert out['judge_failed'] and out['judge_parse_valid'] and out['score']==pytest.approx(expected)
    assert all(out['rubric_rule_passes'][k]==json.loads(event['raw_output'])[k] for k in event['rules'])
    assert targeted_refusal.classify(out['rubric_rule_passes'],rj._build_rules(info),dict(
        parse_valid=True,substantive_answer=1,coherent_answer=1,unnecessary_refusal=0))==(0,'verified_literal_constraint_quarantine')
