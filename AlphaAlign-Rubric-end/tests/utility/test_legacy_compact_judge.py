import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import pytest

from verl.utils import legacy_compact_judge as legacy, consistency_quarantine as cq
from verl.utils.reward_score import rubric_judge as rj
from scripts import judge_replay_server as profiles

ROOT=Path(__file__).resolve().parents[2]


def setup_api(monkeypatch,tmp_path,outputs):
    for k,v in {'RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE':'1','RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE':'0',
        'RUBRIC_JUDGE_GRANULARITY':'all_rules','RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE':'1',
        'RUBRIC_JUDGE_API_LOG_ENABLE':'0','RUBRIC_JUDGE_API_RETRY_BACKOFF':'0',
        'RUBRIC_JUDGE_API_MAX_RETRIES':'3','RUBRIC_JUDGE_MAX_NEW_TOKENS':'1024',
        'RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE':'1','RUBRIC_JUDGE_DROP_FAILED':'1',
        'RUBRIC_BASE_JUDGE_TRACE_DIR':str(tmp_path)}.items():monkeypatch.setenv(k,v)
    requests=[];sequence=iter(outputs)
    def create(**kw):
        requests.append(kw);value=next(sequence)
        if isinstance(value,Exception):raise value
        text= value if isinstance(value,str) else json.dumps(value)
        return SimpleNamespace(id='response-'+str(len(requests)),model='qwen3_32B_instruct',usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=text),finish_reason='stop')])
    fake=ModuleType('openai');fake.OpenAI=lambda **kw:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules,'openai',fake)
    return rj.ApiJudge('http://localhost:8000/v1','not-displayed','qwen3_32B_instruct'),requests


def rules():return rj.RubricRules(['Safe.'],['Useful.','Coherent.'],[],['Invokes a safety/policy restriction.'])
def raw(bit):return dict(outcome_safety_rule=[bit],outcome_useful_rule=[bit,bit],reasoning_process_rule=[],refusal_rule=[0])


def test_exact_original_system_and_user_for_every_frozen_case():
    text=subprocess.check_output(['git','show','d7af3b13f7a8af20c6b6a95d929bcf2112881b94:AlphaAlign-Rubric-end/verl/utils/reward_score/rubric_judge.py'],cwd=ROOT,text=True)
    tree=ast.parse(text);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ApiJudge');fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='judge')
    # Extract original request assembly, not a hand-written copy of expected text.
    body=[]
    for n in fn.body:
        if isinstance(n,ast.FunctionDef) and n.name=='fmt_rules':body.append(n)
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['system','user'] or isinstance(t,ast.Tuple) for t in n.targets):
            if any(isinstance(x,ast.Name) and x.id in ['think_text','system','user'] for x in ast.walk(n)):body.append(n)
    source=ast.Module(body=body,type_ignores=[])
    files=['answer_grounding_existing_answers_20261028.json','targeted_refusal_existing_answers.json']
    count=0
    for name in files:
        for c in json.loads((ROOT/'tests/fixtures'/name).read_text())['cases']:
            rr=rj._build_rules(c['original_info']);env=dict(prompt=c['prompt'],response=c['response'],rules=rr,List=list,_split_think_answer=rj._split_think_answer)
            exec(compile(ast.fix_missing_locations(source),'original_d7af_request','exec'),env)
            assert legacy.request(c['prompt'],c['response'],rr)==(env['system'],env['user']);count+=1
    assert count==36


def test_one_full_request_exact_arrays_no_reasons_and_source_verified_audit(monkeypatch,tmp_path):
    api,requests=setup_api(monkeypatch,tmp_path,[raw(1)]*3)
    out=api.judge_batch(['p','q','r'],['<think>T</think><answer>A</answer>']*3,[rules()]*3)
    assert len(requests)==3 and all(o['_parse_valid'] for o in out)
    assert all(len(q['messages'])==2 and q['max_tokens']==1024 and q['extra_body']['seed']==0 for q in requests)
    assert all(q['messages'][0]['content']==legacy.SYSTEM for q in requests)
    events=[json.loads(v) for p in tmp_path.glob('*.jsonl') for v in p.read_text().splitlines()]
    assert len(events)==3 and all(cq.verified_audit_event(e) for e in events)
    assert all(e['request_audit']['evidence_led'] is False for e in events)


def test_conflicting_independent_compact_arrays_preserved_and_whole_proof_verified(monkeypatch,tmp_path):
    api,_=setup_api(monkeypatch,tmp_path,[raw(1),raw(0)])
    a=api.judge('p','A',rules());b=api.judge('p','A',rules())
    assert {k:b[k] for k in rules().all_rules}==raw(0) and b['_judge_failed']
    assert cq.verified(b,rules().all_rules) and a['outcome_safety_rule']==[1]
    audit=cq.audit_directory(tmp_path);assert not audit['passed'] and len(audit['conflicts'])==1 and audit['conflicts'][0]['evidence_verified']
    corrupt=deepcopy(b);corrupt[cq.KEY][0]['original_response']='changed'
    assert not cq.verified(corrupt,rules().all_rules)


@pytest.mark.parametrize('bad',['{}','{"outcome_safety_rule":[true]}',json.dumps(dict(raw(1),extra=0))])
def test_four_finite_invalid_returns_are_not_imputed_or_accepted(monkeypatch,tmp_path,bad):
    api,req=setup_api(monkeypatch,tmp_path,[bad]*4)
    result=api.judge('p','A',rules())
    assert len(req)==4 and result['_parse_valid'] is False and result['_judge_failed'] is True


def test_unknown_runtime_error_not_swallowed(monkeypatch,tmp_path):
    api,_=setup_api(monkeypatch,tmp_path,[RuntimeError('unknown')]*4)
    with pytest.raises(RuntimeError,match='no reward was fabricated'):api.judge('p','A',rules())


def test_targeted_correction_keeps_original_single_rule_grounding(monkeypatch,tmp_path):
    rawg=dict(evidence_line_ids=[1],reason='Observed.',predicate_truth='TRUE',score=1)
    api,requests=setup_api(monkeypatch,tmp_path,[rawg]);rr=rj.RubricRules([],['Written predicate.'],[],[])
    out=api.judge('p','<answer>A</answer>',rr,{'targeted_refusal_diagnostic_only':True})
    assert out['_parse_valid'] and len(requests)==1
    assert 'rule_to_evaluate' in json.loads(requests[0]['messages'][1]['content'])


def test_prefix_profile_retains_model_budget_and_precision(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE','legacy_compact_v8')
    argv=profiles.command('/model','qwen3_32B_instruct',8000)
    assert '--enable-prefix-caching' in argv and '--no-enable-prefix-caching' not in argv
    assert '--no-enable-chunked-prefill' in argv and '--enforce-eager' not in argv
    assert argv[argv.index('--max-model-len')+1]=='16384'
    assert argv[argv.index('--max-num-batched-tokens')+1]=='16384'


def test_protocol_transition_is_durable_and_blocks_final_without_altering_best(monkeypatch,tmp_path):
    from verl.utils.validation_recovery import require_complete_best
    monkeypatch.setenv('RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE','1')
    a,b=tmp_path/'run',tmp_path/'root';a.mkdir();b.mkdir();best=b/'best_checkpoint_info.json';best.write_text('{"original":true}')
    recovery=dict(source_checkpoint='/original/global_step_150',restored_step=150)
    legacy.transition(a,b,recovery);legacy.transition(a,b,recovery)
    assert best.read_text()=='{"original":true}' and (a/'scoring_protocol_transition.json').read_bytes()==(b/'scoring_protocol_transition.json').read_bytes()
    with pytest.raises(ValueError,match='same-protocol'):require_complete_best(b)


def test_actual_native_grammar_all36_shapes():
    from scripts import validate_existing_answer_judge as val,smoke_targeted_refusal as smoke
    from scripts.validate_legacy_compact import grammar
    result=grammar(val.load_cases()+json.loads(smoke.FIXTURE.read_text())['cases'])
    assert result['passed'] and result['case_count']==36 and result['backend_version']=='0.7.30'


def test_duplicate_json_keys_cannot_be_accepted_as_a_compact_verdict(monkeypatch,tmp_path):
    bad=json.dumps(raw(1)).replace('"outcome_safety_rule": [1]', '"outcome_safety_rule": [0], "outcome_safety_rule": [1]')
    api,requests=setup_api(monkeypatch,tmp_path,[bad]*4)
    result=api.judge('p','A',rules())
    assert len(requests)==4 and result['_judge_failed'] and not result['_parse_valid']


def test_explicit_context400_is_archived_and_typed_once_without_shortening(monkeypatch,tmp_path):
    from verl.utils.validation_recovery import JudgeScoringUnavailable
    class CapacityError(Exception):status_code=400
    error=CapacityError("This model's maximum context length is 16384 tokens. However, you requested 17024 tokens (16000 in the messages, 1024 in the completion).")
    api,requests=setup_api(monkeypatch,tmp_path,[error]);monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE','1')
    with pytest.raises(JudgeScoringUnavailable,match='context capacity'):api.judge('p','A',rules())
    assert len(requests)==1 and requests[0]['max_tokens']==1024
    events=[json.loads(v) for p in tmp_path.glob('*.jsonl') for v in p.read_text().splitlines()]
    assert len(events)==1 and events[0]['raw_output'] is None and events[0]['request_audit']['context_capacity_failure']['no_verdict']
