import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
from verl.utils import grouped_grounded_judge as gg
from verl.utils import consistency_quarantine as cq
from verl.utils.reward_score import rubric_judge as rj

RULES = rj.RubricRules(['Does not add prohibited content.'], ['Actually answers the question.', 'Uses a clear list.'], ['Reasons correctly.'], ['Declines help.'])
RESPONSE = '<think>hidden reasoning\ncalculation</think><answer>final answer\nsecond line</answer>'


def raw(scores=None):
    scores = scores or [1, 0, 1, 1, 0]
    return {key: dict(evidence_line_ids=[1], reason='Original evidence for this rule',
        predicate_truth='TRUE' if score else 'FALSE', score=score)
        for (key, *_), score in zip(gg.slots(RULES), scores)}


def api(monkeypatch, tmp_path, responses):
    monkeypatch.setenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_BASE_JUDGE_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_JUDGE_API_MAX_RETRIES', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    requests = []
    responses = iter(responses)
    def create(**kw):
        requests.append(kw)
        value = next(responses)
        if isinstance(value, Exception):
            raise value
        return SimpleNamespace(id='independent-' + str(len(requests)), model='judge',
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(value)), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    return rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge'), requests


def test_full_rule_group_uses_one_http_call_and_preserves_every_binary_slot(monkeypatch, tmp_path):
    backend, requests = api(monkeypatch, tmp_path, [raw()])
    result = backend.judge_batch(['question'], [RESPONSE], [RULES])[0]
    assert len(requests) == 1
    assert result == dict(outcome_safety_rule=[1], outcome_useful_rule=[0, 1],
        reasoning_process_rule=[1], refusal_rule=[0], _parse_valid=True)
    payload = json.loads(requests[0]['messages'][1]['content'])
    assert [x['rule'] for x in payload['rules_to_evaluate']] == [x[3] for x in gg.slots(RULES)]
    assert payload['assistant_evidence_sections']['final_answer'][0]['text'] == 'final answer\n'
    assert payload['assistant_evidence_sections']['reasoning'][0]['text'] == 'hidden reasoning\n'
    events = [json.loads(line) for p in tmp_path.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert len(events) == 1 and cq.verified_audit_event(events[0])
    assert requests[0]['extra_body']['seed'] == 0
    assert requests[0]['max_tokens'] == 512 * 5


@pytest.mark.parametrize('count,budget', [(21,1024),(42,512)])
def test_overflow_rule_count_keeps_all_slots_and_configured_completion_budget(monkeypatch, tmp_path,count,budget):
    monkeypatch.setenv('RUBRIC_JUDGE_MAX_NEW_TOKENS',str(budget))
    many = rj.RubricRules([], [f'Original predicate {i}' for i in range(count)], [], [])
    complete = {key: dict(evidence_line_ids=[1], reason='Complete original reasoning',
        predicate_truth='TRUE', score=1) for key,*_ in gg.slots(many)}
    backend, requests = api(monkeypatch, tmp_path, [complete])
    out = backend.judge('question', RESPONSE, many)
    assert out['outcome_useful_rule'] == [1] * count and out['_parse_valid']
    assert len(requests) == 1 and requests[0]['max_tokens'] == 21504
    payload = json.loads(requests[0]['messages'][1]['content'])
    assert len(payload['rules_to_evaluate']) == count
    assert requests[0]['extra_body']['seed'] == 0


@pytest.mark.parametrize('change', ['missing_rule', 'extra_rule', 'bool', 'truth', 'duplicate_evidence',
    'invented_evidence', 'wrong_scope', 'long_reason', 'empty_reason', 'extra_field', 'absence_without_counterexample'])
def test_grouped_evidence_is_never_filled_coerced_or_misaligned(change):
    value = raw(); key = gg.slots(RULES)[0][0]
    if change == 'missing_rule': del value[key]
    if change == 'extra_rule': value['unknown:0'] = value[key]
    if change == 'bool': value[key]['score'] = True
    if change == 'truth': value[key]['predicate_truth'] = 'FALSE'
    if change == 'duplicate_evidence': value[key]['evidence_line_ids'] = [1, 1]
    if change == 'invented_evidence': value[key]['evidence_line_ids'] = [1000]
    if change == 'wrong_scope': value['reasoning_process_rule:0']['evidence_line_ids'] = [3]
    if change == 'long_reason': value[key]['reason'] = 'a' * 601
    if change == 'empty_reason': value[key]['reason'] = ' '
    if change == 'extra_field': value[key]['unrequested'] = 1
    if change == 'absence_without_counterexample': value[key].update(score=0, predicate_truth='FALSE', evidence_line_ids=[])
    _, _, _, sections = gg.request('question', RESPONSE, RULES)
    with pytest.raises(ValueError): gg.output(value, RULES, sections)


def test_duplicate_rule_identity_in_raw_json_is_rejected():
    with pytest.raises(ValueError, match='Duplicate'):
        gg.parse('{"outcome_useful_rule:0":{},"outcome_useful_rule:0":{}}')


def test_independent_vector_conflict_keeps_original_bits_and_provable_group_exclusion(monkeypatch, tmp_path):
    first = raw(); changed = raw([1, 1, 1, 1, 0])
    backend, requests = api(monkeypatch, tmp_path, [first, changed, changed])
    a = backend.judge('question', RESPONSE, RULES)
    b = backend.judge('question', RESPONSE, RULES)
    c = backend.judge('question', RESPONSE, RULES)
    assert a['outcome_useful_rule'] == [0, 1]
    assert b['outcome_useful_rule'] == c['outcome_useful_rule'] == [1, 1]
    assert b['_judge_failed'] and c['_judge_failed'] and cq.verified(b, RULES.all_rules) and cq.verified(c, RULES.all_rules)
    audit = cq.audit_directory(tmp_path)
    assert not audit['passed'] and audit['independent_repeats_checked'] == 2
    assert all(row['evidence_verified'] for row in audit['conflicts'])
    changed = copy.deepcopy(b)
    changed[cq.KEY][0]['conflict']['first']['response_id'] = changed[cq.KEY][0]['conflict']['current']['response_id']
    assert not cq.verified(changed, RULES.all_rules)
    from verl.utils.judge_filter import valid_group_indices
    # Two complete prompt groups, each eight; tainted group is entirely excluded.
    assert valid_group_indices(['tainted'] * 8 + ['valid'] * 8, [True] + [False] * 15, 8) == (list(range(8,16)), 1)


def test_grouped_trace_tampering_is_detected(monkeypatch, tmp_path):
    backend, _ = api(monkeypatch, tmp_path, [raw()])
    backend.judge('question', RESPONSE, RULES)
    e = json.loads(next(tmp_path.glob('*.jsonl')).read_text().splitlines()[0])
    assert gg.verified_event(e)
    e['response'] = e['response'].replace('final answer', 'changed answer')
    assert not gg.verified_event(e)


def test_targeted_correction_stays_on_existing_single_rule_contract(monkeypatch, tmp_path):
    single = dict(evidence_line_ids=[1], reason='literal', predicate_truth='TRUE', score=1)
    backend, requests = api(monkeypatch, tmp_path, [single])
    one = rj._single_rule_set('outcome_useful_rule', 'Actually answers the question.')
    out = backend.judge('question', RESPONSE, one, {'targeted_refusal_reference_only': True})
    assert out['outcome_useful_rule'] == [1] and len(requests) == 1
    assert 'rule_to_evaluate' in json.loads(requests[0]['messages'][1]['content'])


def test_unknown_program_error_is_not_swallowed(monkeypatch, tmp_path):
    backend, requests = api(monkeypatch, tmp_path, [RuntimeError('program integrity')])
    with pytest.raises(RuntimeError, match='program integrity'):
        backend.judge('question', RESPONSE, RULES)
    assert len(requests) == 1


class ContextRequestRejected(RuntimeError):
    status_code = 400


CONTEXT_ERROR = ("This model's maximum context length is 16384 tokens. However, "
    "you requested 20547 tokens (2115 in the messages, 18432 in the completion).")


def test_real_context400_is_archived_once_and_excluded_without_shortening_or_retry(monkeypatch,tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_MAX_NEW_TOKENS','1024')
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED','1')
    backend,requests=api(monkeypatch,tmp_path,[ContextRequestRejected(CONTEXT_ERROR)])
    monkeypatch.setenv('RUBRIC_JUDGE_API_MAX_RETRIES','3')
    many=rj.RubricRules([],['Unchanged original predicate']*18,[],[])
    out=backend.judge('question',RESPONSE,many)
    assert len(requests)==1 and requests[0]['max_tokens']==18432
    assert out['_judge_failed'] and out['_parse_valid'] is False
    events=[json.loads(s) for p in tmp_path.glob('*.jsonl') for s in p.read_text().splitlines()]
    assert len(events)==1 and events[0]['raw_output'] is None
    audit=events[0]['request_audit'];e=audit['context_capacity_failure']
    assert e['original_error']==CONTEXT_ERROR and e['requested_total_tokens']==20547
    assert e['completion_tokens']==18432 and e['no_verdict'] and not gg.verified_event(events[0])
    from verl.utils.judge_filter import valid_group_indices
    assert valid_group_indices(['bad']*8+['good']*8,[out['_judge_failed']]+[False]*15,8)==(list(range(8,16)),1)


def test_context_unavailability_is_typed_for_whole_validation_without_fabricated_aggregate(monkeypatch,tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_MAX_NEW_TOKENS','1024')
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED','0')
    monkeypatch.setenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE','1')
    backend,requests=api(monkeypatch,tmp_path,[ContextRequestRejected(CONTEXT_ERROR)])
    from verl.utils.validation_recovery import JudgeScoringUnavailable
    with pytest.raises(JudgeScoringUnavailable,match='context capacity'):
        backend.judge('question',RESPONSE,rj.RubricRules([],['Original']*18,[],[]))
    assert len(requests)==1


@pytest.mark.parametrize('message,status,budget', [
    ('unknown request400',400,18432),
    (CONTEXT_ERROR.replace('20547','20548'),400,18432),
    (CONTEXT_ERROR,500,18432),
    (CONTEXT_ERROR,400,512),
])
def test_unknown_or_unverifiable_capacity_error_is_not_swallowed(message,status,budget):
    error=ContextRequestRejected(message);error.status_code=status
    assert gg.context_capacity_failure(error,dict(max_tokens=budget)) is None


def test_multiple_targeted_rules_do_not_recurse_through_grouped_dispatch(monkeypatch, tmp_path):
    single = dict(evidence_line_ids=[1], reason='literal', predicate_truth='TRUE', score=1)
    backend, requests = api(monkeypatch, tmp_path, [single] * 5)
    out = backend.judge('question', RESPONSE, RULES, {'targeted_refusal_reference_only': True})
    assert out['_parse_valid'] and len(requests) == 5
    assert all('rule_to_evaluate' in json.loads(r['messages'][1]['content']) for r in requests)


def test_exhausted_format_is_unavailable_not_a_valid_filled_score(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED', '1')
    backend, requests = api(monkeypatch, tmp_path, [{}])
    result = backend.judge('question', RESPONSE, RULES)
    assert result['_parse_valid'] is False and result['_judge_failed'] is True
    assert len(requests) == 1


def compact(value):
    return [list(row.values()) for row in value.values()]


def test_compact_serialization_preserves_all_four_fields_every_rule_and_original_evidence():
    system,user,guidance,sections=gg.request('question',RESPONSE,RULES,'tuple')
    normal_system,normal_user,_,normal_sections=gg.request('question',RESPONSE,RULES,'object')
    assert user==normal_user and sections==normal_sections
    assert guidance['guided_json']['type']=='array'
    assert guidance['seed']==0 and guidance['chat_template_kwargs']=={'enable_thinking':False}
    assert gg.output(compact(raw()),RULES,sections,'tuple')==gg.output(raw(),RULES,sections)


@pytest.mark.parametrize('bad',['missing_row','missing_field','extra_field','bool_score','contradictory_truth','duplicate_evidence','long_reason'])
def test_compact_format_never_drops_pads_truncates_or_coerces_fields(bad):
    values=compact(raw());_,_,_,sections=gg.request('question',RESPONSE,RULES,'tuple')
    if bad=='missing_row':values.pop()
    if bad=='missing_field':values[0].pop()
    if bad=='extra_field':values[0].append('extra')
    if bad=='bool_score':values[0][3]=True
    if bad=='contradictory_truth':values[0][2]='FALSE'
    if bad=='duplicate_evidence':values[0][0]=[1,1]
    if bad=='long_reason':values[0][1]='x'*601
    with pytest.raises(ValueError):gg.output(values,RULES,sections,'tuple')


def test_actual_compact_api_calls_keep_original_scores_and_verified_independent_conflicts(monkeypatch,tmp_path):
    monkeypatch.setenv('RUBRIC_GROUPED_WIRE_FORMAT','tuple')
    backend,requests=api(monkeypatch,tmp_path,[compact(raw()),compact(raw([1,1,1,1,0]))])
    a=backend.judge('question',RESPONSE,RULES);b=backend.judge('question',RESPONSE,RULES)
    assert a['outcome_useful_rule']==[0,1] and b['outcome_useful_rule']==[1,1]
    assert cq.verified(b,RULES.all_rules) and b['_judge_failed']
    report=cq.audit_directory(tmp_path)
    assert not report['passed'] and all(x['evidence_verified'] for x in report['conflicts'])
    events=[json.loads(line) for p in tmp_path.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert all(cq.verified_audit_event(e) for e in events)
    assert all(e['request_audit']['wire_format']=='tuple' for e in events)
