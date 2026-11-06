from copy import deepcopy
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from verl.utils import consistency_quarantine as cq
from verl.utils import targeted_refusal as tr
from verl.utils.judge_filter import valid_group_indices
from verl.utils.reward_score import rubric_judge as rj


def backend(monkeypatch, scores):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY', '1')
    calls = []
    bits = iter(scores)
    def create(**request):
        calls.append(request)
        score = next(bits)
        raw = json.dumps(dict(evidence_line_ids=[1], reason='Independent original observation',
                              predicate_truth='TRUE' if score else 'FALSE', score=score))
        return SimpleNamespace(id=f'independent-{len(calls)}', model='judge', choices=[
            SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    return rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge'), calls


def test_conflict_preserves_current_bits_and_both_originals_without_retry(monkeypatch):
    api, calls = backend(monkeypatch, [0, 1, 0])
    rule = rj.RubricRules([], ['The answer supplies relevant information.'], [], [])
    first = api.judge('q', '<answer>original</answer>', rule)
    second = api.judge('q', '<answer>original</answer>', rule)
    third = api.judge('q', '<answer>original</answer>', rule)
    assert first['outcome_useful_rule'] == [0] and not first.get('_judge_failed')
    assert second['outcome_useful_rule'] == [1] and second['_parse_valid']
    assert third['outcome_useful_rule'] == [0] and third['_parse_valid']
    assert second['_judge_failed'] and third['_judge_failed']
    assert cq.verified(second, rule.all_rules) and cq.verified(third, rule.all_rules)
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    pair = second[cq.KEY][0]['conflict']
    assert pair['first']['response_id'] != pair['current']['response_id']
    assert pair['first']['score'] == 0 and pair['current']['score'] == 1
    assert api.grounded_consistency_summary()['passed'] is False
    assert api.grounded_consistency_summary()['cached_verdicts_returned'] == 0


def test_actual_batch_returns_other_groups_and_filter_removes_entire_conflicted_group(monkeypatch):
    api, calls = backend(monkeypatch, [1, 0, 1, 1])
    rule = rj.RubricRules([], ['The answer supplies relevant information.'], [], [])
    outputs = api.judge_batch(['a', 'a', 'b', 'b'], ['<answer>original</answer>'] * 4,
                              [rule] * 4)
    assert len(outputs) == 4 and len(calls) == 4
    assert [out['outcome_useful_rule'][0] for out in outputs] == [1, 0, 1, 1]
    assert all(out['_parse_valid'] for out in outputs)
    assert cq.verified(outputs[1], rule.all_rules)
    keep, dropped = valid_group_indices(['a', 'a', 'b', 'b'],
                                        [out.get('_judge_failed', False) for out in outputs])
    assert keep == [2, 3] and dropped == 1
    assert not any(outputs[i].get('_judge_failed') for i in keep)


@pytest.mark.parametrize('batch_api', [False, True])
def test_conflict_proof_survives_original_multi_rule_index_alignment(monkeypatch, batch_api):
    api, _ = backend(monkeypatch,[1,0,1,1])
    rules = rj.RubricRules([],['First unchanged proposition.','Second unchanged proposition.'],[],[])
    def call():
        if batch_api:return api.judge_batch(['q'],['<answer>original</answer>'],[rules])[0]
        return api.judge('q','<answer>original</answer>',rules)
    first, second = call(), call()
    assert first['outcome_useful_rule'] == [1,0]
    assert second['outcome_useful_rule'] == [1,1] and second['_judge_failed']
    assert second[cq.KEY][0]['rule_index'] == 1 and cq.verified(second,rules.all_rules)


@pytest.mark.parametrize('tamper', ['score', 'same_id', 'request', 'rule', 'original_disagreement'])
def test_quarantine_proof_cannot_be_forged_or_misaligned(monkeypatch, tamper):
    api, _ = backend(monkeypatch, [0, 1, 0])
    rule = rj.RubricRules([], ['The answer supplies relevant information.'], [], [])
    api.judge('q', '<answer>original</answer>', rule)
    out = api.judge('q', '<answer>original</answer>', rule)
    if tamper == 'original_disagreement':
        out = api.judge('q', '<answer>original</answer>', rule)
    out = deepcopy(out);proof = out[cq.KEY][0]
    if tamper == 'score': out['outcome_useful_rule'][0] = 0
    if tamper == 'same_id': proof['conflict']['current']['response_id'] = proof['conflict']['first']['response_id']
    if tamper == 'request': proof['request']['user'] += 'changed'
    if tamper == 'rule': proof['rule'] = 'a different rule'
    if tamper == 'original_disagreement': proof['conflict']['original_disagreement']['current']['raw_output'] = '{}'
    assert not cq.verified(out, rule.all_rules)


def test_conflict_cannot_create_positive_targeted_correction(monkeypatch):
    api, _ = backend(monkeypatch, [0, 1])
    rule = rj.RubricRules([], ['The answer supplies relevant information.'], [], [])
    api.judge('q', '<answer>original</answer>', rule)
    out = api.judge('q', '<answer>original</answer>', rule)
    role, reason = tr.classify(out, rule, dict(parse_valid=True, substantive_answer=1,
                                              coherent_answer=1, unnecessary_refusal=0))
    assert role == 0 and reason == 'failed_judge_group_quarantine'


@pytest.mark.parametrize('mode,floor', [('legacy','0'), ('bounded_residual','.25')])
def test_actual_online_wrapper_and_grpo_exclude_conflict_without_ending_batch(monkeypatch, tmp_path, mode, floor):
    from scripts import graded_presentation_gate as gate
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as evolve
    api, calls = backend(monkeypatch, [])
    counts = {}
    def create(**request):
        calls.append(request)
        user = json.loads(request['messages'][1]['content'])
        key = request['messages'][1]['content'];counts[key] = counts.get(key, 0) + 1
        rule = user['rule_to_evaluate']
        text = ''.join(row['text'] for row in user['assistant_evidence_lines'])
        score = 0 if 'refusal' in rule or 'short' in text else 1
        if user['question_context'] == 'conflict-group' and 'first' in text and 'relevant' in rule:
            score = int(counts[key] == 1)
        raw = json.dumps(dict(evidence_line_ids=[1], reason='Independent original observation',
                              predicate_truth='TRUE' if score else 'FALSE', score=score))
        return SimpleNamespace(id=f'live-{len(calls)}', model='judge', choices=[
            SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    sys.modules['openai'].OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    for key,value in dict(REWARD_MODE='rubric_safe_helpful_balanced', RUBRIC_GUARD_ENABLE='0',
        RUBRIC_EVOLVE='1', RUBRIC_EVOLVE_START_EPOCH='1', RUBRIC_EVOLVE_START_STEP='360',
        RUBRIC_EVOLVE_CACHE_DIR=str(tmp_path/'cache'), RUBRIC_EVOLVE_LOG_DIR=str(tmp_path/'logs'),
        RUBRIC_EVOLVE_REPLAY_ENABLE='0', RUBRIC_EVOLVE_ALLOWED_PROFILES='benign',
        RUBRIC_EVOLVE_REWARD_MODE=mode, RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,
        RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='0', RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',
        RUBRIC_EVOLVE_GROUP_RANK_GUARD='0', RUBRIC_EVOLVE_SAVE_DIR=str(tmp_path/'evolve'),
        RUBRIC_TARGETED_REFUSAL_ENABLE='0', RUBRIC_PRESENTATION_DAMPING_ENABLE='1',
        RUBRIC_PRESENTATION_REWARD_FACTOR='.2', RUBRIC_PRESENTATION_ADV_CAP='.05',
        GLOBAL_REASONING_RULE_ENABLE='0', RUBRIC_FORMAT_WEIGHT='.1',
        RUBRIC_FORMAT_REQUIRE_THINK='0', RUBRIC_LENGTH_PENALTY_WEIGHT='0',
        RUBRIC_CRITICAL_FAILURE_GATE='off', RUBRIC_BASE_JUDGE_TRACE_DIR=str(tmp_path/'attempts')).items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(rj,'_get_judge_backend',lambda:api)
    monkeypatch.setattr(evolve,'_get_base_guard_impl',lambda:rj)
    rule = rj.RubricRules(['The answer is safe.'], ['The answer gives relevant information.'], [],
                          ['The answer is a refusal.'])
    cases = [dict(id=f'synthetic_{i}',prompt=prompt,response=f'<answer>{answer}</answer>',
        original_info=dict(rule.all_rules,uid=prompt),profile='benign',replay_cohort='original_twenty')
        for i,(prompt,answer) in enumerate([('conflict-group','first'),('conflict-group','second'),
                                          ('valid-group','helpful'),('valid-group','short')])]
    result = gate.reward_replay(cases,api,tmp_path)
    assert result['passed'] and result['case_count'] == 4
    assert result['online_evolution_wrapper_executed'] and result['actual_advantage_dispatch_executed']
    assert result['whole_group_filter_executed'] and result['quarantined_group_count'] == 1
    first,companion = result['rows'][:2]
    assert first['baseline_reward']['rubric_rule_passes']['outcome_useful_rule'] == [1]
    assert first['damped_reward']['rubric_rule_passes']['outcome_useful_rule'] == [0]
    assert first['baseline_reward']['score'] > first['damped_reward']['score']
    assert not first['raw_arrays_equal']
    assert all(row['quarantined'] and row['effective_update_weight'] == row['update_advantage'] == 0
               for row in [first,companion])
    assert all(not row['quarantined'] for row in result['rows'][2:])
    assert result['rows'][2]['update_advantage'] > 0 > result['rows'][3]['update_advantage']
    audit = gate.cross_request_audit(tmp_path)
    assert not audit['passed'] and audit['independent_repeats_checked'] > 0
    assert cq.containment_verified(audit,result['rows'])
    changed = deepcopy(result['rows']);changed[0]['effective_update_weight'] = 1
    assert not cq.containment_verified(audit,changed)


def test_unknown_or_cached_conflict_evidence_cannot_pass_containment():
    rows = [dict(prompt_sha256='q',quarantined=True,effective_update_weight=0,update_advantage=0)]
    audit = dict(conflicts=[dict(evidence_verified=False,first=dict(prompt_sha256='q'),
                                 current=dict(prompt_sha256='q'))])
    assert not cq.containment_verified(audit,rows)


def archived_targeted_pair():
    path = Path(__file__).parents[1] / 'fixtures/targeted_audit_conflict_20261031.json'
    return json.loads(path.read_text())['events']


def test_real_targeted_trace_hash_only_identity_and_whole_group_containment(tmp_path):
    first, current = archived_targeted_pair()
    assert 'prompt' not in first and 'prompt' not in current
    assert first['request_audit']['messages'] == current['request_audit']['messages']
    assert json.loads(first['raw_output'])['score'] == 1
    assert json.loads(current['raw_output'])['score'] == 0
    assert cq.verified_audit_event(first) and cq.verified_audit_event(current)
    (tmp_path / 'events.jsonl').write_text('\n'.join(map(json.dumps, [first, current])))
    audit = cq.audit_directory(tmp_path)
    assert not audit['passed'] and audit['independent_repeats_checked'] == 1
    assert audit['cached_verdicts_returned'] == 0
    assert len(audit['conflicts']) == 1 and audit['conflicts'][0]['evidence_verified']
    prompt = first['prompt_sha256']
    rows = [dict(prompt_sha256=prompt,quarantined=True,effective_update_weight=0,update_advantage=0)] * 2
    assert cq.tainted_prompts(audit) == {prompt}
    assert cq.containment_verified(audit, rows)
    assert not cq.containment_verified(audit, [])
    assert not cq.containment_verified(audit, rows + [dict(prompt_sha256=prompt,
        quarantined=False,effective_update_weight=1,update_advantage=.1)])


@pytest.mark.parametrize('tamper', ['prompt_hash','original_hash','question_hash','context',
    'raw_verdict','response_id','event_type','missing_context','wrapped_unknown'])
def test_archived_targeted_identity_cannot_be_guessed_or_bypassed(tamper):
    event = deepcopy(archived_targeted_pair()[0]); audit = event['request_audit']
    if tamper == 'prompt_hash':event['prompt_sha256'] = 'changed'
    if tamper == 'original_hash':audit['original_prompt_sha256'] = 'changed'
    if tamper == 'question_hash':audit['question_context_sha256'] = 'changed'
    if tamper == 'raw_verdict':event['raw_output'] = '{}'
    if tamper == 'response_id':audit['response_id'] = ''
    if tamper == 'event_type':event['event'] = 'base_judge_attempt'
    if tamper in ('context','missing_context'):
        payload = json.loads(audit['messages'][1]['content'])
        if tamper == 'context':payload['question_context'] += 'changed'
        else:payload.pop('question_context')
        audit['messages'][1]['content'] = json.dumps(payload)
        audit['user_sha256'] = cq.fingerprint(audit['messages'][1]['content'])
    if tamper == 'wrapped_unknown':
        event['prompt_sha256'] = audit['original_prompt_sha256'] = cq.fingerprint('unknown original wrapper')
    assert not cq.verified_audit_event(event)


def test_new_targeted_writer_retains_full_original_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_TARGETED_JUDGE_TRACE_DIR',str(tmp_path))
    info = dict(targeted_refusal_diagnostic_only=True)
    rj._targeted_attempt('unchanged original','<answer>original</answer>',
        rj.RubricRules([],['original rule'],[],[]),info,'{}','judge',1,'stop',True)
    event = json.loads(next(tmp_path.glob('*.jsonl')).read_text())
    assert event['prompt'] == 'unchanged original'
    assert event['prompt_sha256'] == cq.fingerprint(event['prompt'])


@pytest.mark.parametrize('index', [0, 1])
def test_real_recorded_conflict_pairs_replay_without_selecting_a_verdict(monkeypatch, index):
    fixture = Path(__file__).parents[1]/'fixtures/judge_conflicts_20261031.json'
    event = json.loads(fixture.read_text())['events'][index]
    assert cq.verified_audit_event(event)
    api, calls = backend(monkeypatch, [])
    pair = event['request_audit']['consistency']; observations = iter([pair['first'],pair['current']])
    def create(**request):
        calls.append(request);observed = next(observations)
        return SimpleNamespace(id=observed['response_id'],model=event['model'],choices=[
            SimpleNamespace(message=SimpleNamespace(content=observed['raw_output']),finish_reason='stop')])
    sys.modules['openai'].OpenAI = lambda **kwargs:SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=create)))
    rules = rj.RubricRules(**event['rules'])
    first = api.judge(event['prompt'],event['response'],rules)
    second = api.judge(event['prompt'],event['response'],rules)
    category = next(k for k,v in rules.all_rules.items() if v)
    assert first[category] == [pair['first']['score']]
    assert second[category] == [pair['current']['score']]
    assert second['_judge_failed'] and second['_parse_valid'] and cq.verified(second,rules.all_rules)
    proof = second[cq.KEY][0]['conflict']
    assert proof['first']['raw_output'] == pair['first']['raw_output']
    assert proof['current']['raw_output'] == pair['current']['raw_output']
    assert len(calls) == 2 and calls[0] == calls[1]


@pytest.mark.parametrize('failure', [None, 'wrong_mask', 'wrong_uids', 'missing_original', 'unbounded'])
def test_first_batch_preserves_all_64_rows_but_only_updates_aligned_valid_groups(monkeypatch, tmp_path, failure):
    import ast
    import os
    import numpy as np
    import torch
    from verl.utils.training_evidence import atomic_json
    root = Path(__file__).parents[2]
    source = root/'verl/trainer/ppo/ray_trainer.py'
    function = next(node for node in ast.parse(source.read_text()).body
                    if isinstance(node,ast.FunctionDef) and node.name == '_write_training_batch_gate')
    env = dict(np=np,torch=torch,os=os,atomic_json=atomic_json,__file__=str(source))
    exec(compile(ast.Module(body=[function],type_ignores=[]),'actual_first_batch_gate','exec'),env)
    monkeypatch.setenv('RUBRIC_RUN_DIR',str(tmp_path))
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE','1')
    uids = [f'q{i//8}' for i in range(64)]
    traces = [dict(judge_parse_valid=True,judge_failed=i==0,score=.5) for i in range(64)]
    kept = list(range(8,64))
    context = dict(uids=uids,traces=traces,failed=[i==0 for i in range(64)],keep=kept,actor_world_size=4)
    class Batch:
        batch = {k:torch.ones(56,1) for k in ['token_level_scores','token_level_rewards','advantages']}
        non_tensor_batch = dict(uid=uids[8:],presentation_advantage_contribution=[.001]*56)
        def __len__(self):return 56
    batch = Batch()
    if failure == 'wrong_mask':context['failed'] = [False]*64
    if failure == 'wrong_uids':batch.non_tensor_batch['uid'] = uids[:56]
    if failure == 'missing_original':context['traces'] = traces[8:]
    if failure == 'unbounded':batch.non_tensor_batch['presentation_advantage_contribution'][0] = .06
    call = lambda:env[function.name](batch,traces[8:],1,dict(train=3120,val=304),context)
    if failure is None:
        report = call()
        assert report['passed'] and report['response_count'] == 64 and report['prompt_count'] == 8
        assert report['eligible_response_count'] == 56 and report['eligible_prompt_count'] == 7
        assert len(report['reward_traces']) == 64 and len(report['eligible_reward_traces']) == 56
        assert report['group_filter_check']['quarantined_group_count'] == 1
        assert report['optimizer_updates'] == 0
    else:
        with pytest.raises(ValueError):call()
