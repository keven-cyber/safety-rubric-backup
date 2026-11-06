import copy
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from types import ModuleType, SimpleNamespace
import sys

import pytest

from verl.utils.judge_dispatch import interleave, choose_dispatch
from verl.utils.reward_score import rubric_judge as rj


def test_interleave_keeps_every_request_and_endpoint_fifo():
    routes = ['a'] * 100 + ['b'] * 37 + ['c'] * 90
    order = interleave(range(len(routes)), routes.__getitem__)
    assert sorted(order) == list(range(len(routes)))
    assert [routes[i] for i in order[:6]] == ['a', 'b', 'c'] * 2
    for route in set(routes):
        assert [i for i in order if routes[i] == route] == [i for i,r in enumerate(routes) if r == route]
    assert interleave([], lambda _: None) == []


@pytest.mark.parametrize('grounded', [False, True])
def test_routing_prediction_is_the_actual_scoped_sticky_route(monkeypatch, grounded):
    monkeypatch.setenv('RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', str(int(grounded)))
    api = rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1', 'unused', 'unchanged')
    specs, actual = [], []
    for i in range(40):
        response = f'<think>reason{i}</think><answer>answer{i}</answer>'
        for category in ('reasoning_process_rule', 'outcome_useful_rule'):
            specs.append((i, category, 0, str(i), response, rj._single_rule_set(category, 'unchanged'), None))
            scoped = response
            if grounded:
                think, answer = rj._split_think_answer(response, strict_final=True)
                scoped = f'<think>{think}</think>' if category == 'reasoning_process_rule' else f'<answer>{answer}</answer>'
            actual.append(api._base_url_for_request(str(i), scoped, 1))
    order = api._dispatch_indices(specs)
    for endpoint in api.base_urls:
        assert [i for i in order if actual[i] == endpoint] == [i for i in range(len(specs)) if actual[i] == endpoint]
    monkeypatch.setenv('RUBRIC_JUDGE_LOAD_BALANCE', 'round_robin')
    assert api._dispatch_indices(specs) == list(range(len(specs)))
    assert api._base_url_idx == 0  # Prediction never advances the live round-robin cursor.


def sample_report():
    reward = dict(score=.595, rubric_rule_passes=dict(outcome_safety_rule=[1],
        outcome_useful_rule=[0], reasoning_process_rule=[], refusal_rule=[], _parse_valid=True),
        presentation_reward_component=.1, presentation_scored_reward=.595,
        presentation_content_reward=.495)
    return dict(passed=True, rows=[dict(id='original', profile='safe', baseline_reward=reward,
        damped_reward=copy.deepcopy(reward), quarantined=False, effective_update_weight=1,
        update_advantage=.125)], advantages=[[.125]], presentation_contribution=[.02],
        quarantined_group_count=0)


@pytest.mark.parametrize('mutation', [
    lambda r: r['rows'][0]['damped_reward']['rubric_rule_passes']['outcome_useful_rule'].__setitem__(0,1),
    lambda r: r['rows'][0]['damped_reward'].__setitem__('score',.596),
    lambda r: r['rows'][0].__setitem__('effective_update_weight',0),
    lambda r: r['rows'][0].__setitem__('quarantined',True),
    lambda r: r['rows'][0].__setitem__('update_advantage',.124),
    lambda r: r.__setitem__('passed',False),
    lambda r: r['advantages'][0].__setitem__(0,.126),
    lambda r: r['rows'][0]['damped_reward'].__setitem__('presentation_content_reward',.496),
])
def test_speed_never_substitutes_for_identical_raw_verdicts_and_updates(mutation):
    baseline = sample_report(); candidate = copy.deepcopy(baseline); mutation(candidate)
    selection = choose_dispatch(baseline,candidate,['req','req'],['req','req'],90,30,True)
    assert selection['selected_worker_concurrency'] == 48
    assert not selection['raw_verdict_reward_update_equal']


def test_candidate_requires_independent_complete_equal_requests_and_actual_benefit():
    report = sample_report()
    for requests, seconds, complete in [(['req'],30,True),(['req','req'],30,False),
                                        (['req','req'],89,True)]:
        assert choose_dispatch(report,report,['req','req'],requests,90,seconds,complete)['selected_worker_concurrency'] == 48
    selection = choose_dispatch(report,report,['req','req'],['req','req'],90,30,True)
    assert selection['selected_worker_concurrency'] == 144
    assert selection['baseline_over_candidate_speedup'] == 3
    assert selection['optimizer_updates'] == 0 and not selection['verdict_cache_enabled']


@pytest.mark.parametrize('changed', [False,True])
def test_real_probe_orchestration_uses_all_four_fixed_runs_and_defaults_safely(monkeypatch, tmp_path, changed):
    # Test doubles exercise orchestration only, never evidence of GPU equivalence.
    from scripts import benchmark_judge_dispatch as probe
    backend=rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','unused','unchanged')
    monkeypatch.setattr(probe.rj,'_get_judge_backend',lambda:backend)
    snapshots=[]
    def replay(cases, api, path):
        assert len(cases)==36
        snapshots.append((path.name,int(probe.os.environ['RUBRIC_JUDGE_API_CONCURRENCY']),
                          int(probe.os.environ['RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE'])))
        report=sample_report()
        if changed and path.name=='baseline_2':report['rows'][0]['damped_reward']['score']=.594
        return report
    monkeypatch.setattr(probe,'reward_replay',replay)
    monkeypatch.setattr(probe,'requests',lambda p:(['req','req'],dict(independent_response_ids=[str(p)+'0',str(p)+'1'])))
    monkeypatch.setattr(probe.cq,'audit_directory',lambda p:dict(passed=True))
    times=iter([0,100,100,140,140,180,180,280]);monkeypatch.setattr(probe.time,'monotonic',lambda:next(times))
    monkeypatch.setattr(sys,'argv',['probe','--output',str(tmp_path/'real_probe_test_doubles')])
    probe.main()
    selection=json.loads((tmp_path/'real_probe_test_doubles/selection.json').read_text())
    assert snapshots==[('baseline_1',48,0),('candidate_1',144,1),('candidate_2',144,1),('baseline_2',48,0)]
    assert selection['selected_worker_concurrency']==(48 if changed else 144)
    assert selection['predeclared_order']==[v[0] for v in snapshots]


@pytest.mark.parametrize('workers,fair', [(48,False),(144,True)])
def test_actual_dispatch_keeps_payloads_and_raw_arrays_with_bounded_endpoint_admission(monkeypatch, workers, fair):
    monkeypatch.setenv('RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE', str(int(fair)))
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY', str(workers))
    monkeypatch.setenv('RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT','48')
    monkeypatch.setenv('RUBRIC_JUDGE_GRANULARITY','single_rule')
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE','0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE','0')
    monkeypatch.setenv('RUBRIC_JUDGE_GUIDED_OUTPUT','0')
    api = rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','unused','unchanged')
    active = Counter(); peaks = Counter(); calls=[]; lock=threading.Lock(); release=threading.Event()
    inputs=[]
    # Exactly96 requests per original sticky endpoint; the scheduler must not reroute.
    for endpoint in api.base_urls:
        prompt=next(str(i) for i in range(1000) if api._base_url_for_request(str(i),'answer',1)==endpoint)
        inputs.extend([prompt]*96)
    def client(**options):
        endpoint=options['base_url']
        def create(**request):
            with lock:
                active[endpoint]+=1; peaks[endpoint]=max(peaks[endpoint],active[endpoint]);calls.append((endpoint,request))
                if sum(active.values())==workers:release.set()
            assert release.wait(5), 'Dispatch workers failed to fill their declared capacity'
            with lock:active[endpoint]-=1
            output=dict(outcome_safety_rule=[],outcome_useful_rule=[1],reasoning_process_rule=[],refusal_rule=[])
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(output)),finish_reason='stop')],usage=SimpleNamespace(prompt_tokens=10,completion_tokens=5,total_tokens=15))
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    fake=ModuleType('openai');fake.OpenAI=client;monkeypatch.setitem(sys.modules,'openai',fake)
    rules=rj.RubricRules([],['Unchanged predicate'],[],[])
    result=api.judge_batch(inputs,['answer']*len(inputs),[rules]*len(inputs))
    assert len(result)==len(inputs)==len(calls)
    assert all(r['outcome_useful_rule']==[1] and r['_parse_valid'] for r in result)
    assert max(peaks.values())<=48
    for endpoint,request in calls:
        assert request['model']=='unchanged' and request['temperature']==0
        assert 'Unchanged predicate' in request['messages'][1]['content']
        prompt=request['messages'][1]['content'].split('User request:\n',1)[1].split('\n',1)[0]
        assert endpoint==api._base_url_for_request(prompt,'answer',1)
    if fair:assert all(peaks[e]==48 for e in api.base_urls)
