import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from verl.utils import grouped_grounded_judge as gg
from verl.utils import grouped_dispatch as gd
from verl.utils.reward_score import rubric_judge as rj

ROOT=Path(__file__).resolve().parents[2]


def original_request_function():
    code=subprocess.check_output(['git','show','7b274ec7421467e0d661e81e69ef0544eaf41854:AlphaAlign-Rubric-end/verl/utils/grouped_grounded_judge.py'],cwd=ROOT,text=True)
    scope={'__name__':'original_grouped_request'}
    exec(compile(code,'original_grouped_request','exec'),scope)
    return scope['request']


def fixed_cases():
    from scripts.validate_existing_answer_judge import load_cases
    from scripts import smoke_targeted_refusal as smoke
    return load_cases()+json.loads(smoke.FIXTURE.read_text())['cases']


def test_all36_requests_schemas_and_sections_are_byte_identical_before_after_memoization():
    old=original_request_function()
    gg._system.cache_clear();gg._guidance_json.cache_clear()
    for case in fixed_cases():
        rules=rj._build_rules(case['original_info'])
        before=old(case['prompt'],case['response'],rules)
        after=gg.request(case['prompt'],case['response'],rules)
        assert before==after
        assert json.dumps(before,ensure_ascii=False)==json.dumps(after,ensure_ascii=False)
    assert gg._system.cache_info().currsize==1
    assert 0<gg._guidance_json.cache_info().currsize<=256


def test_syntax_cache_never_retains_or_reuses_a_model_verdict_and_cannot_be_mutated():
    rules=rj.RubricRules([],['Does not add X.'],[],[])
    first=gg.request('q','<answer>one</answer>',rules)
    first[2]['guided_json']['properties'].clear()
    next_value=gg.request('q','<answer>two</answer>',rules)
    assert next_value[2]['guided_json']['properties']
    assert 'two' in next_value[1] and 'one' not in next_value[1]
    assert 'one' not in gg._guidance_json.cache_info().__repr__()


def test_fair_grouped_order_preserves_each_actual_endpoint_fifo_and_all_result_indices(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_LOAD_BALANCE','sticky')
    api=rj.ApiJudge('http://one/v1,http://two/v1,http://three/v1','unused','judge')
    prompts=['q']*72;responses=[str(i) for i in range(72)]
    baseline=list(range(72));fair=gd.ordered_indices(api,prompts,responses,True)
    assert sorted(fair)==baseline
    for route in api.base_urls:
        select=lambda order:[i for i in order if api._base_url_for_request(prompts[i],responses[i],1)==route]
        assert select(fair)==select(baseline)
    assert gd.ordered_indices(api,prompts,responses,False)==baseline


@pytest.mark.parametrize('workers,fair',[(48,'0'),(48,'1'),(72,'1')])
def test_only_predeclared_grouped_profiles_are_supported(monkeypatch,workers,fair):
    monkeypatch.setenv('RUBRIC_GROUPED_API_CONCURRENCY',str(workers));monkeypatch.setenv('RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE',fair)
    assert gd.configuration()==(workers,fair=='1')


@pytest.mark.parametrize('workers,fair',[(96,'1'),(72,'0'),(48,'yes')])
def test_unknown_profile_is_not_silently_enabled(monkeypatch,workers,fair):
    monkeypatch.setenv('RUBRIC_GROUPED_API_CONCURRENCY',str(workers));monkeypatch.setenv('RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE',fair)
    with pytest.raises(ValueError):gd.configuration()


def report():
    reward=dict(score=.5,judge_failed=False,rubric_rule_passes=dict(outcome_safety_rule=[1],outcome_useful_rule=[1],reasoning_process_rule=[],refusal_rule=[],_parse_valid=True))
    return dict(passed=True,rows=[dict(id='original',profile='benign',baseline_reward=copy.deepcopy(reward),damped_reward=copy.deepcopy(reward),prompt_group_sha256='original-group',quarantined=False,effective_update_weight=1,update_advantage=.02)],advantages=[[.02]],presentation_contribution=[[.02]],quarantined_group_count=0,eligible_group_response_count=1,whole_group_filter_executed=True)


def snapshots():
    return {name:dict(report=report(),both_grpo={'G1':[.02],'G3':[.02]},requests=['one','two'],seconds=(100 if name.startswith('baseline') else 80 if name.startswith('fair48') else 90 if name.startswith('compact') else 60),wire_format=gd.WIRE_FORMATS[name.rsplit('_',1)[0]],semantic_requests=['same1','same2'],evidence_complete=True,response_ids=[name+'-1',name+'-2']) for name in gd.ORDER}


def test_selects_fastest_equivalent_measured_profile_not_largest_by_assumption():
    data=snapshots();chosen=gd.choose(data)
    assert chosen['selected_profile']=='fair72' and chosen['selected_grouped_workers']==72
    data['fair72_1']['seconds']=data['fair72_2']['seconds']=90
    assert gd.choose(data)['selected_profile']=='fair48'


@pytest.mark.parametrize('failure',['raw_bit','reward','mask','advantage','g3','request','response_id','evidence','baseline_stability','slow'])
def test_any_difference_missing_evidence_or_no_speed_benefit_prevents_optimization(failure):
    data=snapshots()
    for name in (n for n in gd.ORDER if not n.startswith('baseline')):
        s=data[name]
        if failure=='raw_bit':s['report']['rows'][0]['damped_reward']['rubric_rule_passes']['outcome_useful_rule']=[0]
        if failure=='reward':s['report']['rows'][0]['damped_reward']['score']=.6
        if failure=='mask':s['report']['rows'][0]['effective_update_weight']=0
        if failure=='advantage':s['report']['advantages']=[[.03]]
        if failure=='g3':s['both_grpo']['G3']=[.03]
        if failure=='request':s['requests']=['changed'];s['semantic_requests']=['changed']
        if failure=='response_id':s['response_ids']=['repeated']
        if failure=='evidence':s['evidence_complete']=False
        if failure=='slow':s['seconds']=99
    if failure=='baseline_stability':data['baseline_2']['both_grpo']['G1']=[.03]
    chosen=gd.choose(data)
    assert chosen['selected_profile']=='baseline' and chosen['selected_grouped_workers']==48
    assert not chosen['selected_grouped_fair']


def test_incomplete_optional_probe_retains48():
    assert gd.choose({})['selected_grouped_workers']==48


@pytest.mark.parametrize('changed',[False,True])
def test_one_finite_probe_runs_all8_rounds_with_declared_load_and_rechecks_original_evidence(monkeypatch,tmp_path,changed):
    from scripts import benchmark_grouped_dispatch as probe
    from scripts import validate_grouped_grounded as syntax
    monkeypatch.setattr(syntax,'grammar',lambda *a:dict(passed=True,orchestration_test_double=True))
    from Evaluation.safety.json_evidence import write_json_atomic
    monkeypatch.setenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE','1')
    monkeypatch.setenv('RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT','48')
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY','48')
    backend=rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','unused','judge')
    monkeypatch.setattr(probe.rj,'_get_judge_backend',lambda:backend)
    seen=[]
    def replay(cases,api,path):
        assert len(cases)==72
        assert [{k:v for k,v in c.items() if k!='stress_repetition'} for c in cases[:36]]==[{k:v for k,v in c.items() if k!='stress_repetition'} for c in cases[36:]]
        assert os.environ['RUBRIC_JUDGE_API_CONCURRENCY']=='48'
        seen.append((path.name,int(os.environ['RUBRIC_GROUPED_API_CONCURRENCY']),int(os.environ['RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE'])))
        value=report()
        if changed and path.name=='baseline_2':value['advantages']=[[.03]]
        return value
    monkeypatch.setattr(probe,'reward_replay',replay)
    monkeypatch.setattr(probe,'both_grpo',lambda report:dict(G1=report['advantages'],G3=report['advantages']))
    monkeypatch.setattr(probe,'requests',lambda directory:(['request'],dict(independent_response_ids=[str(directory)+'1'])))
    monkeypatch.setattr(probe,'semantic_requests',lambda directory:['original-task'])
    monkeypatch.setattr(probe.cq,'audit_directory',lambda directory:dict(passed=True))
    values=[];clock=0
    for label in gd.ORDER:
        values.extend([clock,clock+(100 if label.startswith('baseline') else 80 if label.startswith('fair48') else 90 if label.startswith('compact') else 60)])
        clock=values[-1]
    ticks=iter(values);monkeypatch.setattr(probe.time,'monotonic',lambda:next(ticks))
    monkeypatch.setattr(sys,'argv',['probe','--output',str(tmp_path/'load')])
    probe.main()
    selected=json.loads((tmp_path/'load/selection.json').read_text())
    assert seen==[(name,*gd.PROFILES[name.rsplit('_',1)[0]][:1],int(gd.PROFILES[name.rsplit('_',1)[0]][1])) for name in gd.ORDER]
    assert selected['selected_profile']==('baseline' if changed else 'fair72')
    assert selected['real_policy_rollouts'] is False and selected['unique_frozen_answer_count']==36
    assert probe.verified_selection(tmp_path/'load/selection.json')==((48,False,'object') if changed else (72,True,'object'))
    if not changed:
        selected['selected_grouped_workers']=48
        write_json_atomic(tmp_path/'load/selection.json',selected)
        with pytest.raises(ValueError):probe.verified_selection(tmp_path/'load/selection.json')


def test_optional_error_keeps_original48_and_restores_backend(monkeypatch,tmp_path):
    from scripts import benchmark_grouped_dispatch as probe
    from scripts import validate_grouped_grounded as syntax
    monkeypatch.setattr(syntax,'grammar',lambda *a:dict(passed=True,orchestration_test_double=True))
    monkeypatch.setenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE','1')
    monkeypatch.setenv('RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT','48')
    backend=rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','unused','judge')
    monkeypatch.setattr(probe.rj,'_get_judge_backend',lambda:backend)
    def fail(*a):raise ValueError('optional probe evidence unavailable')
    monkeypatch.setattr(probe,'reward_replay',fail)
    monkeypatch.setattr(sys,'argv',['probe','--output',str(tmp_path/'load')])
    probe.main()
    result=json.loads((tmp_path/'load/selection.json').read_text())
    assert result['selected_grouped_workers']==48 and not result['selected_grouped_fair']
    assert result['error_type']=='ValueError' and 'traceback' in result
    assert probe.rj._JUDGE_INSTANCE is backend
    assert probe.verified_selection(tmp_path/'load/selection.json')==(48,False,'object')


def test_persistent_timing_is_bounded_and_never_contains_answers_or_credentials(monkeypatch,tmp_path):
    monkeypatch.setenv('RUBRIC_GROUPED_TIMING_ENABLE','1');monkeypatch.setenv('RUBRIC_RUN_DIR',str(tmp_path))
    monkeypatch.setenv('RUBRIC_GLOBAL_STEP','91');monkeypatch.setenv('RUBRIC_EVOLVE','1')
    api=rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','credential-must-never-appear','judge')
    observation=dict(endpoint_port=8000,error_type=None,local_queue_wait_seconds=.1,http_seconds=2.,preprocessing_seconds=.002,token_usage={'prompt_tokens':30,'completion_tokens':5,'total_tokens':35})
    for _ in range(10):gd.publish_timing(api,64,[observation],2.1)
    files=list((tmp_path/'throughput').glob('*.json'))
    primary=[p for p in files if not p.name.endswith('.integrity.json')]
    assert len(primary)==5
    values=[json.loads(p.read_text()) for p in primary]
    assert all(v['response_rows']==64 and v['tokens']['completion_tokens']['total']==5 for v in values)
    assert all('credential-must-never-appear' not in p.read_text() for p in files)
    monkeypatch.setenv('RUBRIC_GLOBAL_STEP','120');gd.publish_timing(api,304,[observation],2.1)
    assert len([p for p in (tmp_path/'throughput').glob('*.json') if not p.name.endswith('.integrity.json')])==6


def test_actual_grouped_pool_returns64_in_original_order_despite_out_of_order_completion(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE','1')
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE','1')
    monkeypatch.setenv('RUBRIC_GROUPED_API_CONCURRENCY','72')
    monkeypatch.setenv('RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE','1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY','48')
    api=rj.ApiJudge('http://a/v1,http://b/v1,http://c/v1','unused','judge')
    observed=[]
    def judgement(prompt,response,rules,info):
        value=int(response);assert info['uid']==value
        time.sleep(.002*(value%3))
        observed.append(value)
        gd.observe(dict(http_seconds=.002,token_usage={}))
        return dict(outcome_safety_rule=[],outcome_useful_rule=[value%2],reasoning_process_rule=[],refusal_rule=[],_parse_valid=True)
    monkeypatch.setattr(api,'judge',judgement)
    rules=rj.RubricRules([],['unchanged'],[],[])
    outputs=api.judge_batch(['q']*64,[str(i) for i in range(64)],[rules]*64,[{'uid':i} for i in range(64)])
    assert [r['outcome_useful_rule'][0] for r in outputs]==[i%2 for i in range(64)]
    assert sorted(observed)==list(range(64))
    assert os.environ['RUBRIC_JUDGE_API_CONCURRENCY']=='48'
    assert gd._OBSERVER.get() is None


def test_both_actual_grpo_formulas_keep_group_exclusion_and_expression_bound_for_synthetic_numbers():
    # Synthetic numeric tensors only: these are not historical model responses.
    from scripts.benchmark_grouped_dispatch import both_grpo
    from verl.utils import presentation_reward as pr
    rows=[]
    for i in range(16):
        component=.02 if i%8>=4 else 0.
        reward={'score':.5+component,pr.COMPONENT:component,pr.SCORED:.5+component,pr.CONTENT:.5}
        rows.append(dict(prompt_group_sha256='excluded' if i<8 else 'valid',quarantined=i<8,damped_reward=reward))
    report={'rows':rows};before=copy.deepcopy(report);result=both_grpo(report)
    assert report==before
    for mode in ['G1','G3']:
        assert result[mode]['eligible_indices']==list(range(8,16))
        assert len(result[mode]['advantages'])==8
        assert max(map(abs,result[mode]['presentation_contribution']))<=.050001
    assert result['G1']['std_floor']==0. and result['G3']['std_floor']==.25


def test_compact_wire_may_differ_only_when_original_task_raw_verdicts_rewards_and_updates_match():
    data=snapshots()
    for key in ['compact72_1','compact72_2']:
        data[key]['requests']=['actual compact request one','actual compact request two']
        data[key]['seconds']=40
    chosen=gd.choose(data)
    assert chosen['selected_profile']=='compact72' and chosen['selected_wire_format']=='tuple'
    assert chosen['candidate_checks']['compact72']['identical_raw_http_request_multiset'] is False
    assert chosen['candidate_checks']['compact72']['intended_wire_format_change'] is True
    data['compact72_2']['semantic_requests']=['changed original rule']
    assert gd.choose(data)['selected_profile']=='fair72'
