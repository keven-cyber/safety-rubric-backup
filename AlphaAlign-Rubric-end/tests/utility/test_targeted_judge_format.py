"""Transport failures are not semantic verdicts or permission to drop cases."""
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from verl.utils.reward_score import rubric_judge as rj

ROOT = Path(__file__).resolve().parents[2]


def smoke():
    spec = importlib.util.spec_from_file_location('format_smoke', ROOT / 'scripts/smoke_targeted_refusal.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verdict(rules):
    return {k: [1] * len(v) for k, v in rules.all_rules.items()}


@pytest.mark.parametrize('mutation', [
    lambda o: o.update(extra=1),
    lambda o: o.pop('refusal_rule'),
    lambda o: o.update(outcome_safety_rule=[1]),
    lambda o: o.update(outcome_useful_rule=[True]),
    lambda o: o.update(outcome_useful_rule=['1']),
    lambda o: o.update(outcome_useful_rule=[2]),
    lambda o: o.update(outcome_useful_rule=[]),
])
def test_scoped_schema_never_coerces_or_ignores_malformed_values(mutation):
    rules = rj.RubricRules([], ['Useful'], [], [])
    output = verdict(rules)
    mutation(output)
    with pytest.raises(ValueError):
        rj._targeted_output(output, rules)


def test_guidance_retry_trace_and_base_reward_isolation(monkeypatch, tmp_path):
    requests = []
    rules = rj.RubricRules([], ['Useful'], [], [])
    good = json.dumps(verdict(rules))
    answers = iter(['not JSON', good, good])
    def create(**kw):
        requests.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=next(answers)), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_RETRY_BACKOFF', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_GUIDED_OUTPUT', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_GRANULARITY', 'single_rule')
    monkeypatch.setenv('RUBRIC_TARGETED_JUDGE_TRACE_DIR', str(tmp_path))
    api = rj.ApiJudge('http://localhost:8000/v1', 'secret-not-for-evidence', 'judge')
    meta = dict(targeted_refusal_reference_only=True, validation_case_ids=['kept_case'],
                validation_request_id=7, validation_phase='reference')
    assert api.judge_batch(['p'], ['a'], [rules], [meta])[0]['_parse_valid']
    assert api.judge_batch(['p'], ['a'], [rules], [{}])[0]['_parse_valid']
    for request in requests[:2]:
        choices = [json.loads(s) for s in request['extra_body']['guided_choice']]
        assert [o['outcome_useful_rule'] for o in choices] == [[0], [1]]
    assert 'previous output did not match' in requests[1]['messages'][0]['content']
    assert 'extra_body' not in requests[2]
    events = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.read_text().splitlines()]
    assert len(events) == 2
    assert events[0]['raw_output'] == 'not JSON' and events[0]['error_type'] == 'JSONDecodeError'
    assert events[1]['error_type'] is None
    assert all(e['case_ids'] == ['kept_case'] and e['request_id'] == 7
               and e['category'] == 'outcome_useful_rule' and e['rule_index'] == 0 for e in events)
    assert 'secret-not-for-evidence' not in json.dumps(events)


def test_validation_continues_all_twenty_cases_but_exhaustion_still_fails(monkeypatch, tmp_path):
    module = smoke()
    data = json.loads(module.FIXTURE.read_text())
    class FailingJudge:
        model = 'real-model-test-double'
        def judge_batch(self, prompts, responses, rules_list, infos):
            raise RuntimeError('secret-must-not-be-recorded')
    wrapper = module.CaseRecordingJudge(FailingJudge(), data['cases'], tmp_path / 'cases')
    monkeypatch.setenv('RUBRIC_TARGETED_REFUSAL_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    result = module.run_gate(data, wrapper, rj.RubricRules)
    assert not result['passed'] and result['case_count'] == 20
    assert set(result['failed_case_ids']) == {c['id'] for c in data['cases']}
    assert result['evidence_complete'] and result['request_errors']
    assert {cid for r in wrapper.records for cid in r['case_ids']} == {c['id'] for c in data['cases']}
    assert all(v is None for record in wrapper.records for k, values in record['output'].items()
               if not k.startswith('_') for v in values)
    assert len(list((tmp_path / 'cases').glob('*.json'))) == len(wrapper.records)
    assert 'secret-must-not-be-recorded' not in json.dumps(wrapper.records)


def test_only_one_exhausted_case_does_not_erase_other_verdicts(tmp_path):
    module = smoke()
    data = json.loads(module.FIXTURE.read_text())
    cases = data['cases']
    rules = [rj.RubricRules([], ['Useful'], [], [])] * 20
    class Judge:
        model = 'test-double'
        def judge_batch(self, ps, rs, rules_list, infos):
            if infos[0]['validation_request_id'] == 7:
                raise RuntimeError('exhausted')
            return [dict(verdict(rules_list[0]), _parse_valid=True)]
    wrapper = module.CaseRecordingJudge(Judge(), cases, tmp_path)
    outputs = wrapper.judge_batch([c['prompt'] for c in cases], [c['response'] for c in cases], rules)
    assert len(outputs) == 20 and len(wrapper.records) == 20 and len(wrapper.errors) == 1
    assert outputs[7]['_judge_failed']
    assert sum(o['_parse_valid'] for o in outputs) == 19


@pytest.mark.parametrize('flag', ['targeted_refusal_reference_only', 'targeted_refusal_diagnostic_only'])
def test_targeted_granularity_isolated_from_historical_all_rules(monkeypatch, flag):
    monkeypatch.setenv('RUBRIC_JUDGE_GRANULARITY', 'all_rules')
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY', '1')
    api = rj.ApiJudge('http://localhost:8000/v1', 'EMPTY', 'judge')
    rules = rj.RubricRules(['safe'], ['useful', 'coherent'], [], ['refusal'])
    metadata = {flag: True, 'uid': 'preserved'}
    calls = []
    def score(prompt, response, requested, extra):
        calls.append((prompt, requested, extra))
        return dict(verdict(requested), _parse_valid=True)
    monkeypatch.setattr(api, 'judge', score)
    outputs = api.judge_batch(['base', 'target', 'base2'], ['a'] * 3, [rules] * 3,
                              [{}, metadata, {}])
    assert outputs == [dict(verdict(rules), _parse_valid=True)] * 3
    assert [(p, sum(map(len, r.all_rules.values()))) for p, r, _ in calls] == [
        ('base', 4), ('base2', 4), ('target', 1), ('target', 1), ('target', 1), ('target', 1)]
    assert metadata == {flag: True, 'uid': 'preserved'}
    assert [x[2]['targeted_rule_index'] for x in calls[2:]] == [0, 0, 1, 0]
    with pytest.raises(ValueError, match='Incomplete Judge batch'):
        api.judge_batch(['base', 'missing'], ['a'], [rules, rules])


def test_direct_targeted_call_uses_single_rule_contract_and_audits_endpoint(monkeypatch, tmp_path):
    requests = []
    def create(**kw):
        requests.append(kw)
        raw = kw['extra_body']['guided_choice'][-1]
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    monkeypatch.setenv('RUBRIC_JUDGE_GRANULARITY', 'all_rules')
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_TARGETED_JUDGE_TRACE_DIR', str(tmp_path))
    api = rj.ApiJudge('http://localhost:8002/v1', 'secret', 'judge')
    rules = rj.RubricRules(['safe'], ['useful'], [], [])
    output = api.judge('p', 'a', rules, {'targeted_refusal_reference_only': True})
    assert output == dict(verdict(rules), _parse_valid=True)
    assert len(requests) == 2
    events = [json.loads(line) for path in tmp_path.glob('*.jsonl') for line in path.read_text().splitlines()]
    assert len(events) == 2
    assert all(e['request_audit']['endpoint_port'] == 8002
               and e['request_audit']['configured_granularity'] == 'all_rules'
               and e['request_audit']['effective_granularity'] == 'single_rule'
               and len(e['request_audit']['system_sha256']) == 64 for e in events)
    assert 'secret' not in json.dumps(events)


def test_replica_gate_requires_every_replica_even_when_pool_passes(monkeypatch, tmp_path):
    module = smoke()
    monkeypatch.setenv('JUDGE_REPLICA_COUNT', '3')
    monkeypatch.setenv('RUBRIC_TARGETED_JUDGE_TRACE_DIR', 'restore-me')
    backend = SimpleNamespace(base_urls=['http://localhost:8000/v1', 'http://localhost:8002/v1',
                                        'http://localhost:8003/v1'], model='judge')
    seen = []
    def evaluate(data, wrapper, rules_type):
        seen.append(wrapper.judge.base_urls)
        return dict(passed=wrapper.judge.base_urls != [backend.base_urls[1]],
                    failed_case_ids=['retained_failure'] if len(seen) == 2 else [])
    monkeypatch.setattr(module, 'run_gate', evaluate)
    result = module.run_deployment_gate({'cases': []}, backend, rj.RubricRules, tmp_path,
        lambda endpoint: SimpleNamespace(base_urls=[endpoint], model='judge'))
    assert len(seen) == 4 and seen[-1] == backend.base_urls
    assert not result['passed'] and not result['deployment_passed']
    assert result['replica_validation']['replica_1']['failed_case_ids'] == ['retained_failure']
    assert len(list(tmp_path.rglob('gate.json'))) == 4
    assert module.os.environ['RUBRIC_TARGETED_JUDGE_TRACE_DIR'] == 'restore-me'


def test_formal_preflight_uses_training_deployment_and_exits_before_actor():
    source = (ROOT / 'train_val.sh').read_text()
    gate = source.index('python3 scripts/smoke_targeted_refusal.py')
    only = source.index('if [ "${RUBRIC_TARGETED_PREFLIGHT_ONLY:-0}" = "1" ]; then')
    actor = source.index('bash "${PROJECT_ROOT}/examples/ppo_trainer/qwen3B-instruct-GRPO.sh"')
    assert gate < only < actor
    assert 'exit 0' in source[only:actor]
    assert 'exit 1' in source[only:actor]
