"""Real base-reward protocol and one-call failures, not only correction cases."""
import ast
import importlib.util
import itertools
import json
from pathlib import Path
import re
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from verl.utils.reward_score import rubric_judge as rj
from verl.utils.training_evidence import atomic_json

ROOT = Path(__file__).resolve().parents[2]


def load_trainer_function(name):
    path = ROOT / 'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    env = dict(torch=torch, os=__import__('os'), __file__=str(path), atomic_json=atomic_json)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


@pytest.mark.parametrize('shape', [(0, 0, 0, 0), (2, 0, 1, 2), (1, 2, 2, 0), (3, 1, 0, 2)])
def test_regex_accepts_every_binary_combination_and_exactly_requested_slots(monkeypatch, shape):
    monkeypatch.setenv('RUBRIC_JUDGE_GUIDED_OUTPUT', '1')
    rules = rj.RubricRules(*[['r'] * n for n in shape])
    pattern = rj._judge_guided_body(rules)['guided_regex']
    for bits in itertools.product((0, 1), repeat=sum(shape)):
        start = 0
        output = {}
        for key, n in zip(rules.all_rules, shape):
            output[key] = list(bits[start:start + n]); start += n
        assert re.fullmatch(pattern, json.dumps(output))
        assert re.fullmatch(pattern, json.dumps(output, indent=2))
        for key in output:
            malformed = dict(output, **{key: output[key] + [0]})
            assert not re.fullmatch(pattern, json.dumps(malformed))
    for bad in ('true', '"0"', '0.0', '2', '-1'):
        if sum(shape):
            text = json.dumps({k: [0] * n for k, n in zip(rules.all_rules, shape)})
            assert not re.fullmatch(pattern, text.replace('[0', '[' + bad, 1))


def fake_api(monkeypatch, answers):
    requests = []
    answers = iter(answers)
    def create(**kw):
        requests.append(kw)
        raw = next(answers)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    module = ModuleType('openai')
    module.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', module)
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_RETRY_BACKOFF', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_GUIDED_OUTPUT', '1')
    return rj.ApiJudge('http://localhost:8000/v1', 'secret', 'judge'), requests


def test_base_misalignment_evidence_retained_and_success_never_fabricated(monkeypatch, tmp_path):
    rules = rj.RubricRules(['a', 'b'], ['c'], [], [])
    bad = dict(outcome_safety_rule=[1], outcome_useful_rule=[0], reasoning_process_rule=[], refusal_rule=[])
    good = dict(bad, outcome_safety_rule=[0, 1])
    api, requests = fake_api(monkeypatch, [json.dumps(bad), json.dumps(good)])
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    out = api.judge('question', 'original answer', rules, {'uid': 'kept'})
    assert out['outcome_safety_rule'] == [0, 1] and out['_parse_valid']
    assert len(requests) == 2 and all('guided_regex' in q['extra_body'] for q in requests)
    rows = [json.loads(s) for p in (tmp_path / 'judge_errors').glob('*.jsonl') for s in p.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]['raw_output'] == json.dumps(bad)
    assert rows[0]['observed_lengths']['outcome_safety_rule'] == 1
    assert rows[0]['required_lengths']['outcome_safety_rule'] == 2
    assert rows[0]['response'] == 'original answer' and rows[0]['uid'] == 'kept'
    assert 'secret' not in json.dumps(rows)


def test_base_exhaustion_is_error_even_if_drop_failed_would_allow_filtering(monkeypatch, tmp_path):
    rules = rj.RubricRules(['a', 'b'], [], [], [])
    api, requests = fake_api(monkeypatch, ['{}'] * 5)
    monkeypatch.setenv('RUBRIC_JUDGE_DROP_FAILED', '0')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='no reward was fabricated'):
        api.judge('question', 'failed sample', rules)
    assert len(requests) == 4
    assert sum(len(p.read_text().splitlines()) for p in (tmp_path / 'judge_errors').glob('*.jsonl')) == 4


@pytest.mark.parametrize('error', [ValueError('bad arrays'), TypeError('internal failure')])
def test_reward_exception_never_reexecutes_batch(error):
    call_once = load_trainer_function('_call_reward_once')
    calls = []
    def reward(batch, return_dict=False):
        calls.append(return_dict)
        raise error
    with pytest.raises(type(error), match=str(error)):
        call_once(reward, object())
    assert calls == [True]


def test_legacy_signature_is_detected_without_first_failing_call():
    call_once = load_trainer_function('_call_reward_once')
    calls = []
    def reward(batch):
        calls.append(batch)
        return torch.tensor([.3])
    token = object()
    assert call_once(reward, token)['reward_tensor'].item() == pytest.approx(.3)
    assert calls == [token]


@pytest.mark.parametrize('failure', ['none', 'nan', 'wrong_groups', 'wrong_dataset', 'invalid_judge'])
def test_real_batch_gate_checks_finite_values_groups_and_all_rewards(monkeypatch, tmp_path, failure):
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    write_gate = load_trainer_function('_write_training_batch_gate')
    tensors = {name: torch.ones(64, 2) for name in ('token_level_scores', 'token_level_rewards', 'advantages')}
    uids = [str(i) for i in range(8) for _ in range(8)]
    traces = [dict(judge_parse_valid=True, judge_failed=False, score=.5) for _ in range(64)]
    dataset = dict(train=3120, val=304)
    if failure == 'nan': tensors['advantages'][3, 0] = float('nan')
    if failure == 'wrong_groups': uids[0] = 'extra'
    if failure == 'wrong_dataset': dataset['train'] -= 1
    if failure == 'invalid_judge': traces[7]['judge_parse_valid'] = False
    class Batch:
        batch = tensors
        non_tensor_batch = {'uid': uids}
        def __len__(self): return 64
    if failure == 'none':
        assert write_gate(Batch(), traces, 1, dataset)['passed']
    else:
        with pytest.raises(ValueError, match='no optimizer update'):
            write_gate(Batch(), traces, 1, dataset)
    saved = json.loads((tmp_path / 'preflight/training_batch_gate.json').read_text())
    assert saved['passed'] is (failure == 'none') and saved['optimizer_updates'] == 0


def test_preflight_returns_before_any_optimizer_and_final_evaluation():
    trainer = (ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text()
    start = trainer.index('if batch_preflight:', trainer.index('self._dump_rollout_traces(\n                            batch,'))
    assert trainer.index('return', start) < trainer.index('self.critic_wg.update_critic(batch)', start)
    launcher = (ROOT / 'train_val.sh').read_text()
    gate = launcher.index('if [ "${RUBRIC_TRAIN_BATCH_PREFLIGHT_ONLY:-0}" = "1" ]; then')
    assert 'exit 0' in launcher[gate:launcher.index('# Judge replicas', gate)]


@pytest.mark.parametrize('exhausted', [False, True])
def test_base_gate_is_an_actual_unflagged_heterogeneous_64_response_batch(tmp_path, exhausted):
    spec = importlib.util.spec_from_file_location('base_gate_smoke', ROOT / 'scripts/smoke_targeted_refusal.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data = json.loads(module.FIXTURE.read_text())
    observed = []
    class Judge:
        def judge_batch(self, prompts, responses, rules, infos):
            observed.append((prompts, responses, rules, infos))
            assert len(prompts) == len(responses) == len(rules) == len(infos) == 64
            assert all(not rj._targeted_request(info) for info in infos)
            if exhausted:
                raise RuntimeError('must not be converted into passing reward')
            return [dict({k: [1] * len(v) for k, v in r.all_rules.items()}, _parse_valid=True) for r in rules]
    report = module.run_base_protocol_gate(data, Judge(), tmp_path)
    assert len(observed) == 1 and report['passed'] is (not exhausted)
    assert report['batch_returned'] is (not exhausted)
    inputs = json.loads((tmp_path / 'inputs.json').read_text())
    assert len(inputs['responses']) == 64 and len(set(inputs['sample_ids'])) == 8
    assert len({sum(map(len, r.values())) for r in inputs['rules']}) > 1
    assert len(data['cases']) == 20
