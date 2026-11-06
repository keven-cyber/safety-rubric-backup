"""Unknown safety judgments remain unknown; thought cannot replace an answer."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from verl.utils.reward_score import rubric_judge as rj

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('guard_evidence_eval', ROOT / 'Evaluation/safety/eval_answer.py')
guard_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_eval)


@pytest.mark.parametrize('raw', ['', 'I cannot comply.', 'safe',
    'Safety: Safe\nSafety: Unsafe\nRefusal: No', 'Safety: Safe because I think so\nRefusal: No'])
def test_invalid_labels_are_neither_unsafe_nor_safe(raw):
    out = guard_eval._parse_guardrail_text(raw)
    assert out['is_safe'] is None and not out['parse_valid']


def test_controversial_remains_non_safe_and_refusal_is_separate():
    out = guard_eval._parse_guardrail_text('Safety: Controversial\nCategories: Other\nRefusal: Yes')
    assert out['is_safe'] is False and out['refusal_label'] == 'yes'
    out = guard_eval._parse_guardrail_text('Safety: Safe\nCategories: None\nRefusal: No')
    assert out['is_safe'] is True and out['refusal_label'] == 'no'


def test_retry_only_invalid_format_and_preserve_each_raw_attempt(monkeypatch, tmp_path):
    monkeypatch.setenv('SAFETY_GUARD_EVIDENCE_DIR', str(tmp_path))
    api = guard_eval.Qwen3GuardAPI.__new__(guard_eval.Qwen3GuardAPI)
    outputs = iter(['missing', 'Safety: Unsafe\nCategories: Illegal Activities\nRefusal: No'])
    api._call_guard_api = lambda **kw: next(outputs)
    out = api.evaluate_single('original question', 'original answer')
    assert out['is_safe'] is False and out['guard_attempts'] == 2
    rows = [json.loads(l) for p in tmp_path.glob('*.jsonl') for l in p.read_text().splitlines()]
    assert [r['raw_guard_output'] for r in rows] == ['missing', 'Safety: Unsafe\nCategories: Illegal Activities\nRefusal: No']
    assert rows[0]['error_type'] == 'ValueError' and rows[1]['error_type'] is None


def test_case_exhaustion_does_not_remove_later_samples():
    class Guard:
        def evaluate_conversations(self, items):
            if items[0]['prompt'] == 'bad':
                raise ValueError('exhausted')
            return [dict(is_safe=True)]
    out = guard_eval.score_guard_cases(Guard(), [dict(prompt='bad'), dict(prompt='kept')])
    assert len(out) == 2 and out[0]['is_safe'] is None and out[1]['is_safe'] is True


def test_generation_snapshot_is_immutable_and_mismatches_fail(tmp_path):
    path = tmp_path / 'responses.json'
    data = [dict(prompt='p', response='unchanged')]
    snapshot = guard_eval.preserve_generation(path, data)
    original = Path(snapshot).read_bytes()
    guard_eval.preserve_generation(path, dict(data=data, metadata={'evaluated': True}))
    assert Path(snapshot).read_bytes() == original
    with pytest.raises(ValueError, match='does not match'):
        guard_eval.preserve_generation(path, [dict(prompt='p', response='different')])


def test_complete_eval_saves_every_unknown_then_fails(monkeypatch, tmp_path):
    rows = [dict(prompt='p', response='<answer>first</answer>'),
            dict(prompt='q', response='<answer>second</answer>')]
    path = tmp_path / 'advbench/model.json'
    guard_eval.write_json_atomic(path, rows)
    monkeypatch.setenv('SAFETY_RESULT_DIR', str(tmp_path))
    class Guard:
        def evaluate_conversations(self, items):
            if items[0]['prompt'] == 'p':
                raise ConnectionError('real failure')
            return [dict(is_safe=False, safety_label='unsafe', refusal_label='no',
                         violations=[], risk_categories=[], llamaguard_response='Safety: Unsafe\nRefusal: No')]
    monkeypatch.setattr(guard_eval, 'build_guard_evaluator', lambda: Guard())
    # Use non-strict input matching first; all original samples must survive.
    stats = guard_eval.evaluate_safety('advbench', 'model', require_complete=False)['answer_only']
    assert stats['unsafe_count'] == 1 and stats['invalid_judgment_count'] == 1
    assert stats['unsafe_percentage'] is None and stats['conservative_unsafe_percentage'] == 100
    saved = json.loads(path.read_text())
    assert len(saved['data']) == 2 and saved['data'][0]['llamaguard_safe'] is None
    assert json.loads(path.with_name('model.generation.json').read_text()) == rows
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/harmful_behaviors.csv').write_text('prompt\np\nq\n')
    monkeypatch.setattr(guard_eval, '__file__', str(tmp_path / 'eval_answer.py'))
    with pytest.raises(RuntimeError, match='NOT complete'):
        guard_eval.evaluate_safety('advbench', 'model', require_complete=True)


def test_overrefusal_unknown_is_not_dropped_from_the_report(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(ROOT / 'Evaluation/safety'))
    spec = importlib.util.spec_from_file_location('overrefusal_evidence_test', ROOT / 'Evaluation/safety/overrefusal.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    path, csv_path = tmp_path / 'responses.json', tmp_path / 'safe.csv'
    path.write_text(json.dumps([dict(prompt='p', response='a'), dict(prompt='q', response='b')]))
    csv_path.write_text('prompt\np\nq\n')
    class Guard:
        def evaluate_conversations(self, items):
            if items[0]['prompt'] == 'p':
                raise ConnectionError('unknown')
            return [dict(is_safe=True, safety_label='safe', refusal_label='yes')]
    monkeypatch.setattr(module, 'build_guard_evaluator', lambda: Guard())
    with pytest.raises(RuntimeError, match='valid labels'):
        module.evaluate_overrefusal('model', 'xstest', str(csv_path), str(path), require_complete=True)
    data = json.loads(path.read_text())
    report = data['metadata']['xstest_overrefusal_statistics']
    assert report['overrefusal_rate'] is None and report['invalid_refusal_judgments'] == 1
    assert report['refusal_rate_lower_bound'] == 50 and report['refusal_rate_upper_bound'] == 100
    assert len(data['data']) == 2 and report['ignored_none_judgments'] == 0


def test_grounding_separates_final_answer_and_reasoning(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    calls = []
    original = rj.ApiJudge.judge
    def intercepted(self, prompt, response, rules, extra_info=None):
        if (extra_info or {}).get('answer_grounding_scope'):
            calls.append((response, rules, extra_info))
            return dict({k: [0] * len(v) for k, v in rules.all_rules.items()}, _parse_valid=True)
        return original(self, prompt, response, rules, extra_info)
    monkeypatch.setattr(rj.ApiJudge, 'judge', intercepted)
    api = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge')
    rules = rj.RubricRules(['safe'], ['correct'], ['reasons'], ['refuses'])
    out = api.judge('p', '<think>I will answer perfectly.</think><answer>[Your analysis here]</answer>', rules)
    assert len(calls) == 4 and out['_parse_valid']
    assert all(sum(map(len, rule.all_rules.values())) == 1 for _, rule, _ in calls)
    assert 'I will answer perfectly' not in calls[0][0]
    assert calls[0][1].reasoning_process_rule == []
    assert '[Your analysis here]' not in calls[2][0]
    assert calls[2][1].outcome_useful_rule == []
    assert all(v == [0] for k, v in out.items() if k != '_parse_valid')


def test_thinking_only_cannot_be_used_as_final_answer():
    assert rj._split_think_answer('<think>I know the answer.', strict_final=True)[1] == ''
    assert rj._split_think_answer('plain answer', strict_final=True)[1] == 'plain answer'
    assert rj._split_think_answer('<think>x</think><answer>partial', strict_final=True)[1] == 'partial'


@pytest.mark.parametrize('response,answer', [
    ('<think>unsafe private reasoning only', ''),
    ('<think>analysis</think><answer>unfinished final', 'unfinished final'),
    ('<think>analysis</think>plain final', 'plain final'),
    ('<think>analysis</think><answer>complete</answer>', 'complete'),
])
def test_guard_only_reads_actual_final_answer(response, answer):
    assert guard_eval.extract_answer_content(response)[0] == answer


def test_existing_answers_are_frozen_and_expected_labels_stay_outside_requests():
    spec = importlib.util.spec_from_file_location('existing_answer_validation', ROOT / 'scripts/validate_existing_answer_judge.py')
    validation = importlib.util.module_from_spec(spec); spec.loader.exec_module(validation)
    cases = validation.load_cases()
    assert len(cases) == 16
    assert {'placeholder_benign', 'placeholder_harm', 'incorrect_nba_count',
            'correct_nba_answer', 'incorrect_nba_teams'}.issubset({c['id'] for c in cases})
    for case in cases:
        assert 'expected_rule_scores' not in case['original_info']
        # A blanket full-score Judge must fail the fixture, while both binary
        # polarities remain represented, including absence-of-violation rules.
    expected = [v for c in cases for indices in c['expected_rule_scores'].values() for v in indices.values()]
    assert 0 in expected and 1 in expected
