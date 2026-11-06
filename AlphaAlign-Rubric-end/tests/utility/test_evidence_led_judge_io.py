"""Storage truncation and evidence fabrication must fail without losing cases."""
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
from Evaluation.safety import json_evidence as io
from verl.utils.reward_score import rubric_judge as rj


def test_publish_commits_exact_bytes_and_rejects_nonfinite_numbers(tmp_path):
    path = tmp_path / 'response.json'
    value = {'data': [{'prompt': 'question', 'response': '原始回答'}]}
    digest = io.write_json_atomic(path, value)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert io.read_json_complete(path, expected_sha256=digest) == value
    manifest = json.loads(path.with_name(path.name + '.integrity.json').read_bytes())
    assert manifest == dict(sha256=digest, bytes=len(path.read_bytes()))
    with pytest.raises(ValueError):
        io.write_json_atomic(path, {'score': float('nan')})
    assert io.read_json_complete(path) == value


@pytest.mark.parametrize('stale', [b'{"data":[{"response":"unfinished', b'{"data":[]}'])
def test_delayed_mount_read_cannot_pass_truncated_or_stale_generation(monkeypatch, tmp_path, stale):
    path = tmp_path / 'response.json'
    expected = {'data': [{'response': 'every sample survives'}]}
    io.write_json_atomic(path, expected)
    original = Path.read_bytes
    reads = []
    def delayed(self):
        if self == path and len(reads) < 2:
            reads.append(stale)
            return stale
        return original(self)
    monkeypatch.setattr(Path, 'read_bytes', delayed)
    monkeypatch.setattr(io.time, 'sleep', lambda _: None)
    assert io.read_json_complete(path) == expected
    records = list((tmp_path / 'json_read_errors').glob('*.json'))
    assert len(records) == 2
    assert all(p.with_suffix('.bin').read_bytes() == stale for p in records)
    assert all(p.with_suffix('.manifest.bin').exists() for p in records)


def test_permanent_corrupt_json_is_preserved_and_raises(monkeypatch, tmp_path):
    path = tmp_path / 'response.json'
    original = b'{"data":[{"response":"cut off'
    path.write_bytes(original)
    sleeps = []
    monkeypatch.setattr(io.time, 'sleep', sleeps.append)
    with pytest.raises(json.JSONDecodeError):
        io.read_json_complete(path, max_attempts=3)
    assert len(sleeps) == 2 and path.read_bytes() == original
    records = list((tmp_path / 'json_read_errors').glob('*.json'))
    assert len(records) == 3
    assert all(p.with_suffix('.bin').read_bytes() == original for p in records)


def test_integrity_failure_cannot_pass_even_when_json_is_valid(monkeypatch, tmp_path):
    path = tmp_path / 'response.json'
    io.write_json_atomic(path, {'data': [1, 2, 3]})
    path.write_text('{"data":[1,2]}')
    monkeypatch.setattr(io.time, 'sleep', lambda _: None)
    with pytest.raises(ValueError, match='integrity manifest'):
        io.read_json_complete(path, max_attempts=2)
    assert json.loads(path.read_text())['data'] == [1, 2]


@pytest.mark.parametrize('bad', [
    {'evidence_line_ids': [], 'reason': 'absent', 'score': True},
    {'evidence_line_ids': [], 'reason': 'absent', 'score': '0'},
    {'evidence_line_ids': [], 'reason': '', 'score': 0},
    {'evidence_line_ids': [99], 'reason': 'assumed', 'score': 1},
    {'evidence_line_ids': [], 'reason': 'absent', 'score': 0, 'extra': 1},
])
def test_grounded_verdict_never_coerces_or_accepts_invented_evidence(bad):
    bad = dict(predicate_truth='TRUE' if bad.get('score') == 1 else 'FALSE', **bad)
    with pytest.raises(ValueError):
        rj._grounded_output(bad, rj.RubricRules([], ['predicate'], [], []), 'actual answer')


def test_evidence_score_preserves_both_binary_polarities_without_model_override():
    rule = rj.RubricRules([], ['one unchanged predicate'], [], [])
    for score in (0, 1):
        verdict = rj._grounded_output(dict(evidence_line_ids=[1], reason='literal predicate', predicate_truth='TRUE' if score else 'FALSE', score=score),
                                      rule, 'actual answer')
        assert verdict['outcome_useful_rule'] == [score] and verdict['_parse_valid']
    assert rj._grounded_guided_body('actual')['guided_json']['properties']['score']['enum'] == [0, 1]


def test_one_rule_failure_does_not_change_an_independent_rule_or_see_thought(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_BASE_JUDGE_TRACE_DIR', str(tmp_path))
    requests = []
    scores = iter([0, 1])
    def create(**kw):
        requests.append(kw)
        score = next(scores)
        raw = json.dumps(dict(evidence_line_ids=[], reason='this predicate alone', predicate_truth='TRUE' if score else 'FALSE', score=score))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    api = rj.ApiJudge('http://localhost:8000/v1', 'secret', 'judge')
    rules = rj.RubricRules([], ['Completes the task.', 'Introduces no unrequested items.'], [], [])
    out = api.judge('question', '<think>secret thought</think><answer>template</answer>', rules)
    assert out['outcome_useful_rule'] == [0, 1]
    assert len(requests) == 2
    assert all('secret thought' not in q['messages'][1]['content'] for q in requests)
    assert 'Introduces no' not in requests[0]['messages'][1]['content']
    assert 'Completes the task.' not in requests[1]['messages'][1]['content']
    assert all('guided_json' in q['extra_body'] for q in requests)
    assert all(q['extra_body']['chat_template_kwargs'] == {'enable_thinking': False} for q in requests)
    events = [json.loads(l) for p in tmp_path.glob('*.jsonl') for l in p.read_text().splitlines()]
    assert len(events) == 2 and all(e['request_audit']['evidence_led'] for e in events)
    assert [e['rule_index'] for e in events] == [0, 1]
    assert [json.loads(e['raw_output'])['score'] for e in events] == [0, 1]
    assert all(e['request_audit']['messages'] == q['messages'] for e, q in zip(events, requests))
    assert all(e['request_audit']['chat_template_kwargs'] == {'enable_thinking': False} for e in events)


def test_judge_semantic_failure_is_not_retried_to_find_a_pass(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    calls = []
    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            dict(evidence_line_ids=[], reason='required fact is absent', predicate_truth='FALSE', score=0))), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    out = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge').judge(
        'question', '<answer>actual</answer>', rj.RubricRules([], ['States the required fact.'], [], []))
    assert out['outcome_useful_rule'] == [0] and len(calls) == 1


@pytest.mark.parametrize('ids', [[0], [2], [True], [1, 1], ['1']])
def test_evidence_ids_cannot_fabricate_or_duplicate_source_lines(ids):
    rules = rj.RubricRules([], ['One rule'], [], [])
    with pytest.raises(ValueError):
        rj._grounded_output(dict(evidence_line_ids=ids, reason='literal', predicate_truth='TRUE', score=1), rules, 'one line')


def test_long_multiline_evidence_is_lossless_and_single_rule_is_explicit():
    text = 'First line with \"quotes\".\n\n' + 'very long original text ' * 1000 + '\nOriginal final line.'
    rules = rj.RubricRules([], ['Does not add an unrequested item.'], [], [])
    system, user = rj._grounded_request('question', text, rules, 'final_answer')
    data = json.loads(user)
    assert ''.join(line['text'] for line in data['assistant_evidence_lines']) == text
    assert list(data)[-1] == 'rule_to_evaluate'
    assert data['rule_to_evaluate'] == rules.outcome_useful_rule[0]
    assert not any(k in data for k in rules.all_rules)
    assert len(system.split()) < 300
    out = rj._grounded_output(dict(evidence_line_ids=[1, 3], reason='original lines', predicate_truth='FALSE', score=0), rules, text)
    assert out['outcome_useful_rule'] == [0]


@pytest.mark.parametrize('line_count', [0, 1, 4, 32, 100])
def test_real_llguidance_compiles_serving_contract(line_count):
    llguidance = pytest.importorskip('llguidance')
    text = 'original\n' * line_count
    schema = rj._grounded_guided_body(text)['guided_json']
    assert 'uniqueItems' not in schema['properties']['evidence_line_ids']
    grammar = llguidance.LLMatcher.grammar_from_json_schema(schema,
        defaults={'whitespace_flexible': True})
    assert llguidance.LLMatcher.validate_grammar(grammar) == ''
    # Removal from the unsupported generator schema never removes the final
    # application's mandatory duplicate/range/type checks.
    if line_count:
        with pytest.raises(ValueError):
            rj._grounded_output(dict(evidence_line_ids=[1, 1], reason='duplicate', predicate_truth='TRUE', score=1),
                                rj.RubricRules([], ['unchanged rule'], [], []), text)


def test_real_compiler_reproduces_previous_uniqueitems_failure():
    llguidance = pytest.importorskip('llguidance')
    schema = rj._grounded_guided_body('one line')['guided_json']
    schema['properties']['evidence_line_ids']['uniqueItems'] = True
    grammar = llguidance.LLMatcher.grammar_from_json_schema(schema)
    assert 'uniqueItems' in llguidance.LLMatcher.validate_grammar(grammar)


@pytest.mark.parametrize('truth,score,valid', [('TRUE', 1, True), ('FALSE', 0, True),
                                            ('TRUE', 0, False), ('FALSE', 1, False)])
def test_real_decoder_and_parser_agree_on_truth_score_contract(truth, score, valid):
    llguidance = pytest.importorskip('llguidance')
    from llguidance._tokenizer import TokenizerWrapper
    class Bytes:
        eos_token_id = 256
        bos_token_id = None
        tokens = [bytes([i]) for i in range(256)] + [b'<eos>']
        special_token_ids = [256]
        def __call__(self, value): return list(value if isinstance(value, bytes) else value.encode())
    tokenizer = llguidance.LLTokenizer(TokenizerWrapper(Bytes()), slices=[])
    schema = rj._grounded_guided_body('original')['guided_json']
    matcher = llguidance.LLMatcher(tokenizer, llguidance.LLMatcher.grammar_from_json_schema(schema), log_level=0)
    obj = dict(evidence_line_ids=[1], reason='literal proposition', predicate_truth=truth, score=score)
    accepted = matcher.consume_tokens(list(json.dumps(obj, separators=(',', ':')).encode())) and matcher.is_accepting()
    assert bool(accepted) is valid
    rule = rj.RubricRules([], ['unchanged rule'], [], [])
    if valid:
        assert rj._grounded_output(obj, rule, 'original')['outcome_useful_rule'] == [score]
    else:
        with pytest.raises(ValueError, match='consistent truth/score'):
            rj._grounded_output(obj, rule, 'original')


def test_identical_request_disagreement_is_terminal_and_never_overwritten(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_BASE_JUDGE_TRACE_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_JUDGE_API_MAX_RETRIES', '10')
    calls = []
    scores = iter([0, 1, 0])
    def create(**kwargs):
        calls.append(kwargs)
        score = next(scores)
        raw = json.dumps(dict(evidence_line_ids=[1], reason='independent observation',
                              predicate_truth='TRUE' if score else 'FALSE', score=score))
        return SimpleNamespace(id=f'independent-{len(calls)}', model='judge',
            choices=[SimpleNamespace(message=SimpleNamespace(content=raw), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    api = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge')
    rule = rj.RubricRules([], ['one unchanged proposition'], [], [])
    assert api.judge('question', '<answer>original</answer>', rule)['outcome_useful_rule'] == [0]
    for _ in range(2):
        with pytest.raises(rj.JudgeConsistencyError, match='Identical grounded Judge requests disagreed'):
            api.judge('question', '<answer>original</answer>', rule)
    assert len(calls) == 3 and calls[0] == calls[1] == calls[2]
    summary = api.grounded_consistency_summary()
    assert not summary['passed'] and summary['independent_repeats_checked'] == 2
    assert len(summary['conflicts']) == 2 and summary['cached_verdicts_returned'] == 0
    assert summary['conflicts'][0]['first']['score'] == 0
    assert summary['conflicts'][0]['current']['score'] == 1
    events = [json.loads(line) for p in tmp_path.glob('*.jsonl') for line in p.read_text().splitlines()]
    assert [x['error_type'] for x in events] == [None, 'JudgeConsistencyError', 'JudgeConsistencyError']
    assert events[1]['request_audit']['consistency']['first']['response_id'] == 'independent-1'


def test_identical_agreement_still_calls_model_independently(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    calls = []
    def create(**kwargs):
        calls.append(kwargs)
        raw = json.dumps(dict(evidence_line_ids=[], reason=f'independent reason {len(calls)}',
                              predicate_truth='TRUE', score=1))
        return SimpleNamespace(id=str(len(calls)), model='judge', choices=[SimpleNamespace(
            message=SimpleNamespace(content=raw), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    api = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge')
    rule = rj.RubricRules([], ['unchanged absence rule'], [], [])
    for _ in range(2): assert api.judge('q', '<answer>actual</answer>', rule)['outcome_useful_rule'] == [1]
    assert len(calls) == 2 and api.grounded_consistency_summary()['passed']
    assert api.grounded_consistency_summary()['independent_repeats_checked'] == 1


def test_truncated_completion_cannot_taint_consistency_ledger(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_JUDGE_API_MAX_RETRIES', '0')
    raw = json.dumps(dict(evidence_line_ids=[], reason='complete JSON but truncated generation',
                          predicate_truth='TRUE', score=1))
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kwargs: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kw: SimpleNamespace(id='truncated', model='judge', choices=[SimpleNamespace(
            message=SimpleNamespace(content=raw), finish_reason='length')]))))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    api = rj.ApiJudge('http://localhost:8000/v1', 'unused', 'judge')
    with pytest.raises(RuntimeError, match='Judge failed after 1 attempts'):
        api.judge('q', '<answer>actual</answer>', rj.RubricRules([], ['unchanged rule'], [], []))
    assert api.grounded_consistency_summary()['unique_requests'] == 0


@pytest.mark.parametrize('server_mismatch', [False, True])
def test_template_gate_checks_server_tokens_and_keeps_mismatch_evidence(monkeypatch, tmp_path, server_mismatch):
    from scripts import validate_existing_answer_judge as validation
    model = tmp_path / 'model'; model.mkdir()
    (model / 'tokenizer_config.json').write_text('{"chat_template":"original"}')
    monkeypatch.setenv('RUBRIC_JUDGE_TOKENIZER_DIR', str(model))
    calls = []
    class Tokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, **kwargs):
            text = 'assistant\n' + ('<think>\n\n</think>\n\n' if kwargs.get('enable_thinking') is False else '')
            return list(text.encode()) if tokenize else text
    fake = ModuleType('transformers')
    fake.AutoTokenizer = SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())
    monkeypatch.setitem(sys.modules, 'transformers', fake)
    class Handle:
        def __init__(self, data): self.data = data
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return json.dumps(self.data).encode()
    def send(request, timeout):
        payload = json.loads(request.data); calls.append(payload)
        tokens = Tokenizer().apply_chat_template(payload['messages'], True, True, **payload['chat_template_kwargs'])
        if server_mismatch and payload['chat_template_kwargs']: tokens = [999]
        return Handle(dict(tokens=tokens, count=len(tokens)))
    monkeypatch.setattr(validation.urllib.request, 'urlopen', send)
    output = tmp_path / 'evidence'
    backend = SimpleNamespace(model='qwen3_32B_instruct', base_urls=['http://localhost:8000/v1/'])
    if server_mismatch:
        with pytest.raises(ValueError, match='Actual server template differs'):
            validation.template_replay(validation.load_cases(), output, backend)
    else:
        assert validation.template_replay(validation.load_cases(), output, backend)['passed']
    report = io.read_json_complete(output / 'result.json')
    assert report['passed'] is (not server_mismatch) and report['real_judge_executed'] is False
    assert len(calls) == 2 and all('expected' not in json.dumps(c) for c in calls)
    assert io.read_json_complete(output / 'nonthinking_0.response.json')['tokens']
    assert calls[1]['chat_template_kwargs'] == {'enable_thinking': False}
