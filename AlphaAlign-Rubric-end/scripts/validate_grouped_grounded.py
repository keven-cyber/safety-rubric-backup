"""Compile the actual grouped schemas, not a model/semantic speed claim."""
import hashlib
import json
from importlib.metadata import version
from verl.utils import grouped_grounded_judge as gg
from verl.utils.reward_score import rubric_judge as rj


def grammar(cases, wire='object'):
    from llguidance import LLMatcher, LLTokenizer
    from llguidance._tokenizer import TokenizerWrapper
    rows = []
    class Bytes:
        eos_token_id = 256
        bos_token_id = None
        tokens = [bytes([i]) for i in range(256)] + [b'<eos>']
        special_token_ids = [256]
        def __call__(self, value): return list(value if isinstance(value, bytes) else value.encode())
    tokenizer = LLTokenizer(TokenizerWrapper(Bytes()), slices=[])
    for number, case in enumerate(cases):
        rules = rj._build_rules(case['original_info'])
        _, _, guidance, sections = gg.request(case['prompt'], case['response'], rules, wire)
        schema = guidance['guided_json']
        compiled = LLMatcher.grammar_from_json_schema(schema, defaults={'whitespace_flexible': True})
        error = LLMatcher.validate_grammar(compiled)
        if error:
            raise ValueError('Grouped grammar cannot compile: ' + str(error))
        raw = {key: dict(evidence_line_ids=[], reason='syntax probe', predicate_truth='TRUE', score=1)
               for key, *_ in gg.slots(rules)}
        probes = []
        def test(name, value, expected):
            matcher = LLMatcher(tokenizer, compiled, log_level=0)
            accepted = bool(matcher.consume_tokens(list(value.encode())) and matcher.is_accepting())
            probes.append(dict(name=name, accepted=accepted, expected=expected, passed=accepted == expected))
        def packed(value):
            return [list(row.values()) for row in value.values()] if wire=='tuple' else value
        text = json.dumps(packed(raw), separators=(',', ':'))
        test('all_original_slots', text, True)
        missing = dict(raw); missing.pop(next(iter(raw)))
        test('missing_original_slot', json.dumps(packed(missing), separators=(',', ':')), False)
        bad = json.loads(json.dumps(raw)); bad[next(iter(bad))]['score'] = 0
        test('contradictory_truth_score', json.dumps(packed(bad), separators=(',', ':')), False)
        test('inter_field_whitespace_stall', text.replace('"syntax probe"', ' ' * 2048 + '"syntax probe"'), False)
        gg.output(packed(raw), rules, sections, wire)
        rows.append(dict(position=number, id=case['id'], original_rule_count=len(gg.slots(rules)),
            schema_sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
            passed=all(p['passed'] for p in probes), probes=probes))
    return dict(wire_format=wire, passed=all(r['passed'] for r in rows), backend_version=version('llguidance'),
        schemas=rows, real_judge_executed=False, optimizer_updates=0,
        interpretation='Real grammar backend only; no semantic accuracy or GPU speed claim')
