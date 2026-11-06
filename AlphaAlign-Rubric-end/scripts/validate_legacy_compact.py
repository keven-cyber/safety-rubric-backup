"""Actual llguidance compact-array syntax probes, not semantic/accuracy proof."""
import json
from importlib.metadata import version
from verl.utils import legacy_compact_judge as legacy
from verl.utils.reward_score import rubric_judge as rj


def grammar(cases):
    from llguidance import LLMatcher, LLTokenizer
    from llguidance._tokenizer import TokenizerWrapper
    class Bytes:
        eos_token_id=256
        bos_token_id=None
        tokens=[bytes([i]) for i in range(256)]+[b'<eos>']
        special_token_ids=[256]
        def __call__(self,value):return list(value if isinstance(value,bytes) else value.encode())
    tokenizer=LLTokenizer(TokenizerWrapper(Bytes()),slices=[]);rows=[]
    for case in cases:
        rules=rj._build_rules(case['original_info']);body=legacy.guidance(rules)
        if 'guided_regex' in body:
            compiled=LLMatcher.grammar_from_regex(body['guided_regex'])
        else:
            choices=body['guided_choice'];compiled=LLMatcher.grammar_from_json_schema({'enum':[json.loads(x) for x in choices]})
        probes=[]
        for bit in [0,1]:
            value={k:[bit]*len(v) for k,v in rules.all_rules.items()}
            for name,raw,expected in [('valid',json.dumps(value),True),
                    ('extra_key',json.dumps(dict(value,extra=1)),False),
                    ('nonbinary',json.dumps(value).replace(str(bit),'2',1),False)]:
                matcher=LLMatcher(tokenizer,compiled,log_level=0)
                accepted=bool(matcher.consume_tokens(list(raw.encode())) and matcher.is_accepting())
                probes.append(dict(name=name,bit=bit,accepted=accepted,expected=expected,passed=accepted==expected))
        rows.append(dict(id=case['id'],array_lengths={k:len(v) for k,v in rules.all_rules.items()},passed=all(p['passed'] for p in probes),probes=probes))
    return dict(passed=all(r['passed'] for r in rows),case_count=len(rows),backend='llguidance',backend_version=version('llguidance'),
        model_calls=0,optimizer_updates=0,protocol=legacy.PROTOCOL,rows=rows)
