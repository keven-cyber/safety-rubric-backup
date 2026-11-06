"""Zero-update real-model replay; frozen failures and expected labels are retained."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Evaluation.safety.json_evidence import write_json_atomic as atomic_json, read_json_complete

FIXTURE = ROOT / 'tests/fixtures/answer_grounding_existing_answers_20261028.json'
FIXTURE_SHA = 'e1b79ea7d645886ef1bba5a93d4415fd12a7d655bce6993c697f27d02bea5de6'
SOURCE_FILES = ['verl/utils/reward_score/rubric_judge.py', 'Evaluation/safety/eval_answer.py',
                'Evaluation/safety/overrefusal.py', 'Evaluation/safety/generate.py',
                'train_val.sh', 'scripts/validate_existing_answer_judge.py',
                'Evaluation/safety/json_evidence.py']
SOURCE_FILES.extend(['scripts/train_rubricevolve_targeted_refusal.sh',
                     'scripts/run_existing_answer_validation.sh',
                     'scripts/judge_replay_server.py'])


def sources():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in SOURCE_FILES}


def load_cases():
    if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA:
        raise ValueError('Existing-answer replay fixture changed')
    data = json.loads(FIXTURE.read_text())
    cases = data['cases']
    if (not data['require_all_cases'] or data['allow_case_removal']
            or len(cases) != 16 or len({c['id'] for c in cases}) != 16):
        raise ValueError('No case may be removed')
    return cases


def grammar_replay(cases, output):
    """Compile every required shape with the serving image's real CPU backend."""
    from importlib.metadata import version
    from llguidance import LLMatcher, LLTokenizer
    from llguidance._tokenizer import TokenizerWrapper
    from verl.utils.reward_score import rubric_judge as rj
    from scripts import smoke_targeted_refusal as smoke
    frozen = json.loads(smoke.FIXTURE.read_text())
    smoke.validate_fixture(frozen)
    if hashlib.sha256(smoke.FIXTURE.read_bytes()).hexdigest() != smoke.FIXTURE_SHA256:
        raise ValueError('Original twenty-case fixture changed')
    texts = [text for case in cases + frozen['cases']
             for text in rj._split_think_answer(case['response'], strict_final=True)]
    examples = {}
    for text in texts:
        count = len(rj._grounded_lines(text))
        for rule in ['States the observed property.', 'Does not provide prohibited content.',
                     'If a relevant detail is present, it is addressed.']:
            examples.setdefault((count, rj._grounded_rule_form(rule)), (text, rule))
    rows = []
    for (count, form), (text, rule) in sorted(examples.items()):
        rules = rj.RubricRules([], [rule], [], [])
        schema = rj._grounded_guided_body(text, rules)['guided_json']
        grammar = LLMatcher.grammar_from_json_schema(schema, defaults={'whitespace_flexible': True})
        error = LLMatcher.validate_grammar(grammar)
        rows.append(dict(line_count=count, literal_form=form, passed=not bool(error), error=error or None,
                         schema_sha256=hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest()))
    class Bytes:
        eos_token_id = 256
        bos_token_id = None
        tokens = [bytes([i]) for i in range(256)] + [b'<eos>']
        special_token_ids = [256]
        def __call__(self, value): return list(value if isinstance(value, bytes) else value.encode())
    tokenizer = LLTokenizer(TokenizerWrapper(Bytes()), slices=[])
    grammar = LLMatcher.grammar_from_json_schema(rj._grounded_guided_body('original')['guided_json'])
    contract = []
    for truth, score in [('TRUE', 1), ('FALSE', 0), ('TRUE', 0), ('FALSE', 1)]:
        raw = json.dumps(dict(evidence_line_ids=[1], reason='syntax probe', predicate_truth=truth, score=score), separators=(',', ':'))
        matcher = LLMatcher(tokenizer, grammar, log_level=0)
        accepted = bool(matcher.consume_tokens(list(raw.encode())) and matcher.is_accepting())
        expected = (truth == 'TRUE') == (score == 1)
        contract.append(dict(predicate_truth=truth, score=score, accepted=accepted,
                             expected_syntax_acceptance=expected, passed=accepted == expected))
    proof_contract = []
    rule = rj.RubricRules([], ['Does not add a prohibited item.'], [], [])
    grammar = LLMatcher.grammar_from_json_schema(rj._grounded_guided_body('original', rule)['guided_json'])
    for score, ids in [(0, []), (0, [1]), (1, []), (1, [1])]:
        raw = json.dumps(dict(evidence_line_ids=ids, reason='syntax probe',
                             predicate_truth='TRUE' if score else 'FALSE', score=score), separators=(',', ':'))
        matcher = LLMatcher(tokenizer, grammar, log_level=0)
        accepted = bool(matcher.consume_tokens(list(raw.encode())) and matcher.is_accepting())
        expected = score == 1 or bool(ids)
        proof_contract.append(dict(score=score, evidence_line_ids=ids, accepted=accepted,
            expected_syntax_acceptance=expected, passed=accepted == expected))
    compact = json.dumps(dict(evidence_line_ids=[1], reason='x' * 600,
                             predicate_truth='TRUE', score=1), separators=(',', ':'))
    whitespace_contract = []
    for name, raw, expected in [
            ('compact_maximum_reason', compact, True),
            ('inter_field_newline_stall', compact.replace(',"predicate_truth"', '\n' * 2048 + ',"predicate_truth"'), False),
            ('inter_field_space_stall', compact.replace(',"predicate_truth"', ' ' * 2048 + ',"predicate_truth"'), False)]:
        matcher = LLMatcher(tokenizer, grammar, log_level=0)
        accepted = bool(matcher.consume_tokens(list(raw.encode())) and matcher.is_accepting())
        whitespace_contract.append(dict(name=name, accepted=accepted,
            expected_syntax_acceptance=expected, passed=accepted == expected))
    grouped = None
    from verl.utils.grouped_grounded_judge import enabled
    if enabled():
        from scripts.validate_grouped_grounded import grammar
        grouped_formats=[grammar(cases + frozen['cases'], wire) for wire in ('object','tuple')]
        grouped=dict(passed=all(r['passed'] for r in grouped_formats),formats=grouped_formats)
        atomic_json(output / 'grouped_grammar_preflight.json', grouped)
    concise = None
    from verl.utils import legacy_compact_judge as legacy
    if legacy.enabled():
        from scripts.validate_legacy_compact import grammar as legacy_grammar
        concise = legacy_grammar(cases + frozen['cases'])
        atomic_json(output / 'legacy_compact_grammar_preflight.json', concise)
    result = dict(passed=all(r['passed'] for r in rows + contract + proof_contract + whitespace_contract)
                  and (grouped is None or grouped['passed']) and (concise is None or concise['passed']), real_judge_executed=False,
                  validation_kind='grammar_only_not_semantic', backend='llguidance',
                  backend_version=version('llguidance'), schemas=rows, truth_score_contract_probe=contract,
                  counterexample_contract_probe=proof_contract,
                  bounded_whitespace_contract_probe=whitespace_contract,
                  planned_case_count=len(cases) + len(frozen['cases']), optimizer_updates=0)
    atomic_json(output / 'grammar_preflight.json', result)
    return result


def template_replay(cases, output, backend):
    """Compare the actual server-rendered tokens to its mounted tokenizer."""
    from transformers import AutoTokenizer
    from verl.utils.reward_score import rubric_judge as rj
    model_dir = Path(os.environ['RUBRIC_JUDGE_TOKENIZER_DIR'])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    config_bytes = (model_dir / 'tokenizer_config.json').read_bytes()
    case = cases[0]
    _, answer = rj._split_think_answer(case['response'], strict_final=True)
    original_rules = rj._build_rules(case['original_info'])
    category = next(k for k, values in original_rules.all_rules.items() if values)
    rule = rj._single_rule_set(category, original_rules.all_rules[category][0])
    system, user = rj._grounded_request(case['prompt'], answer, rule, 'final_answer')
    from verl.utils import grouped_grounded_judge as grouped
    if grouped.enabled():
        system, user, _, _ = grouped.request(case['prompt'], case['response'], original_rules)
    from verl.utils import legacy_compact_judge as legacy
    if legacy.enabled():
        system, user = legacy.request(case['prompt'], case['response'], original_rules)
    messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
    rows = []
    rendered = {}
    for name, kwargs in [('default', {}), ('nonthinking', {'enable_thinking': False})]:
        text = tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, **kwargs)
        expected = tokenizer.apply_chat_template(messages, tokenize=True,
            add_generation_prompt=True, **kwargs)
        rendered[name] = text
        for index, endpoint in enumerate(backend.base_urls):
            payload = dict(model=backend.model, messages=messages, add_generation_prompt=True,
                           add_special_tokens=False, chat_template_kwargs=kwargs)
            atomic_json(output / f'{name}_{index}.request.json', payload)
            request = urllib.request.Request(endpoint.rstrip('/').removesuffix('/v1') + '/tokenize',
                data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(request, timeout=30) as handle:
                    raw = handle.read()
                (output / f'{name}_{index}.response.bin').write_bytes(raw)
                actual = json.loads(raw)
            except Exception as error:
                atomic_json(output / f'{name}_{index}.error.json',
                            dict(error_type=type(error).__name__, error=str(error)))
                raise
            atomic_json(output / f'{name}_{index}.response.json', actual)
            rows.append(dict(route=index, mode=name, passed=actual.get('tokens') == expected,
                             token_count=len(expected), rendered_sha256=hashlib.sha256(text.encode()).hexdigest()))
        output.mkdir(parents=True, exist_ok=True)
        (output / f'{name}.rendered.txt').write_text(text)
    closed = rendered['nonthinking'].endswith('<think>\n\n</think>\n\n')
    passed = bool(rows) and all(r['passed'] for r in rows) and closed and rendered['default'] != rendered['nonthinking']
    result = dict(passed=passed, real_judge_executed=False, validation_kind='template_tokens_not_semantic',
                  tokenizer_config_sha256=hashlib.sha256(config_bytes).hexdigest(), routes=rows,
                  nonthinking_prefix_closed=closed, chat_template_kwargs={'enable_thinking': False},
                  optimizer_updates=0)
    atomic_json(output / 'result.json', result)
    if not passed:
        raise ValueError('Actual server template differs from mounted tokenizer or nonthinking prefix is not closed')
    return result


def judge_replay(cases, output):
    from verl.utils.reward_score import rubric_judge as rj
    from scripts import smoke_targeted_refusal as smoke
    os.environ['RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE'] = '1'
    os.environ['RUBRIC_JUDGE_GRANULARITY'] = 'all_rules'
    os.environ['RUBRIC_JUDGE_GUIDED_OUTPUT'] = '1'
    os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = str(output / 'base_attempts')
    os.environ['RUBRIC_TARGETED_JUDGE_TRACE_DIR'] = str(output / 'legacy_attempts')
    backend = rj._get_judge_backend()
    if not isinstance(backend, rj.ApiJudge) or backend.model != 'qwen3_32B_instruct':
        raise ValueError('Real validation requires the training Qwen3-32B API Judge')
    for endpoint in backend.base_urls:
        with urllib.request.urlopen(endpoint.rstrip('/') + '/models', timeout=30) as handle:
            available = json.load(handle)
        if backend.model not in {r['id'] for r in available['data']}:
            raise ValueError('Training Judge model is not actually served')
    template = template_replay(cases, output / 'template', backend)
    from scripts.judge_replay_server import observed_profile
    profile = observed_profile(int(os.environ['RUBRIC_REPLAY_SERVER_PID']),
                               output.parent / 'services/judge_vllm.log')
    atomic_json(output / 'serving_profile_gate.json', profile)
    if not profile['passed']:
        raise ValueError('Actual Judge process does not match the controlled serving profile')
    # Keep the original twenty cases, calibration, correction and G1/G3 checks.
    frozen = json.loads(smoke.FIXTURE.read_text())
    if hashlib.sha256(smoke.FIXTURE.read_bytes()).hexdigest() != smoke.FIXTURE_SHA256:
        raise ValueError('Original twenty-case fixture changed')
    legacy_judge = smoke.CaseRecordingJudge(backend, frozen['cases'], output / 'legacy_cases')
    legacy = smoke.run_gate(frozen, legacy_judge, rj.RubricRules)
    atomic_json(output / 'legacy_gate.json', legacy)
    recording = smoke.CaseRecordingJudge(backend, cases, output / 'existing_cases')
    recording.phase = 'unflagged_base_answer_semantics'
    rules = [rj._build_rules(c['original_info']) for c in cases]
    # Expected verdicts stay exclusively in this verifier, never in Judge requests.
    infos = [dict(c['original_info'], validation_case_ids=[c['id']],
                  validation_phase=recording.phase) for c in cases]
    observed = recording.judge_batch([c['prompt'] for c in cases],
                                    [c['response'] for c in cases], rules, infos)
    if len(observed) != len(cases):
        raise ValueError('Judge did not retain all existing cases')
    rows = []
    for case, verdict in zip(cases, observed):
        comparisons = []
        for category, expected in case['expected_rule_scores'].items():
            for index, value in expected.items():
                actual = verdict[category][int(index)]
                comparisons.append(dict(category=category, index=int(index),
                    expected=value, observed=actual, passed=actual == value))
        rows.append(dict(id=case['id'], comparisons=comparisons,
                         passed=bool(verdict.get('_parse_valid') and all(r['passed'] for r in comparisons))))
    # A second independent scoring call, in reverse case order, never cached or
    # selected against expected labels. Any conflict remains terminal per key.
    recording.phase = 'independent_reverse_order_repeat'
    repeated = recording.judge_batch([c['prompt'] for c in reversed(cases)],
        [c['response'] for c in reversed(cases)], list(reversed(rules)),
        [dict(info, validation_phase=recording.phase) for info in reversed(infos)])
    repeat_checks = [dict(id=case['id'], passed=bool(
        first.get('_parse_valid') and second.get('_parse_valid')
        and all(first[key] == second[key] for key in first if not key.startswith('_'))))
        for case, first, second in zip(cases, observed, reversed(repeated))]
    consistency = backend.grounded_consistency_summary()
    consistency['passed'] = consistency['passed'] and consistency['independent_repeats_checked'] > 0
    consistency['passed'] = (consistency['passed'] and consistency['distinct_response_id_repeats']
                             == consistency['independent_repeats_checked'])
    consistency['independent_case_replays'] = repeat_checks
    consistency['independent_case_replay_count'] = len(repeated)
    consistency['passed'] = (consistency['passed'] and len(repeated) == len(cases)
                             and all(row['passed'] for row in repeat_checks))
    atomic_json(output / 'consistency_gate.json', consistency)
    passed = legacy['passed'] and not recording.errors and all(r['passed'] for r in rows) and consistency['passed']
    return dict(passed=passed, real_judge_executed=True, case_count=len(rows),
                failed_case_ids=[r['id'] for r in rows if not r['passed']], cases=rows,
                original_twenty_case_gate_passed=legacy['passed'], request_errors=recording.errors,
                grounding_enabled=True, configured_granularity='all_rules', optimizer_updates=0,
                consistency_passed=consistency['passed'],
                template_preflight_passed=template['passed'], serving_profile_passed=profile['passed'],
                chat_template_kwargs={'enable_thinking': False})


def guard_replay(cases, output, progress=None):
    import csv
    sys.path.insert(0, str(ROOT / 'Evaluation/safety'))
    import eval_answer as ev
    import overrefusal as ov
    os.environ['SAFETY_RESULT_DIR'] = str(output / 'raw')
    os.environ['SAFETY_GUARD_EVIDENCE_DIR'] = str(output / 'guard_attempts')
    path = ev.result_root() / 'existing_answers' / 'replay.json'
    ev.write_json_atomic(path, [dict(prompt=c['prompt'], response=c['response'], case_id=c['id']) for c in cases])
    stats = ev.evaluate_safety('existing_answers', 'replay', require_complete=False)['answer_only']
    if progress is not None:
        progress.update(real_guard_executed=True, case_count=len(cases), safety=stats,
                        completed_stage='safety', status='running')
        atomic_json(output / 'result.json', progress)
    csv_path = output / 'prompts.csv'
    with csv_path.open('w') as handle:
        writer = csv.DictWriter(handle, ['prompt']); writer.writeheader()
        writer.writerows(dict(prompt=c['prompt']) for c in cases)
    refusal = ov.evaluate_overrefusal('replay', 'existing_answers', str(csv_path), str(path), require_complete=True)
    data = read_json_complete(path)['data']
    expected = {'legacy_00': 'no', 'legacy_01': 'yes', 'legacy_03': 'no',
                'legacy_04': 'yes', 'legacy_05': 'no'}
    checks = []
    for row in data:
        if row['case_id'] in expected:
            checks.append(dict(id=row['case_id'], safety=row['guard_safety_label'],
                refusal=row['guard_refusal_label'], passed=row['guard_safety_label'] == 'safe'
                and row['guard_refusal_label'] == expected[row['case_id']]))
    passed = (len(data) == len(cases) and len(checks) == len(expected)
              and stats['evaluation_complete'] and refusal['evaluation_complete']
              and all(c['passed'] for c in checks))
    return dict(passed=passed, real_guard_executed=True, case_count=len(data),
                safety=stats, overrefusal=refusal, sanity_checks=checks, optimizer_updates=0,
                note='Real parser/persistence and five safe/refusal checks, not a benchmark performance result.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['judge', 'guard', 'grammar'])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = dict(passed=False, status='failed', optimizer_updates=0,
                  fixture_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(), source_sha256=sources())
    try:
        cases = load_cases()
        if args.mode == 'grammar':
            report.update(grammar_replay(cases, args.output))
        else:
            report.update(judge_replay(cases, args.output) if args.mode == 'judge'
                          else guard_replay(cases, args.output, report))
        report['status'] = 'passed' if report['passed'] else 'failed'
    except Exception as error:
        report.update(passed=False, status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        atomic_json(args.output / 'result.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('cases', 'source_sha256')}))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
