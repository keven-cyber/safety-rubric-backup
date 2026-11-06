"""Opt-in full-rubric transport; each unchanged predicate keeps strict evidence.

One response is judged in one HTTP request. This changes Judge context, not
reward arithmetic, and requires real semantic/impact gates before deployment.
Targeted correction requests retain their existing single-rule protocol.
"""
import copy
import json
import os
import re
import time
from functools import lru_cache

PROTOCOL = 'grouped_grounded_v1'


def enabled():
    return os.getenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE', '0') == '1'


def slots(rules):
    return [(f'{category}:{i}', category, i, rule,
             'reasoning' if category == 'reasoning_process_rule' else 'final_answer')
            for category, items in rules.all_rules.items() for i, rule in enumerate(items)]


def wire_format():
    value=os.getenv('RUBRIC_GROUPED_WIRE_FORMAT','object')
    if value not in ('object','tuple'):
        raise ValueError('Unknown grouped wire format')
    return value


def context_capacity_failure(error, audit):
    """Recognize only an explicit, arithmetically verified vLLM context400.

    This is unavailable scoring, never a verdict. Preserve the complete error;
    do not shorten input/output, remove rules, retry until a score or accept a
    partial vector. Unknown400 and programming errors retain fatal behavior.
    """
    if getattr(error, 'status_code', None) != 400 or not audit:
        return None
    raw = str(error)
    matches = re.findall(r"This model's maximum context length is (\d+) tokens\. "
        r"However, you requested (\d+) tokens \((\d+) in the messages, "
        r"(\d+) in the completion\)\.", raw)
    if len(matches) != 1:
        return None
    capacity, total, prompt, completion = map(int, matches[0])
    if (min(capacity, prompt, completion) <= 0 or total != prompt + completion
            or total <= capacity or completion != audit.get('max_tokens')):
        return None
    return dict(error_type=type(error).__name__, status_code=400, original_error=raw,
        max_context_tokens=capacity, requested_total_tokens=total,
        prompt_tokens=prompt, completion_tokens=completion,
        scoring_unavailable=True, no_verdict=True, no_budget_or_content_change=True)


def guidance_wire(guidance):
    kind=guidance.get('guided_json',{}).get('type')
    if kind not in ('object','array'):
        raise ValueError('Unknown grouped guidance shape')
    return 'tuple' if kind=='array' else 'object'


def request(prompt, response, rules, wire=None):
    from verl.utils.reward_score import rubric_judge as rj
    think, answer = rj._split_think_answer(response, strict_final=True)
    sections = dict(reasoning=think, final_answer=answer)
    items = slots(rules)
    if not items:
        raise ValueError('Grouped Judge requires at least one original predicate')
    wire=wire_format() if wire is None else wire
    system = _system(wire)
    payload = dict(question_context=rj._grounded_question_context(prompt),
        assistant_evidence_sections={scope: [dict(id=i + 1, text=line)
            for i, line in enumerate(rj._grounded_lines(text))] for scope, text in sections.items()
            if any(s[4] == scope for s in items)},
        rules_to_evaluate=[dict(rule_id=key, category=category, rule_index=index,
            evidence_scope=scope, literal_form=rj._grounded_rule_form(rule), rule=rule)
            for key, category, index, rule, scope in items])
    # Cache syntax templates only, never an answer, Judge output or verdict.
    key = tuple((category, tuple(values)) for category, values in rules.all_rules.items())
    counts = tuple((scope, len(rj._grounded_lines(text))) for scope, text in sections.items())
    guidance = json.loads(_guidance_json(key, counts, wire))
    return system, json.dumps(payload, ensure_ascii=False), guidance, sections


@lru_cache(maxsize=2)
def _system(wire='object'):
    from verl.utils.reward_score import rubric_judge as rj
    # Keep the literal predicate instructions; only transport is grouped.
    single = rj._single_rule_set('outcome_useful_rule', 'schema-only placeholder')
    system, _ = rj._grounded_request('', '', single, 'final_answer')
    system = system.replace('Judge exactly the unchanged rule_to_evaluate against assistant_evidence_lines.',
        'Judge every unchanged predicate in rules_to_evaluate independently against its designated evidence section.')
    system += ('\nReturn exactly one object keyed by the supplied rule_id values. Each value has exactly '
        'evidence_line_ids, reason, predicate_truth, score. Each rule may cite ONLY its own evidence_scope; '
        'line IDs restart at 1 in each section. Reasoning is not evidence of final-answer content. '
        'A result on one rule must not determine any other result. Omit no rule and add no rule.')
    if wire=='tuple':
        system=system.replace('Return exactly one object keyed by the supplied rule_id values. Each value has exactly '
            'evidence_line_ids, reason, predicate_truth, score.',
            'Return one array with exactly one row per rule in rules_to_evaluate order. Each row is exactly '
            '[evidence_line_ids, reason, predicate_truth, score]. The four positions have those same meanings; '
            'include the full reason, not a shortened substitute. Never omit or reorder a row.')
    elif wire!='object':
        raise ValueError('Unknown grouped wire format')
    return system


@lru_cache(maxsize=256)
def _guidance_json(rule_key, counts, wire='object'):
    from verl.utils.reward_score import rubric_judge as rj
    rules = rj.RubricRules(**dict(rule_key))
    sections = {scope: '\n' * count for scope, count in counts}
    items = slots(rules)
    properties = {key: rj._grounded_guided_body(sections[scope],
        rj._single_rule_set(category, rule))['guided_json']
        for key, category, index, rule, scope in items}
    guidance = dict(chat_template_kwargs={'enable_thinking': False}, seed=0,
        guided_json=dict(type='object', additionalProperties=False,
            **{'x-guidance': {'whitespace_flexible': False}},
            properties=properties, required=list(properties)))
    if wire=='tuple':
        names=('evidence_line_ids','reason','predicate_truth','score')
        tuple_rows=[]
        for schema in properties.values():
            branches=[]
            for branch in schema['anyOf']:
                branches.append(dict(items=[branch['properties'].get(name,{}) for name in names]))
            tuple_rows.append(dict(type='array',minItems=4,maxItems=4,
                items=[schema['properties'][name] for name in names],anyOf=branches,
                **{'x-guidance':{'whitespace_flexible':False}}))
        guidance['guided_json']=dict(type='array',minItems=len(tuple_rows),maxItems=len(tuple_rows),
            items=tuple_rows,**{'x-guidance':{'whitespace_flexible':False}})
    elif wire!='object':
        raise ValueError('Unknown grouped wire format')
    return json.dumps(guidance, ensure_ascii=False)


def parse(raw):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError('Duplicate grouped Judge JSON key')
            obj[key] = value
        return obj
    return json.loads(raw, object_pairs_hook=unique)


def output(obj, rules, sections, wire='object'):
    from verl.utils.reward_score import rubric_judge as rj
    items = slots(rules)
    if wire=='tuple':
        if (not isinstance(obj,list) or len(obj)!=len(items)
                or any(not isinstance(row,list) or len(row)!=4 for row in obj)):
            raise ValueError('Compact grouped verdict must preserve every full four-field row')
        names=('evidence_line_ids','reason','predicate_truth','score')
        obj={slot[0]:dict(zip(names,row)) for slot,row in zip(items,obj)}
    elif wire!='object':
        raise ValueError('Unknown grouped wire format')
    if not isinstance(obj, dict) or set(obj) != {s[0] for s in items}:
        raise ValueError('Grouped verdict must contain every original rule exactly once')
    merged = rj._empty_judge_output(rules)
    for key, category, index, rule, scope in items:
        row = rj._grounded_output(obj[key], rj._single_rule_set(category, rule), sections[scope])
        merged[category][index] = row[category][0]
    return dict(merged, _parse_valid=True)


def original_request(payload, rules):
    """Reconstruct exact original evidence and rule binding, without a verdict."""
    from verl.utils.reward_score import rubric_judge as rj
    if set(payload) != {'question_context', 'assistant_evidence_sections', 'rules_to_evaluate'}:
        raise ValueError('Unknown grouped request shape')
    items = slots(rules)
    sections = {}
    if set(payload['assistant_evidence_sections']) != {s[4] for s in items}:
        raise ValueError('Grouped evidence scopes differ')
    for scope, lines in payload['assistant_evidence_sections'].items():
        if (not isinstance(lines, list) or any(set(row) != {'id', 'text'} for row in lines)
                or [row['id'] for row in lines] != list(range(1, len(lines) + 1))
                or any(type(row['id']) is not int or not isinstance(row['text'], str) for row in lines)):
            raise ValueError('Invalid original evidence lines')
        text = ''.join(row['text'] for row in lines)
        if rj._grounded_lines(text) != [row['text'] for row in lines]:
            raise ValueError('Changed original evidence boundaries')
        sections[scope] = text
    expected = [dict(rule_id=key, category=category, rule_index=index, evidence_scope=scope,
        literal_form=rj._grounded_rule_form(rule), rule=rule)
        for key, category, index, rule, scope in items]
    if payload['rules_to_evaluate'] != expected:
        raise ValueError('Original grouped rules or order changed')
    return sections


def vector(obj, rules, sections, wire='object'):
    result = output(obj, rules, sections, wire)
    return {key: result[key] for key in rules.all_rules}


def verified_event(event):
    from verl.utils.reward_score import rubric_judge as rj
    from verl.utils.training_evidence import fingerprint
    try:
        audit = event['request_audit']; system, user = audit['messages']
        if (audit['grounding_protocol'] != PROTOCOL or not audit['evidence_led']
                or system['role'] != 'system' or user['role'] != 'user'
                or audit['system_sha256'] != fingerprint(system['content'])
                or audit['user_sha256'] != fingerprint(user['content'])
                or audit['temperature'] != 0.0 or audit['request_seed'] != 0
                or not audit['response_id'] or event.get('finish_reason') == 'length'):
            return False
        rules = rj.RubricRules(**event['rules'])
        payload = parse(user['content']); sections = original_request(payload, rules)
        prompt, response = event['prompt'], event['response']
        if (fingerprint(prompt) != audit['original_prompt_sha256']
                or rj._grounded_question_context(prompt) != payload['question_context']
                or fingerprint(payload['question_context']) != audit['question_context_sha256']):
            return False
        think, answer = rj._split_think_answer(response, strict_final=True)
        originals = dict(reasoning=think, final_answer=answer)
        if any(originals[k] != text for k, text in sections.items()):
            return False
        wire=audit.get('wire_format','object')
        _, _, guidance, _ = request(prompt, response, rules, wire)
        if system['content']!=_system(wire):
            return False
        if fingerprint(json.dumps(guidance, sort_keys=True)) != audit['guidance_sha256']:
            return False
        output(parse(event['raw_output']), rules, sections, wire)
        return True
    except (KeyError, ValueError, TypeError, IndexError):
        return False


def proof(conflict, request_value, rules):
    return dict(schema_version=2, protocol=PROTOCOL, request=copy.deepcopy(request_value),
                conflict=copy.deepcopy(conflict), rules=copy.deepcopy(rules.all_rules))


def verified_proof(proof_value, out, rule_values):
    from verl.utils.reward_score import rubric_judge as rj
    from verl.utils.training_evidence import fingerprint
    try:
        if (proof_value['schema_version'] != 2 or proof_value['protocol'] != PROTOCOL
                or proof_value['rules'] != rule_values or out.get('_judge_failed') is not True):
            return False
        req, conflict = proof_value['request'], proof_value['conflict']
        if (set(req) != {'model', 'system', 'user', 'guidance', 'max_tokens', 'temperature'}
                or req['temperature'] != 0.0
                or fingerprint(json.dumps(req, sort_keys=True)) != conflict['request_sha256']):
            return False
        rules = rj.RubricRules(**rule_values)
        sections = original_request(parse(req['user']), rules)
        wire=guidance_wire(req['guidance'])
        key=tuple((category,tuple(values)) for category,values in rules.all_rules.items())
        counts=tuple((scope,len(rj._grounded_lines(text))) for scope,text in sections.items())
        if req['system']!=_system(wire) or req['guidance']!=json.loads(_guidance_json(key,counts,wire)):
            return False
        disagreement = conflict.get('original_disagreement', conflict)
        first, current = conflict['first'], conflict['current']
        if disagreement['first'] != first:
            return False
        for a, b in [(first, current), (disagreement['first'], disagreement['current'])]:
            if not a['response_id'] or not b['response_id'] or a['response_id'] == b['response_id']:
                return False
            for row in (a, b):
                if vector(parse(row['raw_output']), rules, sections, wire) != row['score']:
                    return False
        if disagreement['first']['score'] == disagreement['current']['score']:
            return False
        return all(out[k] == current['score'][k] for k in rule_values)
    except (KeyError, ValueError, TypeError, IndexError):
        return False


def judge(api, prompt, response, rules, extra_info):
    from openai import OpenAI
    from verl.utils.reward_score import rubric_judge as rj
    from verl.utils import consistency_quarantine as cq
    from verl.utils.literal_constraint_guard import protect
    from verl.utils.training_evidence import fingerprint
    prepared_at=time.monotonic()
    wire=wire_format()
    system, user, guidance, sections = request(prompt, response, rules, wire)
    preprocessing_seconds=time.monotonic()-prepared_at
    max_tokens = int(os.getenv('RUBRIC_JUDGE_MAX_NEW_TOKENS', '512')) * len(slots(rules))
    max_attempts = rj._api_max_retries() + 1
    api._call_idx += 1
    for attempt in range(1, max_attempts + 1):
        raw = None; finish = None; audit = None; attempt_error = None
        started = time.monotonic()
        try:
            endpoint = api._base_url_for_request(prompt, response, attempt)
            from urllib.parse import urlsplit
            parsed = urlsplit(endpoint)
            audit = dict(endpoint_host=parsed.hostname, endpoint_port=parsed.port,
                configured_granularity=rj._judge_granularity(), effective_granularity='all_rules',
                evidence_led=True, grounding_protocol=PROTOCOL, wire_format=wire, grouped_rule_count=len(slots(rules)),
                messages=[dict(role='system', content=system), dict(role='user', content=user)],
                chat_template_kwargs=guidance['chat_template_kwargs'],
                system_sha256=fingerprint(system), user_sha256=fingerprint(user), temperature=0.0,
                max_tokens=max_tokens, request_seed=0, original_prompt_sha256=fingerprint(prompt),
                question_context_sha256=fingerprint(rj._grounded_question_context(prompt)),
                guidance_sha256=fingerprint(json.dumps(guidance, sort_keys=True)))
            result = api._create_admitted_completion(endpoint, api._client_for_base_url(OpenAI, endpoint), audit,
                model=api.model, temperature=0.0, max_tokens=max_tokens,
                messages=audit['messages'], extra_body=guidance)
            raw = result.choices[0].message.content or ''
            finish = getattr(result.choices[0], 'finish_reason', None)
            audit.update(response_id=getattr(result, 'id', None), served_model=getattr(result, 'model', None))
            obj = parse(raw)
            normalized = output(obj, rules, sections, wire)
            if finish == 'length':
                raise ValueError('Truncated grouped verdict is unavailable, never partially accepted')
            api._check_grounded_consistency(system, user, guidance, max_tokens,
                {k: normalized[k] for k in rules.all_rules}, raw, audit)
            normalized = protect(normalized, rules.all_rules, sections.get('final_answer', ''))
            rj._base_judge_attempt(prompt, response, rules, extra_info, raw, api.model, attempt, finish, audit)
            rj._judge_evidence(prompt, response, rules, extra_info, raw, api.model, attempt, time.monotonic() - started)
            return normalized
        except Exception as error:
            attempt_error=type(error).__name__
            capacity_failure = context_capacity_failure(error, audit)
            if capacity_failure is not None:
                audit['context_capacity_failure'] = capacity_failure
            contained = None
            if isinstance(error, rj.JudgeConsistencyError) and cq.enabled():
                req = dict(model=api.model, system=system, user=user, guidance=guidance,
                           max_tokens=max_tokens, temperature=0.0)
                p = proof(audit['consistency'], req, rules)
                contained = dict(normalized, _judge_failed=True, **{cq.KEY: [p]})
                if not verified_proof(p, contained, rules.all_rules):
                    raise RuntimeError('Grouped independent conflict evidence could not be verified') from error
                contained = protect(contained, rules.all_rules, sections.get('final_answer', ''))
                audit['consistency_quarantine'] = p
            rj._base_judge_attempt(prompt, response, rules, extra_info, raw, api.model, attempt, finish, audit, error)
            rj._judge_evidence(prompt, response, rules, extra_info, raw, api.model, attempt, time.monotonic() - started, error)
            if capacity_failure is not None:
                # A deterministic context rejection cannot benefit from retry.
                # Existing whole-prompt-group filtering/whole-validation
                # unavailability handles this; no scalar score is fabricated.
                if os.getenv('RUBRIC_JUDGE_DROP_FAILED', '0') == '1':
                    return dict(rj._empty_judge_output(rules), _parse_valid=False, _judge_failed=True)
                from verl.utils.validation_recovery import enabled as recovery_enabled, JudgeScoringUnavailable
                if recovery_enabled():
                    raise JudgeScoringUnavailable('Grouped Judge context capacity unavailable',
                        dict(prompt=prompt, response=response, rules=rules.all_rules,
                             parse_valid=False, judge_failed=True,
                             context_capacity_failure=capacity_failure, protocol=PROTOCOL)) from error
                raise
            if isinstance(error, rj.JudgeConsistencyError):
                if contained is not None:
                    return contained
                raise
            recoverable = ((isinstance(error, ValueError) and raw is not None)
                or isinstance(error, TimeoutError) or type(error).__name__ in {'APITimeoutError', 'APIConnectionError'}
                or getattr(error, 'status_code', None) in {429, 500, 502, 503, 504})
            if not recoverable:
                raise
            if attempt == max_attempts:
                if os.getenv('RUBRIC_JUDGE_DROP_FAILED', '0') == '1':
                    # Invalid transport/format yields no eligible scalar update.
                    # The existing manager filters the whole training group or
                    # types the entire evaluation as unavailable.
                    return dict(rj._empty_judge_output(rules), _parse_valid=False, _judge_failed=True)
                from verl.utils.validation_recovery import enabled as recovery_enabled, JudgeScoringUnavailable
                if recovery_enabled():
                    raise JudgeScoringUnavailable('Grouped grounded Judge unavailable after finite retries',
                        dict(prompt=prompt, response=response, rules=rules.all_rules, parse_valid=False,
                             judge_failed=True, protocol=PROTOCOL)) from error
                raise RuntimeError('Grouped Judge unavailable; no reward was fabricated') from error
            time.sleep(min(rj._api_retry_backoff() * 2 ** (attempt - 1), rj._api_retry_backoff_max()))
        finally:
            from verl.utils.grouped_dispatch import observe
            value=dict(audit or {})
            observe(dict(endpoint_port=value.get('endpoint_port'),
                local_queue_wait_seconds=value.get('local_queue_wait_seconds'),
                http_seconds=value.get('http_seconds'),token_usage=value.get('token_usage') or {},
                preprocessing_seconds=preprocessing_seconds if attempt==1 else 0.,
                error_type=attempt_error or (None if value.get('response_id') else 'missing_response_identity')))
    raise AssertionError('Unreachable retry state')
