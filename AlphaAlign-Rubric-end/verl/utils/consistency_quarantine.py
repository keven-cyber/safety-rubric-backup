"""Preserve independently conflicting Judge verdicts and exclude their groups.

This is evidence of disagreement, never a replacement semantic verdict.
"""
import copy
import json
import os

from verl.utils.training_evidence import fingerprint

KEY = '_verified_consistency_conflicts'


def enabled():
    return os.getenv('RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE', '0') == '1'


def make_proof(conflict, request, rules, extra_info):
    info = extra_info if isinstance(extra_info, dict) else {}
    category = next(k for k, items in rules.all_rules.items() if items)
    return dict(schema_version=1, request=copy.deepcopy(request),
                conflict=copy.deepcopy(conflict), category=category,
                rule_index=0,
                source_rule_index=int(info.get('grounding_rule_index', 0)),
                rule=rules.all_rules[category][0])


def verified(output, rules):
    """Reparse BOTH originals and reproduce request identity and rule binding."""
    if not isinstance(output, dict) or not output.get(KEY):
        return False
    from verl.utils.reward_score import rubric_judge as rj
    try:
        for proof in output[KEY]:
            if proof.get('protocol') == 'legacy_compact_v8':
                from verl.utils.legacy_compact_judge import verified_proof
                if not verified_proof(proof, output, rules):
                    return False
                continue
            if proof.get('protocol') == 'grouped_grounded_v1':
                from verl.utils.grouped_grounded_judge import verified_proof
                if not verified_proof(proof, output, rules):
                    return False
                continue
            request, conflict = proof['request'], proof['conflict']
            if proof['schema_version'] != 1 or set(request) != {
                    'model', 'system', 'user', 'guidance', 'max_tokens', 'temperature'}:
                return False
            if fingerprint(json.dumps(request, sort_keys=True)) != conflict['request_sha256']:
                return False
            if request['temperature'] != 0.0 or not isinstance(request['model'], str):
                return False
            category, index = proof['category'], proof['rule_index']
            if (category not in rules or type(index) is not int or not 0 <= index < len(rules[category])
                    or rules[category][index] != proof['rule']):
                return False
            user = json.loads(request['user'])
            if user['rule_to_evaluate'] != proof['rule']:
                return False
            lines = user['assistant_evidence_lines']
            if [row['id'] for row in lines] != list(range(1, len(lines) + 1)):
                return False
            text = ''.join(row['text'] for row in lines)
            if rj._grounded_lines(text) != [row['text'] for row in lines]:
                return False
            scoped = rj._single_rule_set(category, proof['rule'])
            first, current = conflict['first'], conflict['current']
            if (not first['response_id'] or not current['response_id']
                    or first['response_id'] == current['response_id']):
                return False
            for observation in (first, current):
                raw = json.loads(observation['raw_output'])
                rj._grounded_output(raw, scoped, text)
                if raw['score'] != observation['score']:
                    return False
            # Once tainted, later independent repeats are also excluded even if
            # their bit agrees with the first. The original disagreement stays.
            disagreement = conflict.get('original_disagreement', conflict)
            if disagreement['first'] != first:
                return False
            for observation in (disagreement['first'], disagreement['current']):
                raw = json.loads(observation['raw_output'])
                rj._grounded_output(raw, scoped, text)
                if raw['score'] != observation['score']:
                    return False
            if (not disagreement['current']['response_id'] or
                    disagreement['current']['response_id'] == first['response_id']):
                return False
            if disagreement['first']['score'] == disagreement['current']['score']:
                return False
            if output[category][index] != current['score']:
                return False
        return True
    except (KeyError, TypeError, ValueError, IndexError, StopIteration):
        return False


def merge(target, source, category=None, index=None):
    if source.get(KEY):
        proofs = copy.deepcopy(source[KEY])
        if category is not None:
            for proof in proofs:
                if proof['category'] != category:
                    raise ValueError('Consistency quarantine category is misaligned')
                proof['rule_index'] = index
        target.setdefault(KEY, []).extend(proofs)
        target['_judge_failed'] = True


def verified_audit_event(event):
    """Validate an archived independent request/response without trusting labels."""
    from verl.utils.reward_score import rubric_judge as rj
    try:
        audit = event['request_audit']
        if audit.get('grounding_protocol') == 'legacy_compact_v8':
            from verl.utils.legacy_compact_judge import verified_event
            return verified_event(event)
        if audit.get('grounding_protocol') == 'grouped_grounded_v1':
            from verl.utils.grouped_grounded_judge import verified_event
            return verified_event(event)
        system, user = audit['messages']
        if (system['role'] != 'system' or user['role'] != 'user'
                or fingerprint(system['content']) != audit['system_sha256']
                or fingerprint(user['content']) != audit['user_sha256']
                or not audit['response_id'] or audit['temperature'] != 0.0
                or audit['request_seed'] != 0 or not audit['evidence_led']):
            return False
        payload = json.loads(user['content'])
        prompt = event.get('prompt')
        if prompt is None and event.get('event') == 'targeted_judge_attempt':
            # Older diagnostic traces retained the original prompt's fingerprint,
            # not its text. A plain question can be recovered from the actual
            # request only when BOTH archived fingerprints prove exact identity.
            # Wrapped/missing/changed prompts remain unknown and fail closed.
            prompt = payload.get('question_context')
            if (not isinstance(prompt, str)
                    or fingerprint(prompt) != event.get('prompt_sha256')):
                return False
        if (not isinstance(prompt, str)
                or fingerprint(prompt) != audit['original_prompt_sha256']
                or payload.get('question_context') != rj._grounded_question_context(prompt)
                or fingerprint(payload['question_context']) != audit['question_context_sha256']):
            return False
        rules = rj.RubricRules(**event['rules'])
        active = [item for items in rules.all_rules.values() for item in items]
        if active != [payload['rule_to_evaluate']]:
            return False
        lines = payload['assistant_evidence_lines']
        if [row['id'] for row in lines] != list(range(1, len(lines) + 1)):
            return False
        evidence = ''.join(row['text'] for row in lines)
        if rj._grounded_lines(evidence) != [row['text'] for row in lines]:
            return False
        rj._grounded_output(json.loads(event['raw_output']), rules, evidence)
        return True
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def audit_directory(directory):
    from verl.utils.reward_score import rubric_judge as rj
    first, conflicts, repeats = {}, [], 0
    for file in sorted(directory.rglob('*.jsonl')):
        for number, line in enumerate(file.read_text().splitlines(), 1):
            event = json.loads(line)
            audit = event.get('request_audit') or {}
            if (event.get('error_type') not in (None, 'JudgeConsistencyError')
                    or not (audit.get('evidence_led') or audit.get('grounding_protocol') == 'legacy_compact_v8')):
                continue
            raw = json.loads(event['raw_output'])
            key = fingerprint(json.dumps({k: audit.get(k) for k in (
                'system_sha256', 'user_sha256', 'guidance_sha256', 'max_tokens',
                'temperature', 'request_seed', 'served_model')}, sort_keys=True))
            if audit.get('grounding_protocol') == 'legacy_compact_v8':
                from verl.utils.legacy_compact_judge import vector
                score = vector(event['raw_output'], rj.RubricRules(**event['rules']))
            elif audit.get('grounding_protocol') == 'grouped_grounded_v1':
                from verl.utils.grouped_grounded_judge import vector, original_request
                rule_set = rj.RubricRules(**event['rules'])
                score = vector(raw, rule_set, original_request(json.loads(audit['messages'][1]['content']), rule_set), audit.get('wire_format','object'))
            else:
                score = raw['score']
            current = dict(score=score, raw_output=event['raw_output'],
                response_id=audit['response_id'], file=str(file), line=number,
                prompt_sha256=audit.get('original_prompt_sha256'),
                case_ids=event.get('case_ids', []), evidence_verified=verified_audit_event(event))
            if key in first:
                repeats += 1
                previous = first[key]
                if current['score'] != previous['score'] or current['response_id'] == previous['response_id']:
                    verified_pair = (current['evidence_verified'] and previous['evidence_verified']
                        and current['response_id'] != previous['response_id'])
                    conflicts.append(dict(request_sha256=key, first=previous,
                                          current=current, evidence_verified=verified_pair))
            else:
                first[key] = current
    return dict(passed=not conflicts and repeats > 0, independent_repeats_checked=repeats,
                conflicts=conflicts, cached_verdicts_returned=0)


def tainted_prompts(audit):
    return {observation['prompt_sha256'] for conflict in audit['conflicts']
            if conflict.get('evidence_verified') for observation in
            (conflict['first'], conflict['current'])}


def containment_verified(audit, rows):
    """Every witnessed conflict must map to preserved rows with zero update weight."""
    if not audit['conflicts'] or not all(c.get('evidence_verified') for c in audit['conflicts']):
        return False
    for prompt in tainted_prompts(audit):
        affected = [row for row in rows if row['prompt_sha256'] == prompt]
        if not affected or not all(row['quarantined'] and row['effective_update_weight'] == 0
                                   and row['update_advantage'] == 0 for row in affected):
            return False
    return True
