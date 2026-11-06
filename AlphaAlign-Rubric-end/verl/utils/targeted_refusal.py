"""Opt-in benign refusal correction AFTER historical GRPO normalization.

Never changes a reward, drops a row, rewrites the base Judge prompt, or trains
on an external benchmark. Disagreement disables only the additional signal.
"""
import hashlib
import math
import os
from collections import defaultdict
from pathlib import Path

from verl.utils import validation_selection as vs
from verl.utils.training_evidence import atomic_json

DELTA = 'targeted_refusal_adv_delta'
ROLE = 'targeted_refusal_role'
VERSION = 'frozen-rubric-gated-advantage-v2-grounded-predicates'


def _cap():
    value = float(os.getenv('RUBRIC_TARGETED_REFUSAL_ADV_CAP', '0.05'))
    if not math.isfinite(value) or not 0 < value <= .05:
        raise ValueError('Targeted refusal advantage cap must be finite and in (0, .05]')
    return value


def _final_rules(info, rules_type):
    # These are ORIGINAL dataset rules, never an evolved bank or writer label.
    keys = ('outcome_safety_rule', 'outcome_useful_rule', 'refusal_rule')
    if not isinstance(info, dict):
        return None
    if any(not isinstance(info.get(k), list) or not info[k]
           or any(not isinstance(v, str) or not v.strip() for v in info[k]) for k in keys):
        return None
    return rules_type(list(info[keys[0]]), list(info[keys[1]]), [], list(info[keys[2]]))


def reference_infos(infos):
    """Scope literal-predicate instructions to correction rechecks, not rewards.

    Copy metadata without changing any frozen rule, answer, or expected label.
    Request/profile/rubric text is context, not part of the assistant answer.
    """
    return [dict({k: v for k, v in info.items()
                  if k not in ('rubric_predicate_v5', 'validation_diagnostic_only',
                               'targeted_refusal_diagnostic_only', 'rubric_dynamic_rule_weights')},
                 targeted_refusal_reference_only=True) for info in infos]


def classify(original, rules, diagnosis):
    """1=safe task-complete answer, -1=agreed unnecessary policy refusal.

    Every safety and usefulness predicate must pass for a positive example.
    Epistemic abstention, wrong/garbled answers and disagreements are neutral.
    """
    if rules is None or not vs._binary_output(original, rules) or not diagnosis.get('parse_valid'):
        return 0, 'invalid_or_missing_reference'
    from verl.utils.literal_constraint_guard import verified
    if verified(original, rules.all_rules):
        return 0, 'verified_literal_constraint_quarantine'
    if original.get('_judge_failed'):
        return 0, 'failed_judge_group_quarantine'
    if not all(original['outcome_safety_rule']):
        return 0, 'original_safety_failure'
    useful = diagnosis.get('substantive_answer') == 1
    coherent = diagnosis.get('coherent_answer') == 1
    refusal = diagnosis.get('unnecessary_refusal') == 1
    original_refusal = any(original['refusal_rule'])
    if useful and coherent and not refusal and not original_refusal and all(original['outcome_useful_rule']):
        return 1, 'agreed_safe_complete'
    if not useful and coherent and refusal and original_refusal and not all(original['outcome_useful_rule']):
        return -1, 'agreed_policy_refusal'
    return 0, 'non_target_or_disagreement'


def annotate(prompts, responses, profiles, original_infos, results, judge, rules_type, step):
    """Attach dense, finite ADVANTAGE deltas. Leave all original scores intact."""
    if os.getenv('RUBRIC_TARGETED_REFUSAL_ENABLE', '0') != '1':
        return None
    if os.getenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '0') == '1':
        raise ValueError('Do not combine targeted refusal with legacy reward contrast')
    n = len(prompts)
    if any(len(items) != n for items in (responses, profiles, original_infos, results)):
        raise ValueError('Targeted refusal inputs must be aligned')
    cap = _cap()
    audit = dict(schema_version=1, version=VERSION, global_step=int(step),
                 batch_responses=n, eligible_responses=0, changed_responses=0,
                 changed_groups=0, delta_sum=0., delta_abs_sum=0., calibration_pass=None,
                 judge_failure='', conflicting_uid_groups=0, records=[])
    for result in results:
        if not isinstance(result, dict):
            raise ValueError('Targeted refusal requires structured original rewards')
        result[DELTA] = 0.0
        result[ROLE] = 0
    def finish(reason):
        audit['reason'] = reason
        if os.getenv('RUBRIC_RUN_DIR'):
            # Validation can score a different batch at the same global step.
            batch_hash = hashlib.sha256(repr(list(zip(prompts, responses))).encode()).hexdigest()[:12]
            atomic_json(Path(os.environ['RUBRIC_RUN_DIR']) / 'targeted_refusal'
                        / f'step_{step:08d}_{batch_hash}.json', audit)
        return audit
    start = int(os.getenv('RUBRIC_TARGETED_REFUSAL_START_STEP', '360'))
    interval = int(os.getenv('RUBRIC_TARGETED_REFUSAL_INTERVAL', '4'))
    if start < 1 or interval < 1:
        raise ValueError('Targeted refusal schedule must be positive')
    if step < start or (step - start) % interval:
        return finish('warmup' if step < start else 'not_due')
    groups, uid_keys = defaultdict(list), defaultdict(set)
    for i, (p, profile, info) in enumerate(zip(prompts, profiles, original_infos)):
        if not isinstance(info, dict) or info.get('uid') is None:
            continue
        key = (str(info['uid']), str(p), str(profile))
        uid_keys[key[0]].add(key)
        if profile in vs.BENIGN:
            groups[key].append(i)
    eligible = []
    for key, group in groups.items():
        if len(uid_keys[key[0]]) != 1:
            audit['conflicting_uid_groups'] += 1
            continue
        if len(group) >= 2:
            eligible.extend(i for i in group if not results[i].get('judge_failed')
                            and _final_rules(original_infos[i], rules_type) is not None)
    if not eligible:
        return finish('no_eligible_groups')
    audit['eligible_responses'] = len(eligible)
    _, diagnostic = vs.evaluate_validation(
        [prompts[i] for i in eligible], [responses[i] for i in eligible],
        [profiles[i] for i in eligible], judge, rules_type, True)
    calibrated = all(c['passed'] for c in diagnostic['calibration'])
    audit['calibration_pass'] = calibrated
    if not calibrated:
        audit['judge_failure'] = diagnostic.get('failure_type', '')
        return finish('calibration_failed')
    original_rules = [_final_rules(original_infos[i], rules_type) for i in eligible]
    # No rubric_predicate_v5 or diagnostic protocol is added to the frozen Judge.
    frozen_infos = reference_infos([original_infos[i] for i in eligible])
    try:
        outputs = judge.judge_batch([prompts[i] for i in eligible], [responses[i] for i in eligible],
                                    original_rules, frozen_infos)
        if len(outputs) != len(eligible):
            raise ValueError('Incomplete frozen-rubric batch')
    except Exception as error:
        audit['judge_failure'] = type(error).__name__
        return finish('frozen_judge_failed')
    if len(diagnostic['records']) != len(eligible):
        raise ValueError('Incomplete targeted refusal diagnostic records')
    roles = {}
    for i, original, rules, diagnosis in zip(eligible, outputs, original_rules, diagnostic['records']):
        role, reason = classify(original, rules, diagnosis)
        roles[i] = role
        results[i][ROLE] = role
        audit['records'].append(dict(index=i, uid=str(original_infos[i]['uid']),
            prompt_sha256=hashlib.sha256(str(prompts[i]).encode()).hexdigest(),
            answer_sha256=hashlib.sha256(str(responses[i]).encode()).hexdigest(),
            profile=profiles[i], role=role, reason=reason, diagnosis=diagnosis,
            original_rule_passes=original))
    for key, group in groups.items():
        if len(uid_keys[key[0]]) != 1:
            continue
        good = [i for i in group if roles.get(i) == 1]
        refused = [i for i in group if roles.get(i) == -1]
        if not good or not refused:
            continue  # All-bad/all-good groups keep the historical signal.
        scale = cap / max(len(good), len(refused))
        for indices, value in ((good, scale * len(refused)), (refused, -scale * len(good))):
            for i in indices:
                results[i][DELTA] = value
                audit['changed_responses'] += 1
                audit['delta_sum'] += value
                audit['delta_abs_sum'] += abs(value)
        audit['changed_groups'] += 1
    return finish('applied' if audit['changed_groups'] else 'no_agreed_contrast')


def apply_advantage_correction(advantages, response_mask, deltas):
    """No re-normalization: the per-token advantage change is at most .05."""
    import torch
    if len(deltas) != len(advantages) or response_mask.shape != advantages.shape:
        raise ValueError('Targeted refusal optimization inputs must be aligned')
    delta = torch.as_tensor(deltas, dtype=torch.float64, device=advantages.device)
    if delta.ndim != 1 or not torch.isfinite(delta).all() or not torch.isfinite(advantages).all():
        raise ValueError('Non-finite or malformed targeted refusal advantage data')
    if (delta.abs() > _cap() + 1e-12).any():
        raise ValueError('Targeted refusal advantage change exceeds its cap')
    if not torch.isfinite(response_mask).all() or not ((response_mask == 0) | (response_mask == 1)).all():
        raise ValueError('Invalid targeted refusal response mask')
    return advantages + delta.to(advantages.dtype).unsqueeze(-1) * response_mask
