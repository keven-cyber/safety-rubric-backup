"""Opt-in, calibrated within-prompt correction; never labels harmful requests."""
import math
import os
from collections import defaultdict
from pathlib import Path

from verl.utils.validation_selection import BENIGN, evaluate_validation
from verl.utils.training_evidence import atomic_json


def validate_optimization_tensors(**tensors):
    """Fail before optimization on real NaN/Inf, including masked positions."""
    import torch

    for name, tensor in tensors.items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'Non-finite optimization tensor: {name}')


def optimization_statistics(uids, deltas, advantages, reward_trace_info=None):
    """Measure defined diagnostics, never infer applicability from finiteness.

    Dense reward extras pad absent fields with NaN. Only the original row's
    *absence of the key* authorizes skipping that placeholder. An explicitly
    returned NaN/Inf delta or a non-finite advantage is still an error.
    """
    if not (len(uids) == len(deltas) == len(advantages)):
        raise ValueError('Contrast optimization statistics must be aligned')
    if reward_trace_info is not None and len(reward_trace_info) != len(uids):
        raise ValueError('Contrast reward traces must be aligned')
    observed = []
    for i, (uid, delta, advantage) in enumerate(zip(uids, deltas, advantages)):
        advantage = float(advantage)
        if not math.isfinite(advantage):
            raise ValueError(f'Non-finite contrast optimization advantage at row {i}')
        present = delta is not None
        if reward_trace_info is not None:
            row = reward_trace_info[i]
            if not isinstance(row, dict):
                raise ValueError(f'Invalid contrast reward trace at row {i}')
            present = 'benign_contrast_delta' in row
            if present:
                raw_delta = float(row['benign_contrast_delta'])
                if not math.isfinite(raw_delta):
                    raise ValueError(f'Non-finite contrast optimization delta at row {i}')
                if delta is None or float(delta) != raw_delta:
                    raise ValueError(f'Contrast diagnostic and reward trace disagree at row {i}')
            elif delta is not None and not math.isnan(float(delta)):
                raise ValueError(f'Unexpected contrast diagnostic on absent row {i}')
        if not present:
            continue
        delta = float(delta)
        if not math.isfinite(delta):
            raise ValueError(f'Non-finite contrast optimization delta at row {i}')
        observed.append((uid, delta, advantage))
    changed = [(u, d, a) for u, d, a in observed if abs(d) > 1e-12]
    return {'benign_contrast/observed_responses': len(observed),
            'benign_contrast/not_applicable_responses': len(uids) - len(observed),
            'benign_contrast/changed_responses': len(changed),
            'benign_contrast/changed_groups': len({u for u, _, _ in changed}),
            'benign_contrast/delta_sum': sum(d for _, d, _ in observed),
            'benign_contrast/delta_abs_sum': sum(abs(d) for _, d, _ in observed),
            'benign_contrast/positive_delta_positive_advantage': sum(d > 0 and a > 0 for _, d, a in changed),
            'benign_contrast/negative_delta_negative_advantage': sum(d < 0 and a < 0 for _, d, a in changed)}


def apply_benign_contrast(prompts, responses, profiles, extras, results, judge, rules_type, step):
    if os.getenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '0') != '1':
        return
    if not (len(prompts) == len(responses) == len(profiles) == len(extras) == len(results)):
        raise ValueError('Benign contrast inputs must be aligned')
    audit = dict(schema_version=1, global_step=int(step), full_batch=True,
                 batch_responses=len(results), benign_responses=sum(p in BENIGN for p in profiles),
                 eligible_groups=0, eligible_responses=0, mixed_groups=0,
                 all_good_groups=0, refusal_only_groups=0, no_usable_contrast_groups=0,
                 invalid_groups=0, changed_groups=0, changed_responses=0,
                 missing_uid_responses=0, failed_judge_responses=0, singleton_groups=0,
                 good_responses=0, refused_responses=0, other_responses=0,
                 delta_sum=0.0, delta_abs_sum=0.0, calibration_pass=None,
                 profiles={p: dict(eligible_groups=0, mixed_groups=0, changed_responses=0) for p in sorted(BENIGN)})
    def finish(reason):
        audit['reason'] = reason
        if os.getenv('RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE', '0') == '1' and os.getenv('RUBRIC_RUN_DIR'):
            atomic_json(Path(os.environ['RUBRIC_RUN_DIR']) / 'benign_contrast' / f'step_{step:08d}.json', audit)
        return audit
    start = int(os.getenv('RUBRIC_BENIGN_CONTRAST_START_STEP', '360'))
    interval = int(os.getenv('RUBRIC_BENIGN_CONTRAST_INTERVAL', '4'))
    if interval < 1:
        raise ValueError('Benign contrast interval must be positive')
    if step < start or (step - start) % interval:
        return finish('warmup' if step < start else 'not_due')
    strength = float(os.getenv('RUBRIC_BENIGN_CONTRAST_WEIGHT', '.10'))
    if not math.isfinite(strength) or not 0 <= strength <= .15:
        raise ValueError('Benign contrast weight must be finite and in [0, .15]')
    groups = {}
    for i, (prompt, profile, extra, result) in enumerate(zip(prompts, profiles, extras, results)):
        if profile not in BENIGN:
            continue
        if not isinstance(extra, dict) or extra.get('uid') is None:
            audit['missing_uid_responses'] += 1
            continue
        if not isinstance(result, dict) or result.get('judge_failed'):
            audit['failed_judge_responses'] += 1
            continue
        # UID alone is insufficient if malformed input accidentally reuses it.
        groups.setdefault((str(extra['uid']), str(prompt), profile), []).append(i)
    indices = [i for group in groups.values() if len(group) >= 2 for i in group]
    audit['eligible_groups'] = sum(len(g) >= 2 for g in groups.values())
    audit['singleton_groups'] = sum(len(g) < 2 for g in groups.values())
    audit['eligible_responses'] = len(indices)
    if not indices:
        return finish('no_eligible_groups')
    _, report = evaluate_validation([prompts[i] for i in indices], [responses[i] for i in indices],
                                    [profiles[i] for i in indices], judge, rules_type)
    calibrated = all(c['passed'] for c in report['calibration'])
    audit['calibration_pass'] = calibrated
    for i in indices:
        results[i]['benign_contrast_calibrated'] = float(calibrated)
        results[i]['benign_contrast_delta'] = 0.0
    if not calibrated:
        # Keep original reward on service failure, malformed output or canary failure.
        return finish('calibration_failed')
    flags = dict(zip(indices, report['records']))
    for (_, _, profile), group in groups.items():
        if len(group) < 2:
            continue
        audit['profiles'][profile]['eligible_groups'] += 1
        if any(not flags[i]['parse_valid'] for i in group):
            audit['invalid_groups'] += 1
            continue
        good = [i for i in group if flags[i]['unnecessary_refusal'] == 0
                and flags[i]['substantive_answer'] == 1 and flags[i]['coherent_answer'] == 1]
        refused = [i for i in group if flags[i]['unnecessary_refusal'] == 1
                   and flags[i]['substantive_answer'] == 0]
        audit['good_responses'] += len(good)
        audit['refused_responses'] += len(refused)
        audit['other_responses'] += len(group) - len(good) - len(refused)
        if not good or not refused:
            key = ('all_good_groups' if len(good) == len(group) else
                   'refusal_only_groups' if len(refused) == len(group) else 'no_usable_contrast_groups')
            audit[key] += 1
            continue
        audit['mixed_groups'] += 1
        audit['profiles'][profile]['mixed_groups'] += 1
        # Zero-sum within this prompt; no all-bad group gets a positive signal.
        # Wrong/garbled attempts stay untouched, not relabeled as refusal.
        scale = strength / max(len(good), len(refused))
        changes = [(i, scale * len(refused)) for i in good]
        changes += [(i, -scale * len(good)) for i in refused]
        for i, delta in changes:
            results[i]['score'] += delta
            results[i]['benign_contrast_delta'] = delta
        audit['changed_groups'] += int(any(abs(d) > 1e-12 for _, d in changes))
        audit['changed_responses'] += sum(abs(d) > 1e-12 for _, d in changes)
        audit['profiles'][profile]['changed_responses'] += sum(abs(d) > 1e-12 for _, d in changes)
        audit['delta_sum'] += sum(d for _, d in changes)
        audit['delta_abs_sum'] += sum(abs(d) for _, d in changes)
    return finish('applied' if audit['changed_groups'] else 'no_mixed_contrast')
