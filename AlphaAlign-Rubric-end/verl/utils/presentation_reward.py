"""Opt-in attenuation of presentation-only reward; never changes a verdict.

Unknown or substantive criteria keep their historical contribution. Recognition
uses the written rule only, never case IDs, answers, expected labels or profiles.
"""
import math
import os
import re
from collections import defaultdict

COMPONENT = 'presentation_reward_component'
SCORED = 'presentation_scored_reward'
CONTENT = 'presentation_content_reward'


def enabled():
    return os.getenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '0') == '1'


def settings():
    factor = float(os.getenv('RUBRIC_PRESENTATION_REWARD_FACTOR', '0.20'))
    cap = float(os.getenv('RUBRIC_PRESENTATION_ADV_CAP', '0.05'))
    if not math.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('Presentation factor must be finite and in [0, 1]')
    if not math.isfinite(cap) or not 0 <= cap <= .05:
        raise ValueError('Presentation advantage cap must be finite and in [0, .05]')
    return factor, cap


def require_balanced_mode(mode):
    if enabled() and mode != 'rubric_safe_helpful_balanced':
        raise ValueError('Presentation damping currently requires balanced rubric rewards')


def is_presentation_only(rule):
    """Conservative literal recognizer; task explanations are NOT presentation.

    A combined "clear and concise, using bullets" predicate is treated as one
    low-reliability presentation signal. Its raw 0/1 judgment remains unchanged.
    """
    text = re.sub(r'\s+', ' ', str(rule)).strip().lower().rstrip('.')
    text = re.sub(r'^the (?:final )?(?:answer|response) ', '', text)
    if re.search(r'\b(?:accurat\w*|correct\w*|fact\w*|calculat\w*|equation\w*|'
                 r'explain\w*|explanation\w*|reasoning|method\w*|safe\w*|harm\w*|'
                 r'illegal|refus\w*|request\w*|substantiv\w*|all|every)\b', text):
        return False
    prefix = re.match(r'^(?:presents? (?:the )?(?:information|answer|response)\b|'
                      r'(?:is|remains) (?:clear|concise|well[- ]structured)\b|'
                      r'uses? (?:bullets?|bullet points?|brief paragraphs?)\b)', text)
    return bool(prefix and re.search(r'\b(?:clearly|clear|concisely|concise|scannable|'
                                    r'bullets?|paragraphs?|structured)\b', text))


def positive_fraction(rules, values, weights):
    """Keep the original denominator, so removing credit cannot raise a score."""
    if len(rules) != len(values) or len(weights) != len(rules):
        raise ValueError('Presentation accounting requires complete aligned rules')
    if any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError('Presentation accounting requires original binary verdicts')
    if any(not math.isfinite(float(w)) or float(w) < 0 for w in weights):
        raise ValueError('Presentation weights must be finite and nonnegative')
    indices = [i for i, rule in enumerate(rules) if is_presentation_only(rule)]
    denominator = sum(weights)
    if denominator <= 0:
        denominator = len(rules)
        weights = [1.] * len(rules)
    fraction = (sum(weights[i] * values[i] for i in indices) / denominator
                if denominator else 0.)
    return fraction, indices


def after_format(components, format_weight, final_score):
    """Expose the exact style contribution in the final scalar reward units."""
    if COMPONENT not in components:
        return
    if not math.isfinite(format_weight) or not 0 <= format_weight <= 1:
        raise ValueError('Presentation accounting requires format weight in [0, 1]')
    components['presentation_balanced_component'] = components[COMPONENT]
    components[COMPONENT] *= 1.0 - format_weight
    components['presentation_reward_removed'] *= 1.0 - format_weight
    components[SCORED] = float(final_score)
    components[CONTENT] = float(final_score) - components[COMPONENT]


def quantized_components(scored, content, components, reference):
    """Subtract quantized endpoints, avoiding tiny fake variance after removal."""
    import torch
    if not (len(scored) == len(content) == len(components) == len(reference)):
        raise ValueError('Presentation endpoints must align with the reward batch')
    for a, b, c in zip(scored, content, components):
        if (any(isinstance(x, bool) or not math.isfinite(float(x)) for x in (a, b, c))
                or float(c) < 0 or abs((float(a) - float(b)) - float(c)) > 1e-8):
            raise ValueError('Presentation endpoints do not match their scalar contribution')
    a = torch.as_tensor(scored, dtype=reference.dtype, device=reference.device)
    b = torch.as_tensor(content, dtype=reference.dtype, device=reference.device)
    return a - b


def bounded_grpo(token_rewards, response_mask, index, current_advantages,
                 components, epsilon, std_floor, cap):
    """Bound the presentation-only contribution AFTER GRPO normalization.

    Compare actual damped rewards with the same rewards minus their remaining
    presentation contribution. Scale the difference uniformly within each prompt
    group. This preserves its zero sum and caps it without changing content GRPO.
    """
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
    if not math.isfinite(cap) or not 0 <= cap <= .05:
        raise ValueError('Invalid presentation advantage cap')
    if len(components) != len(token_rewards) or len(index) != len(token_rewards):
        raise ValueError('Missing or misaligned presentation reward accounting')
    if any(isinstance(v, bool) for v in components):
        raise ValueError('Boolean presentation components are invalid')
    values = torch.as_tensor(components, dtype=token_rewards.dtype, device=token_rewards.device)
    if values.ndim != 1 or not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError('Invalid presentation reward accounting')
    if (response_mask.shape != token_rewards.shape or
            not torch.isfinite(token_rewards).all() or
            not torch.isfinite(response_mask).all() or
            not ((response_mask == 0) | (response_mask == 1)).all()):
        raise ValueError('Invalid rewards or response mask')
    if ((response_mask.sum(-1) == 0) & (values != 0)).any():
        raise ValueError('An empty answer cannot carry presentation reward')
    # FP64 operates on already quantized token values. This avoids the historical
    # FP32 mean/std roundoff creating nonzero advantages for a tied group.
    without = token_rewards.double().clone()
    last = (response_mask * torch.arange(token_rewards.shape[1],
            device=token_rewards.device)).max(-1).values.long()
    without[torch.arange(len(values), device=values.device), last] -= values.double()
    current_advantages, _ = compute_grpo_outcome_advantage(
        token_rewards.double(), response_mask.double(), index,
        epsilon=epsilon, std_floor=std_floor)
    baseline, _ = compute_grpo_outcome_advantage(
        without, response_mask.double(), index, epsilon=epsilon, std_floor=std_floor)
    delta = current_advantages - baseline
    groups = defaultdict(list)
    for i, uid in enumerate(index):
        groups[uid].append(i)
    for group in groups.values():
        maximum = delta[group].abs().max()
        scale = min(1., cap / (float(maximum) + epsilon))
        delta[group] *= scale
    result = ((baseline + delta) * response_mask).to(token_rewards.dtype)
    if not torch.isfinite(result).all():
        raise ValueError('Nonfinite presentation-bounded advantages')
    return result, result
