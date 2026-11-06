"""Lossless Judge dispatch helpers; never cache or edit a model verdict."""
from collections import Counter, deque


def interleave(indices, route):
    """Keep each endpoint's FIFO order while avoiding a long single-route prefix."""
    queues = {}
    for index in indices:
        queues.setdefault(route(index), deque()).append(index)
    pending = deque(queues.values())
    ordered = []
    while pending:
        queue = pending.popleft()
        ordered.append(queue.popleft())
        if queue:
            pending.append(queue)
    return ordered


def update_projection(report):
    """Compare every raw bit and effective update, excluding diagnostic IDs/time."""
    categories = ('outcome_safety_rule', 'outcome_useful_rule',
                  'reasoning_process_rule', 'refusal_rule')
    def reward(value):
        passes = value['rubric_rule_passes']
        return dict(score=value['score'], judge_failed=value.get('judge_failed', False),
                    raw={key: passes[key] for key in categories},
                    parse_valid=passes.get('_parse_valid'),
                    presentation={key: value.get(key) for key in (
                        'presentation_reward_component', 'presentation_scored_reward',
                        'presentation_content_reward')})
    return dict(rows=[dict(id=row['id'], profile=row['profile'],
        baseline_reward=reward(row['baseline_reward']),
        damped_reward=reward(row['damped_reward']),
        prompt_group_sha256=row.get('prompt_group_sha256'),
        quarantined=row['quarantined'], effective_update_weight=row['effective_update_weight'],
        update_advantage=row['update_advantage']) for row in report['rows']],
        advantages=report['advantages'], presentation_contribution=report['presentation_contribution'],
        quarantined_group_count=report['quarantined_group_count'],
        eligible_group_response_count=report.get('eligible_group_response_count'),
        whole_group_filter_executed=report.get('whole_group_filter_executed'))


def choose_dispatch(baseline, candidate, baseline_requests, candidate_requests,
                    baseline_seconds, candidate_seconds, evidence_ok):
    same_requests = (bool(baseline_requests) and
                     Counter(baseline_requests) == Counter(candidate_requests))
    equivalent = (evidence_ok and baseline['passed'] and candidate['passed'] and
                  same_requests and update_projection(baseline) == update_projection(candidate))
    # This is an operational speed criterion, never a semantic pass-rate threshold.
    faster = candidate_seconds < baseline_seconds * .95
    use_candidate = equivalent and faster
    return dict(selected_worker_concurrency=144 if use_candidate else 48,
                selected_fair_dispatch=use_candidate, raw_verdict_reward_update_equal=equivalent,
                identical_independent_request_multiset=same_requests,
                speed_improvement_at_least_five_percent=faster,
                baseline_seconds=baseline_seconds, candidate_seconds=candidate_seconds,
                baseline_over_candidate_speedup=baseline_seconds / candidate_seconds,
                reason='equivalent_and_faster' if use_candidate else 'retain_original_dispatch',
                verdict_cache_enabled=False, optimizer_updates=0)
