"""Finite tuning of grouped transport. No verdict caching or scoring changes."""
from collections import Counter
import os
import math

from verl.utils.judge_dispatch import interleave, update_projection

PROFILES = {'baseline': (48, False), 'fair48': (48, True), 'fair72': (72, True), 'compact72': (72, True)}
WIRE_FORMATS = {name:('tuple' if name=='compact72' else 'object') for name in PROFILES}
ORDER = ('baseline_1','fair48_1','fair72_1','compact72_1','compact72_2','fair72_2','fair48_2','baseline_2')


def configuration():
    workers = int(os.getenv('RUBRIC_GROUPED_API_CONCURRENCY', '48'))
    fair = os.getenv('RUBRIC_GROUPED_FAIR_DISPATCH_ENABLE', '0')
    if workers not in (48, 72) or fair not in ('0', '1'):
        raise ValueError('Unknown grouped dispatch configuration')
    if workers == 72 and fair != '1':
        raise ValueError('Untested grouped dispatch configuration')
    return workers, fair == '1'


def ordered_indices(api, prompts, responses, fair):
    if not fair:
        return list(range(len(prompts)))
    if os.getenv('RUBRIC_JUDGE_LOAD_BALANCE', 'sticky') != 'sticky':
        raise ValueError('Fair grouped dispatch requires unchanged sticky routing')
    return interleave(range(len(prompts)), lambda i: api._base_url_for_request(prompts[i], responses[i], 1))


def choose(snapshots):
    """Choose the fastest admissible measured pair, otherwise original48/FIFO."""
    result = dict(selected_grouped_workers=48, selected_grouped_fair=False,
        selected_profile='baseline', selected_wire_format='object', reason='retain_original_grouped_dispatch', optimizer_updates=0,
        verdict_cache_enabled=False, candidate_checks={}, not_a_training_ETA=True)
    if set(snapshots) != set(ORDER):
        result['reason'] = 'incomplete_probe_retain_original_grouped_dispatch'
        return result
    a, z = snapshots['baseline_1'], snapshots['baseline_2']
    # Every result is a separately executed online reward replay. Same bits,
    # rewards, exclusion masks and BOTH G1/G3 numerical projections must agree.
    def projection(s):
        return dict(reward_and_current_grpo=update_projection(s['report']), both_grpo=s['both_grpo'])
    all_ids = [rid for key in ORDER for rid in snapshots[key]['response_ids']]
    unique = bool(all_ids) and len(all_ids) == len(set(all_ids))
    timing_ok = all(isinstance(s['seconds'], (int,float)) and math.isfinite(s['seconds'])
                    and s['seconds'] > 0 for s in snapshots.values())
    def semantics(value):
        return Counter(value.get('semantic_requests',value['requests']))
    stable = (timing_ok and bool(a['requests']) and unique and a['evidence_complete'] and z['evidence_complete']
        and a['report']['passed'] and z['report']['passed']
        and projection(a) == projection(z) and Counter(a['requests']) == Counter(z['requests'])
        and semantics(a)==semantics(z))
    baseline = (a['seconds'] + z['seconds']) / 2
    result.update(baseline_seconds=baseline, baseline_repeated_projection_equal=stable,
                  all_response_ids_independent=unique)
    valid = []
    for name in ('fair48','fair72','compact72'):
        b, c = snapshots[name + '_1'], snapshots[name + '_2']
        equivalent = (stable and b['evidence_complete'] and c['evidence_complete']
            and b['report']['passed'] and c['report']['passed']
            and projection(a) == projection(b) == projection(c)
            and semantics(a)==semantics(b)==semantics(c)
            and b.get('wire_format','object')==c.get('wire_format','object')==WIRE_FORMATS[name]
            and (WIRE_FORMATS[name]=='object' or bool(b.get('semantic_requests')) and bool(c.get('semantic_requests')))
            and (WIRE_FORMATS[name]=='tuple' or Counter(a['requests'])==Counter(b['requests'])==Counter(c['requests'])))
        seconds = (b['seconds'] + c['seconds']) / 2
        faster = timing_ok and seconds < baseline * .95
        result['candidate_checks'][name] = dict(equivalent=equivalent, mean_seconds=seconds,
            speedup=baseline / seconds if timing_ok else None,
            faster_by_at_least_five_percent=faster,
            identical_raw_http_request_multiset=Counter(a['requests'])==Counter(b['requests'])==Counter(c['requests']),
            intended_wire_format_change=WIRE_FORMATS[name]!='object')
        if equivalent and faster:
            valid.append((seconds, name))
    if valid:
        seconds, name = min(valid)
        workers, fair = PROFILES[name]
        result.update(selected_profile=name, selected_grouped_workers=workers,
            selected_grouped_fair=fair, selected_wire_format=WIRE_FORMATS[name], reason='fastest_equivalent_measured_candidate',
            selected_seconds=seconds, selected_speedup=baseline / seconds)
    return result


# Per-worker observation is isolated from requests and model verdicts. It carries
# no answer text, credentials or scores and is never used by reward arithmetic.
from contextvars import ContextVar
_OBSERVER = ContextVar('grouped_transport_observer', default=None)


def observe(value):
    callback = _OBSERVER.get()
    if callback is not None:
        callback(value)


def execute_one(api, index, prompts, responses, rules, infos, observations):
    token = _OBSERVER.set(observations.append)
    try:
        return api.judge(prompts[index], responses[index], rules[index], infos[index])
    finally:
        _OBSERVER.reset(token)


def publish_timing(api, rows, observations, seconds):
    if os.getenv('RUBRIC_GROUPED_TIMING_ENABLE','0') != '1':
        return
    if os.getenv('RUBRIC_GROUPED_TIMING_SCOPE') == 'preflight_tuning':
        return  # The full optional experiment archives its own request timings.
    if not os.getenv('RUBRIC_RUN_DIR') or rows < 64:
        return
    from pathlib import Path
    import statistics
    import uuid
    from Evaluation.safety.json_evidence import write_json_atomic
    count = getattr(api,'_grouped_timing_batches',0)+1
    api._grouped_timing_batches=count
    step=int(os.getenv('RUBRIC_GLOBAL_STEP','0'))
    if count>5 and step%30 and step%100:
        return
    workers,fair=configuration()
    counts=Counter((x.get('endpoint_port'),x.get('error_type')) for x in observations)
    def times(key):
        values=[x[key] for x in observations if isinstance(x.get(key),(int,float))]
        if any(not math.isfinite(x) or x<0 for x in values):
            raise ValueError('Invalid grouped transport timing')
        return dict(samples=len(values),mean=statistics.mean(values) if values else None,
                    maximum=max(values) if values else None,total=sum(values))
    tokens={}
    for key in ('prompt_tokens','completion_tokens','total_tokens'):
        values=[x.get('token_usage',{}).get(key) for x in observations]
        known=[x for x in values if type(x) is int and x>=0]
        tokens[key]=dict(total=sum(known),known_responses=len(known),unknown_responses=len(values)-len(known))
    path=Path(os.environ['RUBRIC_RUN_DIR'])/'throughput'/f'grouped_{step:08d}_{uuid.uuid4().hex}.json'
    write_json_atomic(path,dict(global_step=step,response_rows=rows,model_call_attempts=len(observations),
        grouped_workers=workers,grouped_fair_dispatch=fair,wire_format=os.getenv('RUBRIC_GROUPED_WIRE_FORMAT','object'),endpoint_max_inflight=api._endpoint_max_inflight,
        batch_wall_seconds=seconds,local_queue=times('local_queue_wait_seconds'),http=times('http_seconds'),
        preprocessing=times('preprocessing_seconds'),tokens=tokens,
        endpoints=[dict(port=port,error_type=error,count=n) for (port,error),n in counts.items()],
        scope='validation' if os.getenv('RUBRIC_EVOLVE')=='0' else 'training_scoring',
        records_raw_request_response_elsewhere=False,optimizer_updates_observed_here=False,
        interpretation='Transport only; actor metrics prove updates. Sum of overlapping HTTP times is not wall time.'))
