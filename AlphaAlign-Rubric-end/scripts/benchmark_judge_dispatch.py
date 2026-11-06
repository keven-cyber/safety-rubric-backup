"""One predeclared paired real-reward replay, then retain48 or select144 workers.

This runs before any optimizer update, using the unchanged 36 frozen answers,
the same real online reward wrapper and GRPO replay as the graded gate. Every
raw request/response is kept. Different outputs, incomplete evidence, or a lack
of measured improvement retains the original dispatch, without rerunning to
pick a favorable result. The full graded/first-actor gates still follow.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Evaluation.safety.json_evidence import write_json_atomic
from scripts.graded_presentation_gate import environment, reward_replay, load_cases, smoke, rj, cq
from verl.utils.judge_dispatch import choose_dispatch


def requests(directory):
    signatures, response_ids, waits, http, tokens = [], [], [], [], []
    for path in sorted(Path(directory).rglob('events_*.jsonl')):
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get('event') not in ('base_judge_attempt', 'targeted_judge_attempt'):
                continue
            audit = event.get('request_audit') or {}
            if not audit.get('response_id') or not event.get('raw_output'):
                raise ValueError('Incomplete real canary response evidence')
            if not cq.verified_audit_event(event):
                raise ValueError('Canary request/raw response could not be independently reverified')
            keys = ('system_sha256', 'user_sha256', 'guidance_sha256',
                    'temperature', 'max_tokens', 'request_seed')
            if any(audit.get(key) is None for key in keys):
                raise ValueError('Incomplete real canary request identity')
            signatures.append(json.dumps(dict(model=event['model'],
                **{key: audit[key] for key in keys}), sort_keys=True))
            response_ids.append(audit['response_id'])
            waits.append(audit.get('local_queue_wait_seconds'))
            http.append(audit.get('http_seconds'))
            tokens.append(audit.get('token_usage'))
    if len(response_ids) != len(set(response_ids)):
        raise ValueError('Repeated canary responseID; not independent request proof')
    return signatures, dict(request_count=len(signatures), independent_response_ids=response_ids,
                           local_queue_wait_seconds=waits, http_seconds=http, token_usage=tokens)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    result = dict(selected_worker_concurrency=48, selected_fair_dispatch=False,
                  optimizer_updates=0, reason='retain_original_dispatch', real_judge_executed=False)
    backend = rj._get_judge_backend()
    if not isinstance(backend, rj.ApiJudge) or len(backend.base_urls) != 3:
        raise ValueError('Throughput replay requires the actual three training Judge replicas')
    cases = [dict(c, replay_cohort='historical_sixteen') for c in load_cases()]
    frozen = smoke.FIXTURE.read_bytes()
    if hashlib.sha256(frozen).hexdigest() != smoke.FIXTURE_SHA256:
        raise ValueError('Original twenty-case fixture changed')
    data = json.loads(frozen); smoke.validate_fixture(data)
    cases += [dict(c, replay_cohort='original_twenty') for c in data['cases']]
    try:
        snapshots = []
        # Fixed ABBA counters simple startup/thermal/order bias. All four runs
        # are retained; never rerun or pick the fastest/easiest semantic result.
        for name, workers, fair in [('baseline_1', 48, 0), ('candidate_1', 144, 1),
                                     ('candidate_2', 144, 1), ('baseline_2', 48, 0)]:
            directory = args.output / name; directory.mkdir()
            independent = rj.ApiJudge(','.join(backend.base_urls), backend.api_key, backend.model)
            rj._JUDGE_INSTANCE = independent
            rj._JUDGE_BACKEND = os.getenv('RUBRIC_JUDGE_BACKEND', 'local_api').lower().strip()
            with environment(RUBRIC_JUDGE_API_CONCURRENCY=workers, RUBRIC_JUDGE_FAIR_DISPATCH_ENABLE=fair,
                RUBRIC_BASE_JUDGE_TRACE_DIR=directory / 'reward_attempts',
                RUBRIC_TARGETED_JUDGE_TRACE_DIR=directory / 'correction_attempts'):
                started = time.monotonic()
                report = reward_replay(cases, independent, directory)
                elapsed = time.monotonic() - started
            signatures, audit = requests(directory)
            consistency = cq.audit_directory(directory)
            complete = consistency['passed'] or cq.containment_verified(consistency, report['rows'])
            write_json_atomic(directory / 'dispatch_timing.json', dict(audit, seconds=elapsed,
                workers=workers, fair_dispatch=bool(fair), consistency=consistency, evidence_complete=complete))
            snapshots.append((report, signatures, elapsed, complete, set(audit['independent_response_ids'])))
        a, b, c, d = snapshots
        from verl.utils.judge_dispatch import update_projection
        stable = (update_projection(a[0]) == update_projection(d[0]) and
                  update_projection(b[0]) == update_projection(c[0]) and
                  Counter(a[1]) == Counter(d[1]) and Counter(b[1]) == Counter(c[1]))
        distinct = sum(len(s[4]) for s in snapshots) == len(set.union(*(s[4] for s in snapshots)))
        result.update(choose_dispatch(a[0], b[0], a[1], b[1],
            (a[2] + d[2]) / 2, (b[2] + c[2]) / 2,
            stable and distinct and all(s[0]['passed'] and s[3] for s in snapshots)))
        result.update(real_judge_executed=True, case_count=len(cases),
                      predeclared_order=['baseline_1','candidate_1','candidate_2','baseline_2'],
                      repeated_raw_verdict_reward_update_equal=stable,
                      timing_is_one_startup_probe_not_training_ETA=True)
    except Exception as error:
        # This optional experiment never substitutes for the mandatory complete
        # graded/first-actor checks that follow with the selected dispatch.
        result.update(error_type=type(error).__name__, error=str(error),
                      reason='incomplete_probe_retain_original_dispatch')
    write_json_atomic(args.output / 'selection.json', result)
    print(json.dumps({key: result[key] for key in ('selected_worker_concurrency',
        'selected_fair_dispatch', 'reason')}), flush=True)


if __name__ == '__main__':
    main()
