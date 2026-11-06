"""Transport backpressure and native grammar probes; no simulated semantic pass."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from verl.utils.reward_score import rubric_judge as rj


@pytest.mark.parametrize('limit', [0, 1, 48])
def test_worker_pool_48_waits_locally_without_sharing_or_editing_responses(monkeypatch, limit):
    monkeypatch.setenv('RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT', str(limit))
    monkeypatch.setenv('RUBRIC_JUDGE_API_CONCURRENCY', '48')
    endpoints = ['http://one/v1', 'http://two/v1']
    api = rj.ApiJudge(','.join(endpoints), 'unused', 'unchanged-model')
    lock = threading.Lock()
    active = dict.fromkeys(endpoints, 0)
    peak = dict.fromkeys(endpoints, 0)
    seen = []
    def call(index):
        endpoint = endpoints[index % 2]
        def create(**request):
            with lock:
                active[endpoint] += 1
                peak[endpoint] = max(peak[endpoint], active[endpoint])
                seen.append(request['sample_id'])
            time.sleep(.01)
            with lock:
                active[endpoint] -= 1
            return dict(sample_id=request['sample_id'], raw_score=index % 2)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        audit = {}
        response = api._create_admitted_completion(endpoint, client, audit, sample_id=index)
        return response, audit
    with ThreadPoolExecutor(max_workers=48) as pool:
        results = list(pool.map(call, range(48)))
    assert sorted(seen) == list(range(48))
    assert [r['sample_id'] for r, _ in results] == list(range(48))
    assert [r['raw_score'] for r, _ in results] == [i % 2 for i in range(48)]
    assert all(a['endpoint_max_inflight'] == limit for _, a in results)
    assert all(a['local_queue_wait_seconds'] >= 0 for _, a in results)
    if limit == 1:
        assert all(n == 1 for n in peak.values())
    else:
        assert all(n > 1 for n in peak.values())
    assert rj.os.environ['RUBRIC_JUDGE_API_CONCURRENCY'] == '48'


def test_http_exception_releases_endpoint_slot_without_converting_failure(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT', '1')
    api = rj.ApiJudge('http://one/v1', 'unused', 'unchanged-model')
    def fail(**request):
        raise TimeoutError('actual transport error')
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail)))
    with pytest.raises(TimeoutError, match='actual transport error'):
        api._create_admitted_completion(api.base_url, client, {})
    # Inspect/acquire only the test's semaphore to prove the exception released it.
    slot = api._endpoint_slots[api.base_url]
    assert slot.acquire(blocking=False)
    slot.release()


def test_native_grammar_blocks_whitespace_stall_but_preserves_both_binary_scores(tmp_path):
    from scripts.validate_existing_answer_judge import grammar_replay, load_cases
    report = grammar_replay(load_cases(), tmp_path)
    assert report['passed'] and report['planned_case_count'] == 36
    assert report['real_judge_executed'] is False
    assert len(report['schemas']) == 36
    assert all(p['passed'] for p in report['truth_score_contract_probe'])
    assert all(p['passed'] for p in report['counterexample_contract_probe'])
    assert all(p['passed'] for p in report['bounded_whitespace_contract_probe'])
    assert rj._grounded_guided_body('unchanged')['guided_json']['x-guidance'] == {'whitespace_flexible': False}


def test_runtime_validator_still_accepts_whitespace_and_rejects_invalid_score():
    rule = rj.RubricRules([], ['One unchanged proposition.'], [], [])
    obj = dict(evidence_line_ids=[1], reason='first line\nsecond line', predicate_truth='TRUE', score=1)
    # Only the generator's optional whitespace is restricted; accepted verdicts
    # still obey the original semantic/shape validation, not a string rewrite.
    spaced = json.dumps(obj, indent=8)
    assert rj._grounded_output(json.loads(spaced), rule, 'original')['outcome_useful_rule'] == [1]
    obj['score'] = 0
    with pytest.raises(ValueError):
        rj._grounded_output(obj, rule, 'original')
