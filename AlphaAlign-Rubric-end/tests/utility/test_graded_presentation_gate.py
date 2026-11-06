from copy import deepcopy
import json
from pathlib import Path

from scripts import graded_presentation_gate as gate


def report():
    data = json.loads(gate.smoke.FIXTURE.read_text())
    row = dict(case_count=20, cases=[dict(id=c['id'], passed=True,
        observed_role=c['expected_roles'][0]) for c in data['cases']],
        request_errors=[], evidence_complete=True, correction_deltas=[0.] * 20,
        calibration=[dict(passed=True)] * 3,
        optimization={arm:dict(passed=True) for arm in ['G1','G3']})
    return data, dict(replica_validation={'pooled': row})


def test_missed_correction_stays_a_raw_failure_and_separate_warning():
    data, raw = report()
    row = raw['replica_validation']['pooled']['cases'][3]
    row.update(passed=False, observed_role=0)
    snapshot = deepcopy(raw)
    blocking, warnings = gate.assess_legacy(data, raw)
    assert not blocking and warnings
    assert raw == snapshot and row['passed'] is False


def test_neutral_wrong_answer_must_not_become_a_positive_correction():
    data, raw = report()
    raw['replica_validation']['pooled']['correction_deltas'][2] = .05
    raw['replica_validation']['pooled']['correction_deltas'][0] = -.05
    blocking, _ = gate.assess_legacy(data, raw)
    assert any(r['reason'] == 'false_positive_or_wrong_direction' for r in blocking)


def test_calibration_or_incomplete_evidence_still_blocks():
    data, raw = report()
    raw['replica_validation']['pooled']['calibration'][0]['passed'] = False
    raw['replica_validation']['pooled']['request_errors'] = [dict(error_type='UnknownScore')]
    blocking, _ = gate.assess_legacy(data, raw)
    assert len(blocking) >= 2


def test_cross_replica_same_request_conflict_cannot_be_selected(tmp_path):
    def event(score, response):
        return dict(error_type=None, raw_output=json.dumps(dict(score=score)),
            request_audit=dict(evidence_led=True, system_sha256='a', user_sha256='b',
                guidance_sha256='c', max_tokens=512, temperature=0., request_seed=0,
                served_model='qwen3_32B_instruct', response_id=response))
    (tmp_path / 'replica0.jsonl').write_text(json.dumps(event(1, 'first'))+'\n')
    (tmp_path / 'replica1.jsonl').write_text(json.dumps(event(0, 'second'))+'\n')
    result = gate.cross_request_audit(tmp_path)
    assert result['passed'] is False and len(result['conflicts']) == 1
    assert result['cached_verdicts_returned'] == 0
