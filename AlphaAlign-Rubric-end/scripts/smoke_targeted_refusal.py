"""Fail-closed real training-Judge gate; no policy optimization is performed."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.request
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verl.utils import targeted_refusal as tr
from verl.utils import validation_selection as vs
from verl.utils.training_evidence import atomic_json

FIXTURE = ROOT / 'tests/fixtures/targeted_refusal_existing_answers.json'
FIXTURE_SHA256 = 'b3f35d0eba94455ae4c3c4646d7e2f8e9749cdde4f9358fc3b451f3fff2b6dc8'


def validate_fixture(data):
    ids = [r['id'] for r in data['cases']]
    if (data.get('require_all_cases') is not True or data.get('allow_case_removal') is not False
            or len(ids) != 20 or len(set(ids)) != len(ids)
            or not {f'legacy_{i:02d}' for i in range(10)}.issubset(ids)
            or sum(r['origin'] == 'historical_training_replay' for r in data['cases']) != 10):
        raise ValueError('Frozen development cases are missing or altered')


class CaseRecordingJudge:
    """Validation only: preserve every case on exhaustion, never invent a verdict."""
    def __init__(self, judge, cases, directory):
        self.judge, self.cases = judge, cases
        self.directory = Path(directory)
        self.model = judge.model
        self.phase = 'diagnostic'
        self.records = []
        self.errors = []

    def judge_batch(self, prompts, responses, rules_list, extra_infos=None):
        infos = extra_infos if extra_infos is not None else [{} for _ in prompts]
        if not len(prompts) == len(responses) == len(rules_list) == len(infos):
            raise ValueError('Incomplete case batch')
        outputs = []
        for p, response, rules, info in zip(prompts, responses, rules_list, infos):
            ids = [c['id'] for c in self.cases if c['prompt'] == p and c['response'] == response]
            request_id = len(self.records)
            record = dict(request_id=request_id, case_ids=ids, phase=self.phase,
                prompt_sha256=hashlib.sha256(p.encode()).hexdigest(),
                response_sha256=hashlib.sha256(response.encode()).hexdigest(),
                rules=rules.all_rules, request_completed=False)
            metadata = dict(info or {}, validation_case_ids=ids, validation_phase=self.phase,
                            validation_request_id=request_id)
            try:
                batch = self.judge.judge_batch([p], [response], [rules], [metadata])
                if len(batch) != 1 or not vs._binary_output(batch[0], rules):
                    raise ValueError('Invalid or incomplete real verdict')
                out = batch[0]
                record.update(request_completed=True, output=out)
            except Exception as error:
                # None is an unknown verdict, not a failed rule or a replacement reward.
                out = {key: [None] * len(items) for key, items in rules.all_rules.items()}
                out.update(_parse_valid=False, _judge_failed=True)
                record.update(output=out, error_type=type(error).__name__)
                self.errors.append(dict(request_id=request_id, case_ids=ids,
                                        phase=self.phase, error_type=type(error).__name__))
            atomic_json(self.directory / f'request_{request_id:04d}.json', record)
            self.records.append(record)
            outputs.append(out)
        return outputs


def run_gate(data, judge, rules_type):
    validate_fixture(data)
    cases = data['cases']
    prompts = [c['prompt'] for c in cases]
    responses = [c['response'] for c in cases]
    profiles = [c['profile'] for c in cases]
    infos = [c['original_info'] for c in cases]
    _, diagnostic = vs.evaluate_validation(prompts, responses, profiles, judge, rules_type, True)
    rules = [tr._final_rules(info, rules_type) for info in infos]
    if isinstance(judge, CaseRecordingJudge):
        judge.phase = 'original_rule_recheck'
    original = judge.judge_batch(prompts, responses, rules, tr.reference_infos(infos))
    if len(original) != len(cases) or len(diagnostic['records']) != len(cases):
        raise ValueError('Incomplete real Judge validation; no case may be removed')
    observations = []
    for c, out, rule, diag in zip(cases, original, rules, diagnostic['records']):
        role, reason = tr.classify(out, rule, diag)
        actual = [diag.get(k) for k in ('substantive_answer', 'coherent_answer', 'unnecessary_refusal')]
        diagnostic_passed = diag['parse_valid']
        if 'expected_diagnosis' in c:
            diagnostic_passed = diagnostic_passed and actual == c['expected_diagnosis']
        if 'expected_policy_refusal' in c:
            diagnostic_passed = diagnostic_passed and actual[2] == c['expected_policy_refusal']
        passed = bool(diagnostic_passed and vs._binary_output(out, rule) and role in c['expected_roles'])
        observations.append(dict(id=c['id'], expected_roles=c['expected_roles'], observed_role=role,
            expected_diagnosis=c.get('expected_diagnosis'), observed_diagnosis=actual,
            expected_policy_refusal=c.get('expected_policy_refusal'), original_rule_passes=out,
            reason=reason, passed=passed))
    original_scores = [dict(score=.5) for _ in cases]
    if isinstance(judge, CaseRecordingJudge):
        judge.phase = 'correction_integration'
    audit = tr.annotate(prompts, responses, profiles, infos, original_scores, judge, rules_type, 360)
    delta = [r[tr.DELTA] for r in original_scores]
    expected_changed = {0: 1, 1: -1, 3: 1, 4: -1}
    contrast_passed = (audit['reason'] == 'applied' and audit['calibration_pass'] is True
        and all((d > 0 if expected_changed[i] == 1 else d < 0) if i in expected_changed
                else d == 0 for i, d in enumerate(delta))
        and all(r['score'] == .5 for r in original_scores)
        and abs(sum(delta)) < 1e-9 and max(map(abs, delta)) <= tr._cap() + 1e-12)
    # Actual historical GRPO function, not a surrogate reward-normalization test.
    import ast
    from collections import defaultdict
    import numpy as np
    import torch
    env = dict(torch=torch, np=np, defaultdict=defaultdict)
    path = ROOT / 'verl/trainer/ppo/core_algos.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'compute_grpo_outcome_advantage')
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), env)
    actual_grpo = env['compute_grpo_outcome_advantage']
    optimization = {}
    for arm, floor in [('G1', 0.), ('G3', .25)]:
        scores = torch.ones(8, 2) * .25
        mask = torch.tensor([[1., 0.]] * 8)
        base, _ = actual_grpo(scores, mask, np.array(['same'] * 8), std_floor=floor)
        adjusted = tr.apply_advantage_correction(base, mask, [.05] * 4 + [-.05] * 4)
        unchanged = tr.apply_advantage_correction(base, mask, [0.] * 8)
        optimization[arm] = dict(passed=bool(torch.equal(base, unchanged)
            and (adjusted - base).abs().max() <= .050001
            and torch.equal(adjusted[:, 1], base[:, 1])), std_floor=floor,
            maximum_actual_advantage_delta=float((adjusted - base).abs().max()))
    passed = bool(all(c['passed'] for c in diagnostic['calibration'])
                  and all(r['passed'] for r in observations) and contrast_passed
                  and all(r['passed'] for r in optimization.values()))
    evidence = {}
    if isinstance(judge, CaseRecordingJudge):
        evidence = dict(request_errors=judge.errors, evidence_count=len(judge.records),
                        evidence_complete=True, evidence_directory='case_evidence',
                        raw_attempt_directory='judge_attempts')
        passed = passed and not judge.errors
    return dict(passed=passed, cases=observations, case_count=len(observations),
                failed_case_ids=[r['id'] for r in observations if not r['passed']],
                calibration=diagnostic['calibration'], correction_passed=contrast_passed,
                correction_deltas=delta, optimization=optimization,
                note='Development gate, not a claim of benchmark superiority.', **evidence)


def run_base_protocol_gate(data, judge, directory):
    """Exercise the unflagged base-reward all_rules route as one 8x8 batch.

    This tests the output contract, not new semantic labels. The original
    twenty-case semantic gate is unchanged and remains mandatory separately.
    """
    from verl.utils.reward_score.rubric_judge import _build_rules
    ranked = sorted(data['cases'], key=lambda c: sum(len(v) for v in _build_rules(c['original_info']).all_rules.values()))
    cases = ranked[:4] + ranked[-4:]
    prompts, responses, rules, infos = [], [], [], []
    for c in cases:
        for rollout in range(8):
            info = dict(c['original_info'], prompt=c['prompt'], uid=c['id'], rollout_idx=rollout + 1,
                        rollout_n=8, validation_case_ids=[c['id']], validation_phase='base_protocol_batch')
            # No targeted flags: the same base-reward path that failed in training.
            info.pop('targeted_refusal_reference_only', None)
            info.pop('targeted_refusal_diagnostic_only', None)
            prompts.append(c['prompt']); responses.append(c['response'])
            rules.append(_build_rules(info)); infos.append(info)
    atomic_json(directory / 'inputs.json', dict(prompts=prompts, responses=responses,
        rules=[r.all_rules for r in rules], sample_ids=[i['uid'] for i in infos]))
    previous_trace = os.environ.get('RUBRIC_BASE_JUDGE_TRACE_DIR')
    os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = str(directory / 'attempts')
    report = dict(passed=False, response_count=64, prompt_count=8, batch_returned=False,
                  contract_only=True, semantic_cases_removed=0)
    try:
        outputs = judge.judge_batch(prompts, responses, rules, infos)
        valid = len(outputs) == 64 and all(vs._binary_output(o, r) for o, r in zip(outputs, rules))
        report.update(passed=valid, batch_returned=True, outputs=outputs)
    except Exception as error:
        report['error_type'] = type(error).__name__
    finally:
        if previous_trace is None:
            os.environ.pop('RUBRIC_BASE_JUDGE_TRACE_DIR', None)
        else:
            os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = previous_trace
    atomic_json(directory / 'result.json', report)
    return report


def run_deployment_gate(data, backend, rules_type, output_dir, judge_factory):
    """All replicas AND the actual pooled route must pass; never select a winner."""
    endpoints = backend.base_urls
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('Duplicate Judge endpoints cannot stand in for independent replicas')
    expected = int(os.getenv('JUDGE_REPLICA_COUNT', str(len(endpoints))))
    if len(endpoints) != expected:
        raise ValueError('Gate must cover every configured training Judge replica')
    reports = {}
    routes = [(f'replica_{i}', judge_factory(endpoint)) for i, endpoint in enumerate(endpoints)]
    if len(endpoints) == 1:
        routes = []  # The pooled single endpoint already is the complete deployment.
    routes.append(('pooled', backend))
    previous_trace = os.environ.get('RUBRIC_TARGETED_JUDGE_TRACE_DIR')
    try:
        for name, route in routes:
            directory = output_dir if name == 'pooled' else output_dir / name
            os.environ['RUBRIC_TARGETED_JUDGE_TRACE_DIR'] = str(directory / 'judge_attempts')
            wrapped = CaseRecordingJudge(route, data['cases'], directory / 'case_evidence')
            report = run_gate(data, wrapped, rules_type)
            if os.getenv('RUBRIC_BASE_PROTOCOL_GATE', '0') == '1':
                report['base_protocol'] = run_base_protocol_gate(data, route, directory / 'base_protocol')
                report['passed'] = report['passed'] and report['base_protocol']['passed']
            report.update(route=name, endpoint_count=len(route.base_urls))
            reports[name] = report
            atomic_json(directory / 'gate.json', report)
    finally:
        if previous_trace is None:
            os.environ.pop('RUBRIC_TARGETED_JUDGE_TRACE_DIR', None)
        else:
            os.environ['RUBRIC_TARGETED_JUDGE_TRACE_DIR'] = previous_trace
    result = dict(reports['pooled'])
    result.update(replica_validation=reports, deployment_passed=all(r['passed'] for r in reports.values()),
                  replica_count=len(endpoints),
                  failed_case_ids=sorted({case_id for r in reports.values() for case_id in r['failed_case_ids']}),
                  failed_routes=[name for name, report in reports.items() if not report['passed']])
    result['passed'] = result['deployment_passed']
    return result


def deployment_manifest(endpoints):
    """Record artifacts and actual service settings, without API credentials."""
    from importlib.metadata import version
    model_dir = Path(os.environ.get('JUDGE_MODEL_PATH', '/root/models/judge'))
    artifacts = {}
    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'generation_config.json'):
        path = model_dir / name
        if path.is_file():
            artifacts[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    versions = {}
    for package in ('vllm', 'torch', 'transformers', 'openai'):
        versions[package] = version(package)
    return dict(package_versions=versions, model_config_sha256=artifacts,
        weight_files=[dict(name=p.name, bytes=p.stat().st_size) for p in sorted(model_dir.glob('*.safetensors'))],
        weight_bytes_verified=False,
        settings={k: os.environ.get(k) for k in (
            'JUDGE_REPLICA_COUNT', 'JUDGE_GPU_IDS', 'JUDGE_TP_SIZE', 'JUDGE_MAX_MODEL_LEN',
            'JUDGE_GPU_MEMORY_UTILIZATION', 'JUDGE_ENABLE_PREFIX_CACHING',
            'RUBRIC_JUDGE_API_CONCURRENCY', 'RUBRIC_JUDGE_LOAD_BALANCE',
            'RUBRIC_JUDGE_GRANULARITY', 'RUBRIC_JUDGE_MAX_NEW_TOKENS')},
        endpoint_count=len(endpoints), targeted_effective_granularity='single_rule')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = dict(passed=False, status='blocked', real_judge_requested=True, real_judge_executed=False,
                  method_version=tr.VERSION, training_started=False)
    judge = None
    try:
        data = json.loads(FIXTURE.read_text())
        validate_fixture(data)
        result.update(case_count=len(data['cases']), evidence_complete=False,
                      cases=[dict(id=c['id'], passed=False, status='not_evaluated')
                             for c in data['cases']],
                      failed_case_ids=[c['id'] for c in data['cases']])
        result['fixture_sha256'] = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        if result['fixture_sha256'] != FIXTURE_SHA256:
            raise ValueError('Frozen validation fixture changed; no case or expectation may be removed')
        if os.getenv('RUBRIC_JUDGE_BACKEND', 'local_api') != 'local_api':
            raise ValueError('Use the real local_api training Judge, not the rubric writer API')
        if os.getenv('RUBRIC_JUDGE_LOCAL_API_MODEL', 'qwen3_32B_instruct') != 'qwen3_32B_instruct':
            raise ValueError('Training Judge model must remain qwen3_32B_instruct')
        # Probe the actual training endpoints before expensive Judge work.
        bases = os.getenv('RUBRIC_JUDGE_LOCAL_API_BASES', '') or os.getenv(
            'RUBRIC_JUDGE_LOCAL_API_BASE', 'http://127.0.0.1:8000/v1')
        for endpoint in bases.split(','):
            endpoint = endpoint.strip().rstrip('/')
            if urlparse(endpoint).hostname not in ('127.0.0.1', 'localhost', '::1'):
                raise ValueError('This gate requires the same-node training Judge')
            with urllib.request.urlopen(endpoint + '/models', timeout=5) as response:
                models = json.load(response)
            if 'qwen3_32B_instruct' not in [m.get('id') for m in models.get('data', [])]:
                raise ValueError('Expected training Judge is not served')
        from verl.utils.reward_score.rubric_judge import RubricRules, ApiJudge, _get_judge_backend
        os.environ['RUBRIC_TARGETED_REFUSAL_ENABLE'] = '1'
        os.environ['RUBRIC_BENIGN_CONTRAST_ENABLE'] = '0'
        os.environ['RUBRIC_BASE_PROTOCOL_GATE'] = '1'
        judge = _get_judge_backend()
        result['deployment'] = deployment_manifest(judge.base_urls)
        result['real_judge_executed'] = True
        result.update(run_deployment_gate(data, judge, RubricRules, args.output.parent,
            lambda endpoint: ApiJudge(endpoint, judge.api_key, judge.model)), real_judge_model=judge.model)
        result['status'] = 'passed' if result['passed'] else 'failed'
    except Exception as error:
        # Never serialize credentials, request headers, or a sensitive URL.
        result['error_type'] = type(error).__name__
        if result['real_judge_executed']:
            result['status'] = 'failed'
        result['error'] = 'Training Judge or validation prerequisites unavailable; full training is blocked.'
        if isinstance(judge, CaseRecordingJudge):
            result.update(request_errors=judge.errors, evidence_count=len(judge.records),
                          evidence_complete=False, evidence_directory='case_evidence',
                          raw_attempt_directory='judge_attempts')
    result['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in ('verl/utils/targeted_refusal.py', 'verl/utils/validation_selection.py',
                     'verl/utils/reward_score/rubric_judge.py',
                     'scripts/smoke_targeted_refusal.py', 'verl/trainer/ppo/core_algos.py',
                     'verl/trainer/ppo/ray_trainer.py',
                     'verl/utils/reward_score/rubric_judge_save_evolve_rubric.py', 'train_val.sh')}
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
