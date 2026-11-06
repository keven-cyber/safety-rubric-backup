"""Real three-replica checks plus separately reported, bounded graded admission.

Raw semantic failures are never relabeled. Unprotected false positive substantive
predicates, invalid outputs, changed fixtures, uncontrolled contradictory requests
and unbounded updates block admission. Proven literal contradictions and fully
archived independent request conflicts retain raw scores but quarantine the entire
prompt group before an update. Raw consistency remains false when conflicts exist.
Missed corrections and isolated presentation errors are warnings.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from Evaluation.safety.json_evidence import write_json_atomic as atomic_json
from scripts import smoke_targeted_refusal as smoke
from scripts.validate_existing_answer_judge import load_cases, grammar_replay, template_replay
from scripts.judge_replay_server import endpoint_limit, observed_profile, selected_profile
from scripts.replay_presentation_rewards import actual_trainer
from verl.utils import presentation_reward as pr
from verl.utils import literal_constraint_guard as lc
from verl.utils import consistency_quarantine as cq
from verl.utils.judge_filter import valid_group_indices
from verl.utils.reward_score import rubric_judge as rj


def digest(value):
    return hashlib.sha256(value).hexdigest()


@contextmanager
def environment(**updates):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update({key: str(value) for key, value in updates.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assess_legacy(data, report, quarantined_case_ids=()):
    blocking, warnings = [], []
    cases = data['cases']
    for route, row in report['replica_validation'].items():
        if (row['case_count'] != 20 or len(row['cases']) != 20 or row.get('request_errors')
                or row.get('evidence_complete') is not True):
            blocking.append(dict(route=route, reason='incomplete_or_invalid_real_verdicts'))
        if not all(c['passed'] for c in row['calibration']):
            blocking.append(dict(route=route, reason='calibration_failed'))
        if not all(r['passed'] for r in row['optimization'].values()):
            blocking.append(dict(route=route, reason='historical_optimization_contract_failed'))
        deltas = row['correction_deltas']
        if (len(deltas) != 20 or not all(np.isfinite(deltas)) or
                max(map(abs, deltas)) > .0500001 or abs(sum(deltas)) > 1e-8):
            blocking.append(dict(route=route, reason='invalid_correction_deltas'))
            continue
        for fixture, observed, delta in zip(cases, row['cases'], deltas):
            reference = observed.get('original_rule_passes', {})
            if reference.get(lc.KEY):
                written = {k: fixture['original_info'].get(k, []) for k in (
                    'outcome_safety_rule', 'outcome_useful_rule', 'reasoning_process_rule', 'refusal_rule')}
                if not lc.verified(reference, written) or delta != 0:
                    blocking.append(dict(route=route, id=fixture['id'], reason='invalid_constraint_quarantine'))
                warnings.append(dict(route=route, id=fixture['id'], delta=delta,
                    reason='raw_constraint_conflict_retained_and_quarantined', proofs=reference[lc.KEY]))
            roles = fixture['expected_roles']
            if ((delta > 0 and 1 not in roles) or (delta < 0 and -1 not in roles)
                    or (observed['observed_role'] == 1 and 1 not in roles)):
                item = dict(route=route, id=fixture['id'], reason='false_positive_or_wrong_direction', delta=delta)
                # Raw role/delta remain in their report. Actual update exclusion
                # must already have been verified for the entire replay group.
                (warnings if fixture['id'] in quarantined_case_ids else blocking).append(item)
            if not observed['passed']:
                warnings.append(dict(route=route, id=fixture['id'], delta=delta,
                    reason='raw_semantic_failure_retained'))
    return blocking, warnings


def cross_request_audit(directory):
    return cq.audit_directory(directory)


def reward_replay(cases, backend, output):
    manifest = json.loads((ROOT / 'configs/presentation_v8_replay_profiles.json').read_text())
    if digest((ROOT / 'tests/fixtures/answer_grounding_existing_answers_20261028.json').read_bytes()) != manifest['fixture_sha256']:
        raise ValueError('Replay routing manifest does not match the frozen fixture')
    infos, profiles, rules = [], [], []
    for c in cases:
        if c.get('replay_cohort') == 'original_twenty':
            saved = dict(profile=c['profile'])
        else:
            saved = manifest['profiles'][c['id']]
            if saved['prompt_sha256'] != digest(c['prompt'].encode()) or saved['response_sha256'] != digest(c['response'].encode()):
                raise ValueError('An existing answer changed')
        profiles.append(saved['profile'])
        infos.append(dict(c['original_info'], prompt=c['prompt'], reward_profile=saved['profile'],
            validation_case_ids=[c['id']], validation_phase='actual_online_reward_replay'))
        rules.append(rj._build_rules(infos[-1]))
    snapshots = []
    atomic_json(output / 'existing_answer_reward_inputs.json', cases)
    # Both calls traverse the actual online evolution wrapper and same live API.
    # No cached verdicts or expected labels enter scoring. Step 1 is the original warmup.
    for enable in [0, 1]:
        with environment(RUBRIC_PRESENTATION_DAMPING_ENABLE=enable, RUBRIC_GLOBAL_STEP=1,
                         RUBRIC_EPOCH=1):
            snapshots.append(rj.compute_score_batch(profiles, [c['response'] for c in cases],
                                                   [None] * len(cases), infos))
    baseline, damped = snapshots
    consistency = cross_request_audit(output)
    conflicted_prompts = cq.tainted_prompts(consistency)
    if any(r.get('rubric_evolve_global_step') != 1 or not r.get('rubric_evolve_scope')
           for snapshot in snapshots for r in snapshot):
        raise ValueError('Actual online evolution wrapper was not executed')
    blocking, warnings, observations, raw_blocking = [], [], [], []
    for c, rule, a, b in zip(cases, rules, baseline, damped):
        verdict = b['rubric_rule_passes']
        conflict_protected = (cq.verified(a['rubric_rule_passes'], rule.all_rules)
            or cq.verified(verdict, rule.all_rules)
            or digest(c['prompt'].encode()) in conflicted_prompts)
        raw_equal = all(a['rubric_rule_passes'].get(k) == verdict.get(k) for k in rule.all_rules)
        if (not rj._targeted_output({k: verdict[k] for k in rule.all_rules}, rule)
                or not verdict.get('_parse_valid') or not a['rubric_rule_passes'].get('_parse_valid')
                or (not raw_equal and not conflict_protected)):
            raise ValueError('Real reward replay is invalid or contradictory')
        protected = lc.verified(verdict, rule.all_rules)
        if (a.get('judge_failed') or b.get('judge_failed')) and not (protected or conflict_protected):
            blocking.append(dict(id=c['id'], reason='unrecovered_invalid_judge_score'))
        if protected and not b.get('judge_failed'):
            blocking.append(dict(id=c['id'], reason='constraint_proof_not_quarantined'))
        comparisons = []
        for category, expected in c.get('expected_rule_scores', {}).items():
            for index, value in expected.items():
                index = int(index)
                actual = verdict[category][index]
                item = dict(id=c['id'], category=category, index=index, expected=value, observed=actual)
                if actual != value:
                    presentation = category == 'outcome_useful_rule' and pr.is_presentation_only(rule.all_rules[category][index])
                    if not presentation and actual == 1:
                        raw_blocking.append(item)
                    (warnings if presentation or actual == 0 or protected or conflict_protected else blocking).append(item)
                comparisons.append(dict(item, passed=actual == value))
        if not all(np.isfinite(r['score']) for r in (a, b)):
            blocking.append(dict(id=c['id'], reason='nonfinite_reward'))
        if not conflict_protected and b['score'] > a['score'] + 1e-9:
            blocking.append(dict(id=c['id'], reason='downward_only_reward_bound_failed'))
        if conflict_protected:
            warnings.append(dict(id=c['id'], reason='independent_score_conflict_group_quarantined',
                                 raw_arrays_equal=raw_equal))
        observations.append(dict(id=c['id'], profile=b['safety_pred_pair'],
            baseline_reward=a, damped_reward=b, comparisons=comparisons,
            prompt_sha256=digest(c['prompt'].encode()), raw_arrays_equal=raw_equal,
            verified_consistency_quarantine=conflict_protected,
            verified_constraint_quarantine=protected, raw_constraint_conflicts=verdict.get(lc.KEY, []),
            replay_cohort=c.get('replay_cohort', 'historical_sixteen')))
    keys = [digest(json.dumps([c['prompt'], p, r.all_rules], sort_keys=True).encode())
            for c, p, r in zip(cases, profiles, rules)]
    counts = {k: keys.count(k) for k in keys}
    keep, quarantined_groups = valid_group_indices(keys, [
        bool(a.get('judge_failed') or b.get('judge_failed') or row['verified_consistency_quarantine'])
        for a, b, row in zip(baseline, damped, observations)])
    kept = set(keep)
    eligible = [i for i,k in enumerate(keys) if i in kept and counts[k] > 1]
    if not eligible:
        raise ValueError('No valid existing-answer groups for actual GRPO replay')
    for i, row in enumerate(observations):
        row.update(prompt_group_sha256=keys[i], quarantined=i not in kept,
                   effective_update_weight=0 if i not in kept else 1,
                   update_advantage=0. if i not in kept else None)
    tensor = torch.tensor([[damped[i]['score']] for i in eligible], dtype=torch.float32)
    metadata = {key: np.array([damped[i][key] for i in eligible]) for key in (pr.COMPONENT, pr.SCORED, pr.CONTENT)}
    metadata.update(uid=np.array([keys[i] for i in eligible]),
                    targeted_refusal_adv_delta=np.zeros(len(eligible)))
    batch = SimpleNamespace(batch=dict(token_level_scores=tensor.clone(),
        token_level_rewards=tensor, response_mask=torch.ones_like(tensor)), non_tensor_batch=metadata)
    actual_trainer()(batch, 'GRPO')
    effects = batch.non_tensor_batch['presentation_advantage_contribution']
    if not torch.isfinite(batch.batch['advantages']).all() or max(np.abs(effects), default=0.) > .050001:
        blocking.append(dict(reason='invalid_actual_grpo_or_unbounded_presentation'))
    for local, original in enumerate(eligible):
        observations[original]['update_advantage'] = float(batch.batch['advantages'][local, 0])
    result = dict(passed=not blocking, real_judge_executed=True, online_evolution_wrapper_executed=True,
        actual_advantage_dispatch_executed=True, optimizer_updates=0, case_count=len(cases),
        blocking=blocking, raw_substantive_mismatches=raw_blocking, warnings=warnings, rows=observations,
        quarantined_group_count=quarantined_groups,
        whole_group_filter_executed=True, singleton_advantages_not_fabricated=True,
        raw_array_disagreements_retained=True,
        eligible_group_response_count=len(eligible),
        advantages=batch.batch['advantages'].tolist(), presentation_contribution=effects.tolist(),
        note='Existing-answer groups, not an actor rollout. Real 8x8 training batch must pass before updates.')
    atomic_json(output / 'existing_answer_reward_advantage.json', result)
    return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    result = dict(legacy_compact_scoring_enabled=os.getenv('RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE','0')=='1',
                  admitted=False, status='blocked', blocking=[], warnings=[], optimizer_updates=0,
                  grouped_scoring_enabled=os.getenv('RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE','0')=='1',
                  raw_semantic_reports_unchanged=True, admission_policy='bounded_impact_not_all_cases_pass')
    try:
        if not lc.enabled():
            raise ValueError('Verified literal-constraint quarantine must be enabled')
        from verl.utils.validation_recovery import enabled, check_runtime_contract
        if selected_profile()['name'] in ('batched_v8','legacy_compact_v8','grouped_context_v8') and not enabled():
            raise ValueError('Fast training requires validation failure containment in the live process')
        if enabled():
            contract = check_runtime_contract(args.output / 'validation_contract')
            result['validation_containment_contract_passed'] = contract['passed']
        cases = load_cases()
        grammar = grammar_replay(cases, args.output / 'grammar')
        if not grammar['passed']:
            raise ValueError('Actual serving backend grammar failed')
        backend = rj._get_judge_backend()
        if not isinstance(backend, rj.ApiJudge) or backend.model != 'qwen3_32B_instruct' or len(backend.base_urls) != 3:
            raise ValueError('Must validate the actual three training Judge replicas')
        if backend._endpoint_max_inflight != endpoint_limit():
            raise ValueError('Judge HTTP admission does not match the selected serving profile')
        from verl.utils import legacy_compact_judge as legacy
        if legacy.enabled() and (rj._judge_granularity() != 'all_rules' or selected_profile()['name'] != 'legacy_compact_v8'):
            raise ValueError('Concise protocol or serving profile mismatch')
        result['serving_profile'] = selected_profile()
        result['endpoint_max_inflight'] = backend._endpoint_max_inflight
        result['api_worker_concurrency'] = int(os.environ['RUBRIC_JUDGE_API_CONCURRENCY'])
        from verl.utils.grouped_grounded_judge import enabled as grouped_enabled
        if grouped_enabled():
            from verl.utils.grouped_dispatch import configuration
            workers,fair=configuration()
            result['grouped_api_worker_concurrency']=workers
            result['grouped_fair_dispatch']=fair
            from verl.utils.grouped_grounded_judge import wire_format
            result['grouped_wire_format']=wire_format()
            if os.getenv('RUBRIC_GROUPED_TUNING_ENABLE','0')=='1':
                from scripts.benchmark_grouped_dispatch import verified_selection
                if (workers,fair,wire_format())!=verified_selection(Path(os.environ['RUBRIC_RUN_DIR'])/'grouped_tuning/selection.json'):
                    raise ValueError('Active grouped dispatch differs from verified measured selection')
        for endpoint in backend.base_urls:
            with urllib.request.urlopen(endpoint + '/models', timeout=30) as handle:
                if backend.model not in {v['id'] for v in json.load(handle)['data']}:
                    raise ValueError('Required training Judge is not served')
        template = template_replay(cases, args.output / 'template', backend)
        profiles = [observed_profile(int(pid), Path(os.environ['RUBRIC_JUDGE_SERVICE_LOG_DIR']) / f'rubric_judge_vllm_{i}.log')
                    for i,pid in enumerate(os.environ['RUBRIC_JUDGE_SERVICE_PIDS'].split(','))]
        atomic_json(args.output / 'serving_profiles.json', profiles)
        if len(profiles) != 3 or not all(v['passed'] for v in profiles):
            raise ValueError('Actual three-replica serving profile not verified')
        data = json.loads(smoke.FIXTURE.read_text()); smoke.validate_fixture(data)
        if digest(smoke.FIXTURE.read_bytes()) != smoke.FIXTURE_SHA256:
            raise ValueError('Frozen twenty-case fixture changed')
        # The real first actor batch replaces the old duplicated-answer 8x8
        # format probe. All original semantic cases still run on all four routes.
        with environment(RUBRIC_BASE_PROTOCOL_GATE=0):
            raw = smoke.run_deployment_gate(data, backend, rj.RubricRules, args.output,
                lambda endpoint: rj.ApiJudge(endpoint, backend.api_key, backend.model))
        raw.update(real_judge_executed=True, status='passed' if raw['passed'] else 'failed',
                   fixture_sha256=smoke.FIXTURE_SHA256)
        atomic_json(args.output / 'targeted_refusal_gate.json', raw)
        result['raw_legacy_blocking'], _ = assess_legacy(data, raw)
        with environment(RUBRIC_BASE_JUDGE_TRACE_DIR=args.output / 'reward_attempts',
                         RUBRIC_TARGETED_JUDGE_TRACE_DIR=args.output / 'correction_attempts'):
            replay_cases = [dict(c, replay_cohort='historical_sixteen') for c in cases] + [
                dict(c, replay_cohort='original_twenty') for c in data['cases']]
            replay = reward_replay(replay_cases, backend, args.output)
        result['blocking'].extend(replay['blocking']); result['warnings'].extend(replay['warnings'])
        consistency = cross_request_audit(args.output)
        consistency['containment_passed'] = cq.containment_verified(consistency, replay['rows'])
        protected_ids = {row['id'] for row in replay['rows']
                         if row['verified_consistency_quarantine'] and row['quarantined']
                         and row['effective_update_weight'] == 0 and row['update_advantage'] == 0}
        blocking, warnings = assess_legacy(data, raw,
            protected_ids if consistency['containment_passed'] else ())
        result['blocking'].extend(blocking); result['warnings'].extend(warnings)
        atomic_json(args.output / 'cross_replica_consistency.json', consistency)
        if not consistency['passed']:
            if consistency['containment_passed']:
                result['warnings'].append(dict(reason='raw_request_conflicts_retained_and_groups_excluded',
                                                conflict_count=len(consistency['conflicts'])))
            else:
                result['blocking'].append(dict(reason='independent_requests_conflict_or_missing_evidence'))
        from scripts.agentic_eight import validation_record
        result['agentic_experiment'] = validation_record()
        result.update(real_judge_executed=True, grammar_passed=grammar['passed'],
            template_passed=template['passed'], serving_profile_passed=True,
            original_semantic_passed=raw['passed'], existing_answers_replayed=replay['case_count'],
            replica_count=3, routes=list(raw['replica_validation']),
            actual_reward_and_advantage_replay_passed=replay['passed'])
        result['admitted'] = not result['blocking']
        result['status'] = 'admitted_with_warnings' if result['admitted'] and result['warnings'] else 'admitted' if result['admitted'] else 'blocked'
    except Exception as error:
        result['blocking'].append(dict(error_type=type(error).__name__, reason=str(error)))
    result['source_sha256'] = {n: digest((ROOT / n).read_bytes()) for n in (
        'verl/utils/frozen_validation_review.py', 'verl/utils/final_stage_resume.py', 'scripts/resume_final_general_v8.sh', 'verl/trainer/main_ppo.py',
        'scripts/agentic_eight.py', 'scripts/train_rubricevolve_agentic_eight.sh',
        'scripts/train_online_rubric_evolve.sh',
        'verl/utils/legacy_compact_judge.py', 'verl/utils/first_batch_snapshot.py',
        'scripts/validate_legacy_compact.py',
        'scripts/graded_presentation_gate.py', 'verl/utils/presentation_reward.py',
        'verl/utils/consistency_quarantine.py', 'verl/utils/tracking.py', 'verl/utils/training_evidence.py',
        'verl/utils/judge_dispatch.py', 'scripts/benchmark_judge_dispatch.py',
        'verl/utils/validation_recovery.py', 'scripts/select_final_checkpoints.py',
        'verl/utils/checkpoint_recovery.py', 'verl/utils/rubric_checkpoint_state.py',
        'verl/utils/best_checkpoint_reference.py',
        'verl/utils/grouped_grounded_judge.py', 'scripts/validate_grouped_grounded.py',
        'scripts/judge_replay_server.py',
        'verl/utils/grouped_dispatch.py', 'scripts/benchmark_grouped_dispatch.py',
        'verl/utils/literal_constraint_guard.py', 'verl/utils/targeted_refusal.py',
        'verl/utils/reward_score/rubric_judge.py', 'verl/utils/reward_score/rubric_judge_save_evolve_rubric.py',
        'verl/trainer/ppo/core_algos.py', 'verl/trainer/ppo/ray_trainer.py', 'train_val.sh')}
    atomic_json(args.output / 'graded_acceptance.json', result)
    print(json.dumps(dict(status=result['status'], blocking=len(result['blocking']), warnings=len(result['warnings']))), flush=True)
    if not result['admitted']:
        raise SystemExit('Operational/impact gate blocked; raw failures retained, no optimizer update.')


if __name__ == '__main__':
    main()
