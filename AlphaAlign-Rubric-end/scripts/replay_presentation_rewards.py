"""Reward/GRPO replay of unchanged answers and recorded REAL Judge outputs.

No Judge, policy, Guard, writer, Ray workers or optimizers are started. Historical
same-question groups are assembled across saved runs, not actual rollout batches.
"""
import argparse
import ast
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from verl.utils import presentation_reward as pr
from verl.utils.reward_score import rubric_judge as rj
from verl.trainer.ppo import core_algos

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = {
    'answer_grounding_existing_answers_20261028.json': 'e1b79ea7d645886ef1bba5a93d4415fd12a7d655bce6993c697f27d02bea5de6',
    'targeted_refusal_existing_answers.json': 'b3f35d0eba94455ae4c3c4646d7e2f8e9749cdde4f9358fc3b451f3fff2b6dc8',
}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def verified_report(folder, name):
    data = (folder / (name + '.json')).read_bytes()
    manifest = read(folder / (name + '.json.integrity.json'))
    if len(data) != manifest['bytes'] or sha(data) != manifest['sha256']:
        raise ValueError('Recorded report integrity failed: ' + name)
    return json.loads(data)


def actual_trainer():
    # Import only the actual function to avoid creating worker/runtime services.
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantage')
    names = ['GAE', 'GRPO', 'REINFORCE_PLUS_PLUS_BASELINE', 'REINFORCE_PLUS_PLUS', 'REMAX', 'RLOO']
    env = dict(DataProto=object, os=os, torch=torch, core_algos=core_algos,
        defaultdict=defaultdict, AdvantageEstimator=SimpleNamespace(**{n: n for n in names}))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_trainer_dispatch', 'exec'), env)
    return env['compute_advantage']


def restore_verdict(case, events, phase):
    rules = rj._build_rules(case['original_info'])
    verdict = {k: [None] * len(v) for k, v in rules.all_rules.items()}
    ids = []
    for event in events:
        if (event['phase'] != phase or case['id'] not in event['case_ids']
                or event['error_type'] is not None):
            continue
        category, i = event['category'], event['rule_index']
        if event['rules'][category] != [rules.all_rules[category][i]]:
            raise ValueError('Recorded rule mismatch')
        raw = json.loads(event['raw_output'])
        if type(raw['score']) is not int or raw['score'] not in (0, 1):
            raise ValueError('Recorded score is not an original binary integer')
        if verdict[category][i] is not None:
            raise ValueError('Repeated verdict cannot be selected for replay')
        verdict[category][i] = raw['score']
        ids.append(event['request_audit']['response_id'])
    if any(x is None for values in verdict.values() for x in values):
        raise ValueError('Missing recorded rules; no scores are filled')
    return dict(verdict, _parse_valid=True), ids


class RecordedVerdicts:
    def __init__(self, rows):
        self.rows = rows

    def judge_batch(self, prompts, responses, rules, infos):
        if len(prompts) != len(self.rows):
            raise ValueError('Replay batch count mismatch')
        outputs = []
        for row, prompt, response, rule in zip(self.rows, prompts, responses, rules):
            if prompt != row['prompt'] or response != row['response']:
                raise ValueError('An original answer or prompt was changed')
            expected = {k: row['info'][k] for k in rule.all_rules}
            if rule.all_rules != expected:
                raise ValueError('An original rubric was changed')
            outputs.append(deepcopy(row['verdict']))
        return outputs


def reward_batch(rows, damped, global_reasoning):
    os.environ['RUBRIC_PRESENTATION_DAMPING_ENABLE'] = str(int(damped))
    os.environ['GLOBAL_REASONING_RULE_ENABLE'] = str(int(global_reasoning))
    backend = rj._get_judge_backend
    rj._get_judge_backend = lambda: RecordedVerdicts(rows)
    try:
        return rj.compute_score_batch([r['profile'] for r in rows],
            [r['response'] for r in rows], [None] * len(rows),
            [dict(r['info'], prompt=r['prompt'], reward_profile=r['profile']) for r in rows],
            _skip_evolve_dispatch=True)
    finally:
        rj._get_judge_backend = backend


def replay_cohort(name, rows, global_reasoning):
    old, new = reward_batch(rows, False, global_reasoning), reward_batch(rows, True, global_reasoning)
    groups = defaultdict(list)
    output = []
    for i, (row, a, b) in enumerate(zip(rows, old, new)):
        if a['rubric_rule_passes'] != b['rubric_rule_passes'] or b['score'] > a['score'] + 1e-12:
            raise ValueError('Verdicts changed or downward-only reward bound failed')
        if row['profile'] in ['ordinary_harm', 'adaptive_harm'] and a['score'] != b['score']:
            raise ValueError('Harmful-profile rewards changed')
        if abs(a['score'] - b['score'] - b['presentation_reward_removed']) > 1e-10:
            raise ValueError('Final reward accounting failed')
        key = sha(canonical([row['prompt'], row['profile'], row['info']]).encode())
        groups[key].append(i)
        output.append(dict(row, baseline_reward=a['score'], damped_reward=b['score'],
            removed_reward=b['presentation_reward_removed'],
            presentation_component=b[pr.COMPONENT],
            matched_rule_indices=[j for j, rule in enumerate(row['info']['outcome_useful_rule'])
                                  if pr.is_presentation_only(rule)],
            category_scores=a['rubric_category_scores'], advantages={}))
    eligible = [i for ids in groups.values() if len(ids) >= 2 for i in ids]
    uid_for = {i: key for key, ids in groups.items() for i in ids}
    fn = actual_trainer()
    group_reports = []
    for mode, floor in [('G1', 0.), ('G3', .25)]:
        os.environ.update(RUBRIC_EVOLVE='1', RUBRIC_EVOLVE_REWARD_MODE=(
            'legacy' if mode == 'G1' else 'bounded_residual'),
            RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=str(floor))
        def advantages(rewards, damping):
            os.environ['RUBRIC_PRESENTATION_DAMPING_ENABLE'] = str(int(damping))
            scalars = [r['score'] for r in rewards]
            # Actual trainer tensors are FP32. The guard constructs its FP64
            # counterfactual from these same quantized endpoints.
            batch = SimpleNamespace(batch={'token_level_rewards': torch.tensor(
                [[scalars[i]] for i in eligible], dtype=torch.float32),
                'token_level_scores': torch.tensor([[scalars[i]] for i in eligible], dtype=torch.float32),
                'response_mask': torch.ones(len(eligible), 1)},
                non_tensor_batch={'uid': np.array([uid_for[i] for i in eligible])})
            if damping:
                for key in [pr.COMPONENT, pr.SCORED, pr.CONTENT]:
                    batch.non_tensor_batch[key] = np.array([rewards[i][key] for i in eligible])
            return fn(batch, 'GRPO').batch['advantages'][:, 0]
        before = advantages(old, False)
        uncapped = advantages(new, False)
        after = advantages(new, True)
        # Reference the exact FP64, quantized counterfactual used by bounded_grpo.
        quantized = pr.quantized_components([new[i][pr.SCORED] for i in eligible],
            [new[i][pr.CONTENT] for i in eligible], [new[i][pr.COMPONENT] for i in eligible],
            torch.tensor([[new[i]['score']] for i in eligible], dtype=torch.float32))
        cf = torch.tensor([[new[i]['score']] for i in eligible], dtype=torch.float32).double()
        cf[:, 0] -= quantized.double()
        content, _ = core_algos.compute_grpo_outcome_advantage(cf, torch.ones_like(cf),
            np.array([uid_for[i] for i in eligible]), std_floor=floor)
        if not torch.isfinite(after).all() or (after.double() - content[:, 0]).abs().max() > .050001:
            raise ValueError('Actual GRPO presentation contribution exceeded its cap')
        for j, i in enumerate(eligible):
            output[i]['advantages'][mode] = dict(baseline=float(before[j]),
                damped_without_cap=float(uncapped[j]), bounded=float(after[j]),
                content_reference=float(content[j, 0]))
        for key, ids in groups.items():
            if len(ids) < 2:
                continue
            group_reports.append(dict(mode=mode, group_sha256=key, responses=len(ids),
                row_ids=[rows[i]['id'] for i in ids],
                maximum_presentation_advantage=max(abs(output[i]['advantages'][mode]['bounded']
                    - output[i]['advantages'][mode]['content_reference']) for i in ids)))
    return dict(name=name, rows=output, groups=group_reports,
        group_sizes=dict(Counter(len(v) for v in groups.values())),
        singleton_rows_retained_without_group_advantage=len(rows) - len(eligible),
        actual_reward_function=True, actual_grpo_and_trainer_dispatch=True,
        live_judge_executed=False, policy_rollout_executed=False, optimizer_updates=0)


def numerical_probe():
    """Synthetic reward tensors only; not model answers or semantic evidence."""
    fn = actual_trainer()
    output = {}
    for mode, floor in [('G1', 0.), ('G3', .25)]:
        os.environ.update(RUBRIC_EVOLVE='1', RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',
            RUBRIC_EVOLVE_REWARD_MODE='legacy' if mode == 'G1' else 'bounded_residual',
            RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=str(floor))
        stages = {}
        for name, credit, bounded in [('undamped', .1, False),
                                     ('damped_only', .02, False), ('bounded', .02, True)]:
            os.environ['RUBRIC_PRESENTATION_DAMPING_ENABLE'] = str(int(bounded))
            scalars = [.3] * 4 + [.3 + credit] * 4
            tensor = torch.tensor(scalars).reshape(8, 1)
            data = SimpleNamespace(batch={'token_level_rewards': tensor,
                'token_level_scores': tensor.clone(), 'response_mask': torch.ones(8, 1)},
                non_tensor_batch={'uid': np.array(['probe'] * 8),
                    pr.SCORED: scalars, pr.CONTENT: [.3] * 8,
                    pr.COMPONENT: [0.] * 4 + [credit] * 4})
            advantage = fn(data, 'GRPO').batch['advantages'][:, 0]
            stages[name] = dict(rewards=scalars, advantages=advantage.tolist(),
                max_absolute_advantage=float(advantage.abs().max()), sum=float(advantage.sum()))
        if stages['bounded']['max_absolute_advantage'] > .050001:
            raise ValueError('Synthetic style-only probe exceeded its advantage cap')
        output[mode] = dict(std_floor=floor, stages=stages)
    return dict(kind='Synthetic numeric stress test, not real answers or Judge calls',
                groups=output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--v7-evidence', type=Path, required=True)
    parser.add_argument('--historical-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(REWARD_MODE='rubric_safe_helpful_balanced', RUBRIC_EVOLVE='0',
        RUBRIC_TARGETED_REFUSAL_ENABLE='0', RUBRIC_EVOLVE_STD_FLOOR_SCOPE='global',
        RUBRIC_PRESENTATION_REWARD_FACTOR='.20', RUBRIC_PRESENTATION_ADV_CAP='.05',
        RUBRIC_FORMAT_WEIGHT='.10', RUBRIC_FORMAT_REQUIRE_THINK='1',
        RUBRIC_LENGTH_PENALTY_WEIGHT='0', RUBRIC_CRITICAL_FAILURE_GATE='off',
        GLOBAL_REASONING_RULE_WEIGHT='2', GLOBAL_REASONING_RULE_ENABLE='0')
    fixed = {}
    for name, expected in FIXTURES.items():
        path = ROOT / 'tests/fixtures' / name
        if sha(path.read_bytes()) != expected:
            raise ValueError('Frozen examples were changed')
        fixed[name] = read(path)['cases']
    cases = fixed['answer_grounding_existing_answers_20261028.json']
    legacy = fixed['targeted_refusal_existing_answers.json']
    legacy_by_id = {c['id']: c for c in legacy}
    evidence = args.v7_evidence
    original_reports = {n: verified_report(evidence, n) for n in
                        ['judge_result', 'legacy_gate', 'consistency_gate', 'guard_result']}
    traces = {arm: read(args.historical_dir / (arm + '_debug.json')) for arm in
              ['oldG1', 'newG1', 'oldG3', 'newG3']}
    trace_maps = {arm: {x['line']: x for x in rows} for arm, rows in traces.items()}
    cohorts = []
    for name, source_cases, file, phase in [
            ('v7_existing16', cases, 'base_attempts.jsonl', 'unflagged_base_answer_semantics'),
            ('v7_frozen20', legacy, 'legacy_attempts.jsonl', 'original_rule_recheck')]:
        events = [json.loads(x) for x in (evidence / file).read_text().splitlines()]
        rows = []
        for c in source_cases:
            verdict, response_ids = restore_verdict(c, events, phase)
            if c['id'] in legacy_by_id:
                profile = legacy_by_id[c['id']]['profile']
            else:
                p = c['provenance'];profile = trace_maps[p['arm']][p['log_line']]['score']['safety_pred_pair'].split('::')[-1]
            rows.append(dict(id=c['id'], prompt=c['prompt'], response=c['response'],
                info=c['original_info'], verdict=verdict, profile=profile,
                provenance=dict(real_v7_response_ids=response_ids)))
        cohorts.append(replay_cohort(name, rows, False))
    targets = {rj._grounded_question_context(c['prompt']).strip() for c in cases}
    historical = []
    for arm, trace_rows in traces.items():
        for row in trace_rows:
            if rj._grounded_question_context(row['prompt']).strip() not in targets:
                continue
            s = row['score']
            historical.append(dict(id=f'{arm}:{row["line"]}', prompt=row['prompt'],
                response=row['response'], info=s['rubric_rules'],
                verdict=s['rubric_rule_passes'], profile=s['safety_pred_pair'].split('::')[-1],
                provenance=dict(arm=arm, log_line=row['line'],
                    global_step=s.get('rubric_evolve_global_step'), saved_score=s['score'])))
    cohorts.append(replay_cohort('historical_same_question_stress_groups', historical, True))
    inputs = [evidence / (n + '.json') for n in original_reports]
    inputs += [evidence / n for n in ['base_attempts.jsonl', 'legacy_attempts.jsonl']]
    inputs += [args.historical_dir / (arm + '_debug.json') for arm in traces]
    inputs += [args.historical_dir / 'audit.json']
    report = dict(kind='Deterministic replay of real recorded Judge verdicts',
        offline_numeric_checks_passed=True, new_semantic_gate_executed=False,
        settings=dict(presentation_reward_factor=.20, presentation_adv_cap=.05,
            reward_mode='rubric_safe_helpful_balanced', format_weight=.10,
            critical_gate='off', evolve_dispatch_executed=False,
            global_reasoning_rule='off for v7; on with original weight 2 for historical logs'),
        frozen_fixture_sha256=FIXTURES, original_report_status={k: v['passed'] for k, v in original_reports.items()},
        input_files={str(p): dict(sha256=sha(p.read_bytes()), bytes=p.stat().st_size) for p in inputs},
        source_hashes={str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in [
            ROOT / x for x in ['verl/utils/presentation_reward.py',
            'verl/utils/reward_score/rubric_judge.py', 'verl/trainer/ppo/core_algos.py',
            'verl/trainer/ppo/ray_trainer.py', 'scripts/replay_presentation_rewards.py']]},
        cohorts=cohorts, synthetic_normalization_probe=numerical_probe(),
        optimizer_updates=0, new_model_calls=0,
        full_training_started=False,
        limitations=['Historical same-prompt groups are assembled across runs, not observed training batches.',
            'V7 and historical Judge protocols are separate cohorts, never pooled.',
            'Singleton cases remain in reward tables; no artificial rollout copies are made.',
            'No new semantic validation, formal three-replica validation, policy rollout or optimizer update occurred.',
            'The online evolution wrapper was not executed; changed scalar rewards with stale accounting are rejected.',
            'The style cap is relative to content-only GRPO, not a cap on total content advantages or an error rate.',
            'Presentation damping does not repair factual/completion Judge mistakes or prove benchmark improvement.'])
    (args.output / 'replay_result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(cohorts=[dict(name=c['name'], responses=len(c['rows']),
        group_sizes=c['group_sizes']) for c in cohorts], new_model_calls=0, optimizer_updates=0)))


if __name__ == '__main__':
    main()
