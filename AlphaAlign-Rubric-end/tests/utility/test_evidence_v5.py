import importlib.util
import ast
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.utils import evolution_evidence as ee
from verl.utils import validation_selection as vs
from verl.utils import benign_contrast as bc

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('evidence_v5_ev', ROOT / 'verl/utils/reward_score/rubric_judge_save_evolve_rubric.py')
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
spec.loader.exec_module(ev)


def metrics(refusal=.1, useful=1.0, coherence=1.0, safety=1.0):
    profiles = {p: dict(count=64, invalid=0, safe_boundary_pass_rate=safety,
                       unnecessary_refusal_rate=refusal, substantive_answer_rate=useful,
                       coherent_answer_rate=coherence) for p in vs.PROFILES}
    return dict(vs.rank_profiles(profiles, True), **{vs.PREFIX + 'calibration_pass': 1.0})


def test_v5_breaks_saturated_score_without_changing_constraints():
    older, improved = metrics(.1), metrics(0)
    assert older[vs.PREFIX + 'rank_score'] == improved[vs.PREFIX + 'rank_score'] == 1
    assert vs.checkpoint_order(improved) > vs.checkpoint_order(older)
    assert vs.checkpoint_order(metrics(0, safety=.97)) < vs.checkpoint_order(older)
    assert vs.checkpoint_order(metrics(0, useful=.79)) < vs.checkpoint_order(older)
    assert vs.checkpoint_order(metrics(0, coherence=.94)) < vs.checkpoint_order(older)
    invalid = dict(improved, **{vs.PREFIX + 'calibration_pass': 0})
    assert vs.checkpoint_order(invalid) is None


def test_frozen_original_rubric_breaks_diagnostic_tie():
    left, right = metrics(0), metrics(0)
    for p in vs.BENIGN:
        left[f'val-aux/{p}/balanced_useful/mean@1'] = .8
        right[f'val-aux/{p}/balanced_useful/mean@1'] = .9
    assert vs.checkpoint_order(right) > vs.checkpoint_order(left)
    right['external/XSTest'] = 1e9
    assert vs.checkpoint_order(right) == vs.checkpoint_order(dict(right, **{'external/XSTest': -1e9}))


def records(mode='blanket_refusal', count=60):
    return [dict(prompt=f'question {q}', trajectory_hash=f'{q}-{label}-{i}', label=label,
                 failure_mode='safe_helpful' if label else mode, global_step=120)
            for q in range(count) for label in (0, 1) for i in range(2)]


@pytest.mark.parametrize('mode,category', list(ee.BENIGN_MODES.items()))
def test_failure_matched_split_is_prompt_disjoint_and_preserves_inputs(mode, category):
    data = records(mode)
    original = json.dumps(data)
    proposal, validation, counts = ee.targeted_split(data, 'benign', 12, 6, 3, 2, 360)
    assert json.dumps(data) == original
    assert counts['target_category'] == category
    assert {ee.prompt_key(r) for r in proposal}.isdisjoint({ee.prompt_key(r) for r in validation})
    for items, minimum in ((proposal, 3), (validation, 6)):
        for label in (0, 1):
            chosen = [r for r in items if r['label'] == label]
            assert len(chosen) >= minimum
            assert len({r['prompt'] for r in chosen}) >= 2
        assert all(r['failure_mode'] == mode for r in items if r['label'] == 0)
    assert all(ee.partition(r) == 'proposal' for r in proposal)
    assert all(ee.partition(r) == 'validation' for r in validation)
    # Assignment does not change when the bank grows in a later round.
    sides = {r['prompt']: ee.partition(r) for r in data}
    for r in records(mode, 90):
        if r['prompt'] in sides:
            assert ee.partition(r) == sides[r['prompt']]


def test_insufficient_evidence_does_not_relax_holdout_count():
    proposal, validation, _ = ee.targeted_split(records(count=2), 'benign', 12, 6, 3, 2, 360)
    assert not proposal and not validation


def test_collection_is_decoupled_from_generation(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE', '1')
    assert ee.collection_due(120)
    assert ee.collection_due(140)
    assert not ee.collection_due(119)
    assert not ee.collection_due(141)
    assert ee.collection_due(360, True)
    monkeypatch.setenv('RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE', '0')
    assert not ee.collection_due(360, True)


def test_failed_and_unknown_diagnoses_remain_in_archive(tmp_path):
    data = [dict(prompt='p', response='failed answer', label=-1, failure_mode='unknown')]
    ee.archive_collection(tmp_path, 120, 'benign', data, 'before_diagnosis')
    ee.archive_collection(tmp_path, 140, 'benign', data, 'diagnosis_failed')
    rows = [json.loads(line) for p in (tmp_path / 'trajectory_archive').glob('*.jsonl')
            for line in p.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['records'] == rows[1]['records'] == data


def test_optimization_statistics_use_absolute_delta_not_zero_mean():
    stats = bc.optimization_statistics(['a', 'a', 'b'], [.1, -.1, None], [.4, -.4, 1.])
    assert stats['benign_contrast/delta_sum'] == 0
    assert stats['benign_contrast/delta_abs_sum'] == pytest.approx(.2)
    assert stats['benign_contrast/changed_groups'] == 1
    assert stats['benign_contrast/changed_responses'] == 2
    with pytest.raises(ValueError):
        bc.optimization_statistics(['a'], [], [])


@pytest.mark.parametrize('order', [(0, 1, 2, 3), (2, 0, 3, 1)])
def test_mixed_batch_dense_padding_is_not_a_numeric_failure(order):
    from collections import defaultdict
    from verl.utils.judge_filter import append_dense_reward_extras, select_aligned_reward_extras

    # Use the actual reward-manager packing contract, including late keys and
    # interleaved harmful/benign rows. Zero is an observed value, not missing.
    rows = [dict(score=.6, benign_contrast_delta=.1), dict(score=.4, benign_contrast_delta=-.1),
            dict(score=.5), dict(score=.5, benign_contrast_delta=0.)]
    traces = [rows[i].copy() for i in order]
    extras = defaultdict(list)
    for i, row in enumerate(traces):
        append_dense_reward_extras(extras, row, i)
    dense, skipped = select_aligned_reward_extras(extras, range(4), 4)
    assert not skipped
    before = [row.copy() for row in traces]
    deltas = dense['benign_contrast_delta'][:]
    stats = bc.optimization_statistics(['same'] * 4, deltas, [.4] * 4, traces)
    assert stats['benign_contrast/observed_responses'] == 3
    assert stats['benign_contrast/not_applicable_responses'] == 1
    assert stats['benign_contrast/changed_responses'] == 2
    assert stats['benign_contrast/delta_abs_sum'] == pytest.approx(.2)
    assert stats['benign_contrast/delta_sum'] == pytest.approx(0.)
    assert traces == before and len(deltas) == len(traces) == 4
    assert all(a == b or math.isnan(a) and math.isnan(b)
               for a, b in zip(deltas, dense['benign_contrast_delta']))


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_explicit_nonfinite_contrast_is_never_treated_as_missing(bad):
    with pytest.raises(ValueError, match='Non-finite'):
        bc.optimization_statistics(['a'], [bad], [1.], [dict(benign_contrast_delta=bad)])
    # Without provenance, a NaN cannot be assumed to mean "not applicable".
    with pytest.raises(ValueError, match='Non-finite'):
        bc.optimization_statistics(['a'], [bad], [1.])


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_nonfinite_advantage_on_nonapplicable_row_still_fails(bad):
    with pytest.raises(ValueError, match='Non-finite.*advantage'):
        bc.optimization_statistics(['harm'], [float('nan')], [bad], [dict(score=.5)])


def test_warmup_and_no_eligible_rows_are_reported_without_fabricating_zero_deltas():
    stats = bc.optimization_statistics(['a', 'b'], [None, None], [.1, -.1], [dict(score=.5)] * 2)
    assert stats['benign_contrast/observed_responses'] == 0
    assert stats['benign_contrast/not_applicable_responses'] == 2
    assert stats['benign_contrast/changed_responses'] == 0


@pytest.mark.parametrize('deltas,traces', [
    ([.1], [{}]), ([float('inf')], [{}]), ([None], [dict(benign_contrast_delta=.1)]),
    ([.1], [dict(benign_contrast_delta=.2)]), ([.1], []), ([.1], [None]),
])
def test_contrast_provenance_must_be_aligned_and_consistent(deltas, traces):
    with pytest.raises(ValueError):
        bc.optimization_statistics(['a'], deltas, [1.], traces)


@pytest.mark.parametrize('name', ['token_level_scores', 'token_level_rewards', 'advantages'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf')])
def test_real_nonfinite_tensors_fail_even_at_masked_positions(name, bad):
    import torch
    values = torch.tensor([[.1, bad]])
    with pytest.raises(ValueError, match=name):
        bc.validate_optimization_tensors(**{name: values})
    assert not torch.isfinite(values[0, 1])  # The guard must not repair/mutate it.
    bc.validate_optimization_tensors(**{name: torch.tensor([[0., .1]])})


def test_actual_trainer_checks_tensors_and_passes_raw_provenance_before_update():
    source = (ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text()
    begin = source.index("# compute rewards. apply_kl_penalty")
    end = source.index('# update critic', begin)
    actual = source[begin:end]
    assert 'token_level_scores=batch.batch' in actual
    assert 'token_level_rewards=batch.batch' in actual
    assert "validate_optimization_tensors(advantages=batch.batch['advantages'])" in actual
    assert 'reward_trace_info=reward_trace_info or None' in actual
    assert actual.index('token_level_rewards=batch.batch') < actual.index('batch = compute_advantage')
    assert 'nan_to_num' not in actual


def test_candidate_calibration_failure_blocks_candidate_call(monkeypatch):
    report = dict(calibration=[dict(passed=False)], records=[])
    monkeypatch.setattr(ee, 'evaluate_validation', lambda *a: ({}, report))
    class Judge:
        def judge_batch(self, *args):
            raise AssertionError('must not judge candidates after failed calibration')
    outputs, evidence = ee.candidate_outputs(Judge(), ['p'], ['r'], None, [{}], None)
    assert not evidence['passed'] and outputs == [{'_parse_valid': False}]


def test_candidate_validation_adds_reference_and_scoped_protocol(monkeypatch):
    rows = [dict(profile='benign', parse_valid=True, substantive_answer=s,
                 coherent_answer=c, unnecessary_refusal=r) for s, c, r in [(1, 1, 0), (0, 1, 0)]]
    rows += [dict(profile='ordinary_harm', parse_valid=True, safe_boundary_pass=s) for s in (1, 0)]
    monkeypatch.setattr(ee, 'evaluate_validation', lambda *a: ({}, dict(calibration=[dict(passed=True)], records=rows)))
    rules = ev.RubricRules([], ['check'], [], [])
    class Judge:
        def judge_batch(self, prompts, responses, supplied_rules, extras):
            assert 'reference requirements' in prompts[0]
            assert 'grounded fact' in prompts[0]
            assert extras[0]['rubric_predicate_v5'] is True
            return [dict(outcome_useful_rule=[1], _parse_valid=True)]
    outputs, evidence = ee.candidate_outputs(Judge(), ['p'], ['r'], rules,
                                             [{'outcome_useful_rule': ['grounded fact']}], ev.RubricRules)
    assert evidence['passed'] and outputs[0]['_parse_valid']


def test_all_pass_rule_and_single_prompt_never_get_promoted(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN', '0')
    rules = ev.RubricRules([], ['observable predicate'], [], [])
    def outs(values):
        return [dict(outcome_safety_rule=[], outcome_useful_rule=[v],
                     reasoning_process_rule=[], refusal_rule=[], _parse_valid=True) for v in values]
    kwargs = dict(labels=[1, 1, 0, 0], failure_modes=['safe_helpful'] * 2 + ['incorrect_answer'] * 2,
                  scope='benign')
    for values, keys in (([1, 1, 1, 1], ['a', 'b', 'a', 'b']), ([1, 1, 0, 0], ['a'] * 4)):
        promoted, _, _ = ev._promote_candidate_rules(rules, outs(values), prompt_keys=keys, **kwargs)
        assert not any(promoted.values())
    promoted, _, _ = ev._promote_candidate_rules(rules, outs([1, 1, 0, 0]),
                                                 prompt_keys=['a', 'b', 'a', 'b'], **kwargs)
    assert promoted['outcome_useful_rule']


def test_new_launcher_is_opt_in_and_keeps_dataset_and_standard():
    source = (ROOT / 'scripts/train_rubricevolve_evidence_v5.sh').read_text()
    assert 'RUBRIC_JUDGE_DROP_FAILED=0' in source
    assert 'RUBRIC_SELECTION_ORDER_V5_ENABLE=1' in source
    assert 'RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE=1' in source
    assert 'RUBRIC_EVOLVE_MIN_GAP=' not in source
    assert 'train_rubricevolve_guarded_residual.sh' in source
    for arm in ('g1_calibrated_selection', 'g3_calibrated_selection'):
        assert 'EVIDENCE_V5_ENABLE' not in (ROOT / f'scripts/train_rubricevolve_{arm}.sh').read_text()


@pytest.mark.parametrize('diagnosis_fails', [False, True])
def test_production_collection_before_generation_preserves_rewards_and_evidence(monkeypatch, tmp_path, diagnosis_fails):
    class Base:
        RubricRules = ev.RubricRules
        @staticmethod
        def _build_rules(info):
            return ev.RubricRules(**ev._normalize_rules_dict(info))
        @staticmethod
        def compute_score_batch(**kw):
            return [dict(score=.5, judge_parse_valid=True, rubric_rule_passes={}) for _ in kw['solution_strs']]
        @staticmethod
        def _get_judge_backend():
            return None
    monkeypatch.setenv('RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(tmp_path / 'oss'))
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_DIR', str(tmp_path / 'local'))
    monkeypatch.setenv('RUBRIC_GLOBAL_STEP', '120')
    monkeypatch.setenv('RUBRIC_EPOCH', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '0')
    monkeypatch.setattr(ev, '_EVOLVE_CACHE_EPOCH', None)
    monkeypatch.setattr(ev, '_RUNTIME_STATUS', {})
    monkeypatch.setattr(ev, '_get_base_guard_impl', lambda: Base())
    monkeypatch.setattr(ev, '_evolve_generation_due', lambda: False)
    monkeypatch.setattr(ev, '_load_evolved_rules', lambda epoch: {})
    monkeypatch.setattr(ev, '_update_variance_rule_weights', lambda **kw: None)
    monkeypatch.setattr(ev, '_anchor_evolution_rewards', lambda *a: None)
    monkeypatch.setattr(ev, '_guard_evolution_group_ranking', lambda *a: None)
    monkeypatch.setattr(ev, '_select_evolution_examples', lambda indices, *a: indices)
    def diagnose(**kw):
        if diagnosis_fails:
            raise ValueError('deliberately invalid diagnosis')
        return dict(trajectory_quality=[1, 0], failure_modes=['safe_helpful', 'blanket_refusal'])
    monkeypatch.setattr(ev, '_diagnose_evolution_trajectories', diagnose)
    monkeypatch.setattr(ev, '_generate_evolved_rules', lambda **kw: (_ for _ in ()).throw(AssertionError('writer not due')))
    results = ev._compute_score_batch_online(['benign'] * 2, ['good', 'refused'], [None] * 2,
                                             [dict(prompt='same prompt', uid='u')] * 2)
    assert [r['score'] for r in results] == [.5, .5]
    archived = [json.loads(line) for p in (tmp_path / 'oss/trajectory_archive').glob('*.jsonl')
                for line in p.read_text().splitlines()]
    assert archived[0]['status'] == 'before_diagnosis'
    assert archived[0]['records'][1]['response'] == 'refused'
    assert len(archived) == (1 if diagnosis_fails else 2)
    assert ev._RUNTIME_STATUS.get('generation_due_count', 0) == 0
    assert ev._RUNTIME_STATUS.get('collection_due_count') == 1


def trainer_method(name):
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(os=os, json=json, math=math)
    exec(compile(ast.Module(body=[method], type_ignores=[]), '<actual-trainer>', 'exec'), namespace)
    return namespace[name]


def test_trainer_replaces_saturated_best_and_exact_tie_does_not_churn(monkeypatch):
    monkeypatch.setenv('RUBRIC_SELECTION_ORDER_V5_ENABLE', '1')
    saves = []
    fake = SimpleNamespace(save_best_ckpt=True, _latest_val_step=600,
                           _latest_val_metrics=metrics(0),
                           best_metric_key=vs.PREFIX + 'rank_score', best_metric_mode='max',
                           best_metric_value=1.0, best_metric_step=180,
                           _best_selection_order=vs.checkpoint_order(metrics(.1)),
                           _pick_best_metric_key=lambda m: vs.PREFIX + 'rank_score',
                           _save_best_checkpoint_copy=lambda **kw: saves.append(kw),
                           _save_best_checkpoint_info=lambda: None)
    fn = trainer_method('_maybe_save_best_checkpoint')
    fn(fake, 600)
    assert fake.best_metric_step == 600 and len(saves) == 1
    fake._latest_val_step = 630
    fn(fake, 630)
    assert fake.best_metric_step == 600 and len(saves) == 1
    fake._latest_val_metrics = metrics(0, safety=.97)
    fake._latest_val_step = 660
    fn(fake, 660)
    assert fake.best_metric_step == 600


def test_selection_order_survives_resume(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_SELECTION_ORDER_V5_ENABLE', '1')
    fake = SimpleNamespace(save_best_ckpt=True, _best_ckpt_root=str(tmp_path),
                           _best_ckpt_info_path=str(tmp_path / 'best.json'),
                           _best_ckpt_iter_path=str(tmp_path / 'iteration.txt'),
                           best_metric_key=vs.PREFIX + 'rank_score', best_metric_mode='max',
                           best_metric_value=1.0, best_metric_step=600,
                           _best_selection_order=vs.checkpoint_order(metrics(0)),
                           _latest_val_metrics=metrics(0))
    trainer_method('_save_best_checkpoint_info')(fake)
    restored = SimpleNamespace(_best_ckpt_info_path=fake._best_ckpt_info_path,
                               best_metric_key=fake.best_metric_key, best_metric_mode='max')
    trainer_method('_load_best_checkpoint_info')(restored)
    assert restored._best_selection_order == fake._best_selection_order
    assert restored.best_metric_step == 600


def test_full_batch_contrast_funnel_is_saved_without_raw_text(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    rows = [dict(parse_valid=True, substantive_answer=s, coherent_answer=1, unnecessary_refusal=r)
            for s, r in ((1, 0), (0, 1), (0, 1), (0, 1))]
    monkeypatch.setattr(bc, 'evaluate_validation', lambda *a: ({}, dict(calibration=[dict(passed=True)], records=rows)))
    outputs = [dict(score=.5) for _ in range(4)]
    report = bc.apply_benign_contrast(['secret q'] * 4, ['secret answer'] * 4,
                                     ['benign'] * 4, [dict(uid=u) for u in ('a', 'a', 'b', 'b')],
                                     outputs, None, None, 360)
    assert report['eligible_groups'] == 2 and report['mixed_groups'] == 1
    assert report['refusal_only_groups'] == 1 and report['changed_responses'] == 2
    saved = (tmp_path / 'benign_contrast/step_00000360.json').read_text()
    assert 'secret' not in saved and json.loads(saved)['full_batch'] is True


def test_v5_protocol_is_scoped_to_actual_api_and_legacy_local_prompt_unchanged(monkeypatch):
    from types import ModuleType
    from verl.utils.reward_score import rubric_judge as rj
    sent = []
    def create(**kwargs):
        sent.append(kwargs['messages'][0]['content'])
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({
            'outcome_safety_rule': [], 'outcome_useful_rule': [1],
            'reasoning_process_rule': [], 'refusal_rule': []})), finish_reason='stop')])
    fake = ModuleType('openai')
    fake.OpenAI = lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setitem(sys.modules, 'openai', fake)
    monkeypatch.setenv('RUBRIC_JUDGE_API_LOG_ENABLE', '0')
    api = rj.ApiJudge('http://localhost:8000/v1', 'EMPTY', 'judge')
    rules = rj.RubricRules([], ['A true predicate.'], [], [])
    for extra in ({}, {'rubric_predicate_v5': True}):
        assert api.judge('p', '<answer>a</answer>', rules, extra)['_parse_valid']
    assert 'PREDICATE PROTOCOL V5' not in sent[0]
    assert 'PREDICATE PROTOCOL V5' in sent[1]
    local_source = (ROOT / 'verl/utils/reward_score/rubric_judge.py').read_text()
    local_class = next(n for n in ast.parse(local_source).body if isinstance(n, ast.ClassDef) and n.name == 'LocalJudge')
    render = next(n for n in local_class.body if isinstance(n, ast.FunctionDef) and n.name == '_render_prompt')
    assert 'rubric_predicate_v5' not in ast.get_source_segment(local_source, render)
