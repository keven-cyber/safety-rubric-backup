"""Regression checks for online evolution and complete final evaluation."""
import importlib.util
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('review_evolution', ROOT / 'verl/utils/reward_score/rubric_judge_save_evolve_rubric.py')
ev = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ev
spec.loader.exec_module(ev)


def test_holdout_must_use_distinct_prompts(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL', '3')
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL', '2')
    records = [dict(trajectory_hash=f'{label}-{i}', prompt='same request', label=label)
               for label in (0, 1) for i in range(8)]
    proposal, heldout, _ = ev._split_contrastive_replay(records)
    assert not ({r['prompt'] for r in proposal} & {r['prompt'] for r in heldout})


def test_rule_survives_epoch_without_new_promotions(monkeypatch):
    class Base:
        seen = []
        @staticmethod
        def _build_rules(info):
            return ev.RubricRules(**ev._normalize_rules_dict(info))
        def compute_score_batch(self, **kwargs):
            self.seen.append(kwargs['extra_infos'][0]['outcome_safety_rule'])
            return [{}]
    base = Base()
    monkeypatch.setattr(ev, '_get_base_guard_impl', lambda: base)
    monkeypatch.setattr(ev, '_write_runtime_status', lambda **kwargs: None)
    monkeypatch.setattr(ev, '_update_variance_rule_weights', lambda **kwargs: None)
    monkeypatch.setattr(ev, '_evolve_generation_due', lambda: False)
    monkeypatch.setattr(ev, '_load_evolved_rules', lambda epoch: {
        'ordinary_harm': {'outcome_safety_rule': ['promoted in epoch 1']}
    } if epoch == 1 else {})
    monkeypatch.setattr(ev, '_EVOLVE_CACHE_EPOCH', None)
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE', '0')
    for epoch in (2, 3):
        monkeypatch.setenv('RUBRIC_EPOCH', str(epoch))
        ev._compute_score_batch_online(['harm_sampled'], ['response'], [None],
                                     [{'outcome_safety_rule': ['base']}])
    assert base.seen[0] == base.seen[1]


def test_prompt_group_split_can_reserve_both_labels_without_starvation(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_PROPOSAL_PER_LABEL', '6')
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_VALIDATION_PER_LABEL', '4')
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_MIN_PROPOSAL_PER_LABEL', '3')
    records = [dict(prompt=f'q{q}', trajectory_hash=f'{q}-{label}-{i}', label=label)
               for q in range(8) for label in (0, 1) for i in range(2)]
    proposal, validation, _ = ev._split_contrastive_replay(records)
    assert {item['prompt'] for item in proposal}.isdisjoint({item['prompt'] for item in validation})
    for label in (0, 1):
        assert sum(item['label'] == label for item in proposal) == 6
        assert sum(item['label'] == label for item in validation) == 4
        assert len({item['prompt'] for item in proposal if item['label'] == label}) >= 2
        assert len({item['prompt'] for item in validation if item['label'] == label}) >= 2


def test_resume_must_not_activate_future_rules(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '1')
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE', None)
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE_PATH', None)
    bank = {'outcome_safety_rule': ['rule promoted at 840']}
    ev._record_evolve_promotion('ordinary_harm', bank, 840)
    monkeypatch.setenv('RUBRIC_GLOBAL_STEP', '600')
    assert not ev._active_evolved_bank('ordinary_harm', bank)['outcome_safety_rule']


def test_harmful_refusal_uses_harmful_failure_contrast(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_MIN_CATEGORY_COUNT', '2')
    monkeypatch.setenv('RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN', '0')
    rule = ev.RubricRules([], [], [], ['Refuses to provide the requested harmful procedure.'])
    outputs = [dict(refusal_rule=[v], _parse_valid=True) for v in (1, 1, 0, 0)]
    promoted, _, _ = ev._promote_candidate_rules(
        rule, outputs, [1, 1, 0, 0],
        failure_modes=['safe_helpful', 'safe_helpful', 'harmful_enablement', 'partial_compliance'],
        scope='ordinary_harm')
    assert promoted['refusal_rule']


def test_first_benign_refusal_detector_can_be_promoted(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_MIN_CATEGORY_COUNT', '2')
    monkeypatch.setenv('RUBRIC_EVOLVE_MIN_BLINDSPOT_COUNT', '2')
    monkeypatch.setenv('RUBRIC_EVOLVE_REQUIRE_INCREMENTAL_GAIN', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REFUSAL_POLARITY_FIX', '1')
    baseline = ev._category_pass_rates({}, ev.RubricRules([], [], [], []), scope='benign')
    rule = ev.RubricRules([], [], [], ['Unnecessarily refuses the benign request.'])
    outputs = [dict(refusal_rule=[v], _parse_valid=True) for v in (0, 0, 1, 1)]
    promoted, _, _ = ev._promote_candidate_rules(
        rule, outputs, [1, 1, 0, 0],
        failure_modes=['safe_helpful', 'safe_helpful', 'blanket_refusal', 'unhelpful_evasion'],
        baseline_pass_rates=[baseline] * 4, scope='benign')
    assert promoted['refusal_rule']


def test_bad_benign_refusal_detector_can_be_rolled_back(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_ROLLBACK_MIN_SUPPORT', '2')
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE', None)
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE_PATH', None)
    rule = 'Unnecessarily refuses the benign request.'
    bank = {'refusal_rule': [rule]}
    ev._record_evolve_promotion('benign', bank, 360)
    rules = ev.RubricRules([], [], [], [rule])
    retired = ev._rollback_weak_evolved_rules(
        'benign', bank, list(range(4)),
        {'trajectory_quality': [1, 1, 0, 0],
         'failure_modes': ['safe_helpful', 'safe_helpful', 'blanket_refusal', 'unhelpful_evasion']},
        [rules] * 4, [{'refusal_rule': [v]} for v in (1, 1, 0, 0)], 480)
    assert retired


def test_generation_failure_must_not_return_partial_success():
    tree = ast.parse((ROOT / 'Evaluation/safety/generate.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'batch_generate_responses')
    namespace = {'SamplingParams': lambda **kwargs: kwargs, 'tqdm': lambda x: x}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<actual-generator>', 'exec'), namespace)
    class FailingLLM:
        calls = 0
        def generate(self, *args):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError('simulated CUDA failure after first batch')
            return [SimpleNamespace(outputs=[SimpleNamespace(text='first answer')])]
    with pytest.raises(RuntimeError):
        namespace['batch_generate_responses'](['p1', 'p2', 'p3'], FailingLLM(), batch_size=1)


def test_completed_checkpoint_does_not_enter_another_training_batch():
    tree = ast.parse((ROOT / 'verl/trainer/ppo/ray_trainer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'RayPPOTrainer')
    fit = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    fit.body = [n for n in fit.body if not isinstance(n, ast.ImportFrom)]
    class Config(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)
    class UnexpectedTraining(RuntimeError):
        pass
    def consume_batch(data):
        raise UnexpectedTraining('completed checkpoint still enters optimizer batch')
    namespace = {
        'Tracking': lambda **kw: None,
        'OmegaConf': SimpleNamespace(to_container=lambda *a, **kw: {}),
        'tqdm': lambda **kw: SimpleNamespace(close=lambda: None),
        'DataProto': SimpleNamespace(from_single_dict=consume_batch),
        'os': __import__('os'),
    }
    exec(compile(ast.Module(body=[fit], type_ignores=[]), '<actual-trainer-fit>', 'exec'), namespace)
    fake = SimpleNamespace(
        config=Config(trainer=Config(project_name='review', experiment_name='review', logger=[], total_epochs=3)),
        total_training_steps=1170,
        val_reward_fn=None,
        train_dataloader=[{}],
    )
    fake._load_checkpoint = lambda: setattr(fake, 'global_steps', 1170)
    namespace['fit'](fake)


@pytest.mark.parametrize('profile', ['benign', 'ordinary_harm'])
@pytest.mark.parametrize('gradual', [False, True])
def test_bounded_residual_preserves_base_safety_and_format(monkeypatch, tmp_path, profile, gradual):
    base_spec = importlib.util.spec_from_file_location('review_base', ROOT / 'verl/utils/reward_score/rubric_judge.py')
    base = importlib.util.module_from_spec(base_spec)
    sys.modules[base_spec.name] = base
    base_spec.loader.exec_module(base)
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'bounded_residual')
    monkeypatch.setenv('RUBRIC_EVOLVE_RESIDUAL_CLIP', '0.1')
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '1' if gradual else '0')
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_INITIAL_WEIGHT', '0.1')
    monkeypatch.setenv('RUBRIC_GLOBAL_STEP', '101')
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE', None)
    monkeypatch.setattr(ev, '_EVOLVE_RULE_STATE_PATH', None)
    ev._record_evolve_promotion(profile, {'outcome_useful_rule': ['new rule']}, 100)
    monkeypatch.setenv('GLOBAL_REASONING_RULE_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_FORMAT_WEIGHT', '0.1')
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', 'off')
    harmful = profile == 'ordinary_harm'
    original = dict(outcome_safety_rule=['safety'], outcome_useful_rule=['useful'],
                    refusal_rule=['boundary'], critical_safety_rule_count=1, reward_profile=profile)
    evolved = dict(original, outcome_useful_rule=['useful', 'new rule'], rubric_evolve_scope=profile)
    passes = dict(outcome_safety_rule=[0 if harmful else 1],
                  outcome_useful_rule=[0, 1] if harmful else [1, 0],
                  refusal_rule=[1 if harmful else 0])
    full_rules = base._build_rules(evolved)
    _, full_categories = base._aggregate_scores(passes, full_rules, evolved)
    full_raw, _, _ = base._compute_balanced_rubric_score(full_categories, passes, evolved, harmful)
    base_passes = dict(passes, outcome_useful_rule=[passes['outcome_useful_rule'][0]])
    _, base_categories = base._aggregate_scores(base_passes, base._build_rules(original), original)
    base_raw, _, _ = base._compute_balanced_rubric_score(base_categories, base_passes, original, harmful)
    # Match the production format reward and length penalty.
    result = dict(score=0.9 * full_raw + 0.1 - 0.05, balanced_raw=full_raw,
                  rubric_rule_passes=passes, format=1.0)
    ev._anchor_evolution_rewards(base, [original], [evolved], [result])
    assert abs(result['rubric_evolve_reward_delta']) <= 0.1 + 1e-12
    if harmful:
        assert result['rubric_evolve_reward_delta'] == 0
    else:
        assert result['rubric_evolve_reward_delta'] == pytest.approx(-0.01 if gradual else -0.1)
    assert result['score'] == pytest.approx(0.9 * (base_raw + result['rubric_evolve_reward_delta']) + 0.1 - 0.05)
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'legacy')
    unchanged = dict(result)
    ev._anchor_evolution_rewards(base, [original], [evolved], [result])
    assert result == unchanged


def test_checkpoint_restores_exact_state_and_preserves_later_evidence(monkeypatch, tmp_path):
    from verl.utils.rubric_checkpoint_state import save_rubric_state, restore_rubric_state
    import json
    store, local, checkpoint = [tmp_path / name for name in ('oss', 'local', 'checkpoint')]
    for path in (store, local, checkpoint):
        path.mkdir()
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(store))
    monkeypatch.setenv('RUBRIC_EVOLVE_REPLAY_DIR', str(local))
    bank = store / 'rubric_evolve_epoch2.json'
    bank.write_text('["old rule"]')
    replay = store / 'contrastive_replay_snapshot.json'
    replay.write_text('{"profiles": {}}')
    save_rubric_state(checkpoint, 600, 2)
    bank.write_text('["future rule"]')
    (store / 'rubric_evolve_epoch3.json').write_text('["future epoch"]')
    (local / 'contrastive_replay.json').write_text('{"future": true}')
    assert restore_rubric_state(checkpoint, 600) == 2
    assert json.loads(bank.read_text()) == ['old rule']
    assert not (store / 'rubric_evolve_epoch3.json').exists()
    assert json.loads((local / 'contrastive_replay.json').read_text()) == {'profiles': {}}
    assert any(store.glob('restore_history/*/rubric_evolve_epoch3.json'))
    assert any(store.glob('restore_history/*/local_contrastive_replay.json'))


def test_incomplete_or_legacy_evolution_snapshot_is_rejected(monkeypatch, tmp_path):
    from verl.utils.rubric_checkpoint_state import save_rubric_state, restore_rubric_state
    store = tmp_path / 'oss'
    store.mkdir()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_SAVE_DIR', str(store))
    with pytest.raises(RuntimeError, match='no synchronized rubric state'):
        restore_rubric_state(checkpoint, 600)
    (store / 'runtime_status.json').write_text('{}')
    save_rubric_state(checkpoint, 600, 2)
    (checkpoint / 'rubric_evolution/runtime_status.json').write_text('corrupt')
    with pytest.raises(RuntimeError, match='corrupt'):
        restore_rubric_state(checkpoint, 600)
    monkeypatch.setenv('RUBRIC_EVOLVE', '0')
    assert restore_rubric_state(checkpoint, 600) is None


def test_safety_completeness_fails_before_judge(monkeypatch, tmp_path):
    import json
    spec = importlib.util.spec_from_file_location('review_safety_eval', ROOT / 'Evaluation/safety/eval_answer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, '__file__', str(tmp_path / 'eval_answer.py'))
    monkeypatch.chdir(tmp_path)
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data/wildjailbreak_adv.csv').write_text('prompt\np1\np2\n')
    result = tmp_path / 'result/wildjailbreak_adv/model.json'
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps([{'prompt': 'p1', 'response': 'a'}]))
    monkeypatch.setattr(module, 'build_guard_evaluator', lambda: pytest.fail('Incomplete results must not be judged'))
    with pytest.raises(ValueError, match='Incomplete/mismatched'):
        module.evaluate_safety('wildjailbreak_adv', 'model', require_complete=True)
    # Each deliberately invalid generation needs its own immutable evidence file.
    # Replacing the first input triggers the stronger evidence-overwrite guard
    # before this second completeness assertion is reached.
    result = result.with_name('duplicate_model.json')
    result.write_text(json.dumps([{'prompt': 'p1', 'response': 'a'}, {'prompt': 'p1', 'response': 'a'}]))
    with pytest.raises(ValueError, match='Incomplete/mismatched'):
        module.evaluate_safety('wildjailbreak_adv', 'duplicate_model', require_complete=True)


def test_tiny_evolution_residual_is_not_amplified_to_unit_advantage():
    import torch
    import numpy as np
    from collections import defaultdict
    tree = ast.parse((ROOT / 'verl/trainer/ppo/core_algos.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'compute_grpo_outcome_advantage')
    namespace = dict(torch=torch, np=np, defaultdict=defaultdict)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<actual-grpo>', 'exec'), namespace)
    compute = namespace[fn.name]
    rewards = torch.tensor([[0.0]] * 7 + [[0.01]])
    mask = torch.ones_like(rewards)
    groups = np.array(['same_question'] * 8)
    default, _ = compute(rewards.clone(), mask, groups)
    floor, _ = compute(rewards.clone(), mask, groups, std_floor=0.25)
    assert default.abs().max() > 2
    assert floor.abs().max() < 0.04
    explicit_default, _ = compute(rewards.clone(), mask, groups, std_floor=0)
    assert torch.equal(default, explicit_default)
    strong = torch.tensor([[0.0], [1.0]] * 4)
    strong_default, _ = compute(strong.clone(), mask, groups)
    strong_floor, _ = compute(strong.clone(), mask, groups, std_floor=0.25)
    assert torch.equal(strong_default, strong_floor)


def test_group_rank_guard_only_repairs_clear_harmful_preference(monkeypatch):
    monkeypatch.setenv('RUBRIC_EVOLVE_GROUP_RANK_GUARD', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'bounded_residual')
    def item(base, evolved, safe):
        return dict(score=evolved, rubric_base_final_reward=base,
                    rubric_base_balanced_reward=base, rubric_base_safety_ok=safe,
                    rubric_evolve_reward_delta=evolved-base, format=1.0)
    infos = [dict(uid='q', rubric_evolve_scope='ordinary_harm')] * 3
    results = [item(.2, .11, True), item(.14, .14, False), item(.25, .25, True)]
    # Third member must reset too, avoiding newly introduced inversions.
    ev._guard_evolution_group_ranking(infos, results)
    assert [r['score'] for r in results] == [.2, .14, .25]
    assert all(r['rubric_evolve_group_rank_reset'] == 1 for r in results)
    for extras in ([dict(uid=str(i), rubric_evolve_scope='ordinary_harm') for i in range(2)],
                   [dict(uid='q', rubric_evolve_scope='benign')] * 2,
                   [dict(rubric_evolve_scope='ordinary_harm')] * 2):
        results = [item(.2, .11, True), item(.14, .14, False)]
        ev._guard_evolution_group_ranking(extras, results)
        assert [r['score'] for r in results] == [.11, .14]
    monkeypatch.setenv('RUBRIC_EVOLVE_GROUP_RANK_GUARD', '0')
    results = [item(.2, .11, True), item(.14, .14, False)]
    ev._guard_evolution_group_ranking(infos[:2], results)
    assert results[0]['score'] == .11


def test_four_parallel_arms_have_explicit_distinct_controls():
    expected = [('g1_control', 'legacy', '0', '0', '0'),
                ('g2_residual', 'bounded_residual', '0', '1', '0'),
                ('g3_stdfloor', 'bounded_residual', '0.25', '1', '0'),
                ('g4_rankguard', 'bounded_residual', '0.25', '1', '1')]
    for arm, mode, floor, gradual, rank in expected:
        text = (ROOT / f'scripts/train_rubricevolve_{arm}.sh').read_text()
        for key, value in dict(RUBRIC_EVOLVE_REWARD_MODE=mode, RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=floor,
                               RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE=gradual,
                               RUBRIC_EVOLVE_GROUP_RANK_GUARD=rank).items():
            assert f'export {key}={value}\n' in text
        assert 'train_rubricevolve_guarded_residual.sh' in text

