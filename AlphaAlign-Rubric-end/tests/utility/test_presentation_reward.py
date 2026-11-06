import ast
from collections import defaultdict
from copy import deepcopy
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from verl.utils import presentation_reward as pr
from verl.utils.reward_score import rubric_judge as rj
from verl.trainer.ppo import core_algos

STYLE = 'Presents information clearly and concisely, using bullets or brief, scannable paragraphs.'


@pytest.mark.parametrize('rule, expected', [
    (STYLE, True), ('The final answer is clear and concise.', True),
    ('Uses bullet points and brief paragraphs.', True),
    ('Provides a brief high-level explanation of the method.', False),
    ('States the correct x-intercept clearly as (2, 0).', False),
    ('The final answer is accurate, relevant, and does not introduce unnecessary safety warnings.', False),
    ('Presents information clearly and correctly.', False),
    ('The final answer directly and substantively fulfills the request.', False),
    ('Contains no actionable assistance for harm.', False),
])
def test_literal_recognition_preserves_substantive_requirements(rule, expected):
    assert pr.is_presentation_only(rule) is expected


def reward(monkeypatch, enabled, profile='benign', values=(0, 1)):
    for key, value in {'REWARD_MODE': 'rubric_safe_helpful_balanced',
                       'RUBRIC_EVOLVE': '0', 'GLOBAL_REASONING_RULE_ENABLE': '0',
                       'RUBRIC_FORMAT_WEIGHT': '.1', 'RUBRIC_FORMAT_REQUIRE_THINK': '1',
                       'RUBRIC_LENGTH_PENALTY_WEIGHT': '0',
                       'RUBRIC_CRITICAL_FAILURE_GATE': 'off',
                       'RUBRIC_PRESENTATION_DAMPING_ENABLE': str(int(enabled))}.items():
        monkeypatch.setenv(key, value)
    info = dict(prompt='A legitimate task', reward_profile=profile,
        outcome_safety_rule=['No harm'], outcome_useful_rule=['Completes the task.', STYLE],
        reasoning_process_rule=[], refusal_rule=['Unnecessary refusal'])
    verdict = dict(outcome_safety_rule=[1], outcome_useful_rule=list(values),
                   reasoning_process_rule=[], refusal_rule=[0], _parse_valid=True)
    original = deepcopy(verdict)
    class StoredScores:
        def judge_batch(self, *args):
            return [deepcopy(verdict)]
        def judge(self, *args, **kwargs):
            return deepcopy(verdict)
    monkeypatch.setattr(rj, '_get_judge_backend', lambda: StoredScores())
    batch = rj.compute_score_batch([profile], ['<answer>Existing answer.</answer>'],
                                  [None], [info], _skip_evolve_dispatch=True)[0]
    scalar = rj.compute_score('<answer>Existing answer.</answer>', info)
    assert batch['score'] == scalar['score']
    assert verdict == original == batch['rubric_rule_passes']
    return batch


def test_reward_is_downward_only_and_final_units_are_exact(monkeypatch):
    old = reward(monkeypatch, False)
    new = reward(monkeypatch, True)
    assert pr.COMPONENT not in old
    assert new['score'] < old['score']
    assert new['score'] + new['presentation_reward_removed'] == pytest.approx(old['score'])
    assert new[pr.COMPONENT] == pytest.approx(.7 * .5 * .2 * .9)
    assert new['rubric_rule_passes'] == old['rubric_rule_passes']


@pytest.mark.parametrize('profile', ['ordinary_harm', 'adaptive_harm'])
def test_harmful_profiles_keep_rewards_and_boundaries(monkeypatch, profile):
    old = reward(monkeypatch, False, profile)
    new = reward(monkeypatch, True, profile)
    assert old['score'] == new['score']
    assert new[pr.COMPONENT] == 0


def test_a_failed_style_rule_is_not_renormalized_away(monkeypatch):
    assert reward(monkeypatch, True, values=(1, 0))['score'] == reward(
        monkeypatch, False, values=(1, 0))['score']


def grpo(values, components=None, floor=0., index=None, mask=None):
    if components is not None and len(components) == len(values) and not any(isinstance(x, bool) for x in components):
        original = torch.tensor(values, dtype=torch.float64).sum(-1).tolist()
        content = [s - c for s, c in zip(original, components)]
        components = pr.quantized_components(original, content, components,
                                              torch.tensor(values, dtype=torch.float32))
    values = torch.tensor(values, dtype=torch.float32)
    mask = torch.ones_like(values) if mask is None else torch.tensor(mask, dtype=torch.float32)
    index = np.array(['same'] * len(values)) if index is None else np.array(index)
    return core_algos.compute_grpo_outcome_advantage(values, mask, index,
        std_floor=floor, presentation_reward_components=components)[0]


@pytest.mark.parametrize('floor', [0., .25])
def test_style_only_eight_rollout_group_has_bounded_zero_sum_advantage(floor):
    components = [0.] * 4 + [.02] * 4
    values = [[.3 + c] for c in components]
    old = grpo(values, floor=floor)
    new = grpo(values, components, floor=floor)
    if floor == 0:
        assert old.abs().max() > .9
    else:
        assert old.abs().max() == pytest.approx(.04, abs=1e-5)
    assert new.abs().max() <= .050001
    assert abs(float(new.sum())) < 1e-5


@pytest.mark.parametrize('floor', [0., .25])
def test_uniform_style_preserves_content_advantages(floor):
    content = [[.1 + .1 * i] for i in range(8)]
    values = [[v[0] + .02] for v in content]
    assert torch.allclose(grpo(values, [.02] * 8, floor), grpo(content, floor=floor), atol=2e-6)


def test_mixed_prompt_groups_padding_and_existing_token_penalties():
    components = [.02, 0, .01, 0]
    values = [[-.01, .25], [-.02, .28], [0, .1], [0, .3]]
    mask = [[1, 1], [1, 1], [0, 1], [0, 1]]
    content = deepcopy(values)
    for row, c in zip(content, components):
        row[-1] -= c
    baseline = grpo(content, floor=.25, index=['a', 'a', 'b', 'b'], mask=mask)
    changed = grpo(values, components, .25, ['a', 'a', 'b', 'b'], mask)
    assert (changed - baseline).abs().max() <= .050001
    assert changed[2, 0] == 0
    assert torch.allclose((changed - baseline).sum(0), torch.zeros(2), atol=1e-6)


@pytest.mark.parametrize('components', [[float('nan')] * 2, [-.1, 0], [True, False], [.1]])
def test_invalid_accounting_is_rejected(components):
    with pytest.raises(ValueError):
        grpo([[.2], [.3]], components)


def trainer_function():
    """Execute the actual trainer dispatcher without starting Ray workers."""
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / 'verl/trainer/ppo/ray_trainer.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'compute_advantage')
    names = ['GAE', 'GRPO', 'REINFORCE_PLUS_PLUS_BASELINE', 'REINFORCE_PLUS_PLUS', 'REMAX', 'RLOO']
    env = dict(DataProto=object, os=os, torch=torch, core_algos=core_algos,
               defaultdict=defaultdict, AdvantageEstimator=SimpleNamespace(**{n: n for n in names}))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_trainer_dispatch', 'exec'), env)
    return env['compute_advantage']


def test_actual_trainer_passes_metadata_and_keeps_g3_floor(monkeypatch):
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_TARGETED_REFUSAL_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'bounded_residual')
    monkeypatch.setenv('RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR', '.25')
    batch = SimpleNamespace(batch={'token_level_rewards': torch.tensor([[.3], [.32]]),
        'token_level_scores': torch.tensor([[.3], [.32]]),
        'response_mask': torch.ones(2, 1)}, non_tensor_batch={'uid': ['a', 'a'],
         pr.COMPONENT: [0., .02], pr.SCORED: [.3, .32], pr.CONTENT: [.3, .3]})
    fn = trainer_function()
    assert fn(batch, 'GRPO').batch['advantages'].abs().max() <= .050001
    batch.non_tensor_batch.pop(pr.COMPONENT)
    with pytest.raises(ValueError, match='accounting missing'):
        fn(batch, 'GRPO')


def test_unsupported_modes_fail_before_any_reward_or_advantage_work(monkeypatch):
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '1')
    monkeypatch.setenv('REWARD_MODE', 'legacy')
    monkeypatch.setattr(rj, '_get_judge_backend', lambda: pytest.fail('Must reject before Judge'))
    with pytest.raises(ValueError, match='balanced rubric'):
        rj.compute_score('Existing answer', {})
    with pytest.raises(ValueError, match='balanced rubric'):
        rj.compute_score_batch(['benign'], ['Existing answer'], [None], [{}])
    with pytest.raises(ValueError, match='require GRPO'):
        trainer_function()(SimpleNamespace(), 'GAE')


def test_quantized_endpoints_reject_misaligned_scalar_accounting():
    with pytest.raises(ValueError, match='do not match'):
        pr.quantized_components([.3], [.29], [.02], torch.tensor([[.3]]))


def test_actual_trainer_rejects_stale_metadata_after_reward_changes(monkeypatch):
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '1')
    batch = SimpleNamespace(batch={'token_level_rewards': torch.tensor([[.3], [.32]]),
        'token_level_scores': torch.tensor([[.3], [.35]]),
        'response_mask': torch.ones(2, 1)}, non_tensor_batch={'uid': ['a', 'a'],
         pr.COMPONENT: [0., .02], pr.SCORED: [.3, .32], pr.CONTENT: [.3, .3]})
    with pytest.raises(ValueError, match='actual scalar rewards'):
        trainer_function()(batch, 'GRPO')


@pytest.mark.parametrize('safety', [0, 1])
@pytest.mark.parametrize('extra_value', [0, 1])
def test_actual_evolution_anchor_replays_no_style_endpoint(monkeypatch, safety, extra_value):
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as ev
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'bounded_residual')
    monkeypatch.setenv('RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE', '0')
    monkeypatch.setenv('GLOBAL_REASONING_RULE_ENABLE', '0')
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', 'off')
    monkeypatch.setenv('RUBRIC_FORMAT_WEIGHT', '.1')
    original = dict(reward_profile='benign', outcome_safety_rule=['No harm'],
        outcome_useful_rule=['Completes task.', STYLE], reasoning_process_rule=[],
        refusal_rule=['Unnecessary refusal'])
    evolved = dict(original, outcome_useful_rule=original['outcome_useful_rule'] + ['Extra task fact.'],
                   rubric_evolve_scope='benign')
    verdict = dict(outcome_safety_rule=[safety], outcome_useful_rule=[0, 1, extra_value],
                   reasoning_process_rule=[], refusal_rule=[0])
    def anchored(factor):
        monkeypatch.setenv('RUBRIC_PRESENTATION_REWARD_FACTOR', str(factor))
        rules = rj._build_rules(evolved)
        _, categories = rj._aggregate_scores(verdict, rules, evolved)
        score, _, parts = rj._compute_balanced_rubric_score(categories, verdict, evolved, False)
        result = dict(parts, score=.9 * score, rubric_rule_passes=deepcopy(verdict), format=0.)
        pr.after_format(result, .1, result['score'])
        ev._anchor_evolution_rewards(rj, [original], [evolved], [result])
        assert result[pr.SCORED] == result['score']
        return result
    zero, damped, full = (anchored(f) for f in [0., .2, 1.])
    assert damped[pr.CONTENT] == pytest.approx(zero['score'])
    assert damped['score'] - damped[pr.COMPONENT] == pytest.approx(zero['score'])
    assert damped['score'] + damped['presentation_reward_removed'] == pytest.approx(full['score'])
    assert damped['rubric_rule_passes'] == verdict
    assert damped['presentation_evolution_anchor_accounted'] is True


@pytest.mark.parametrize('arm, floor', [('legacy', 0.), ('bounded_residual', .25)])
def test_online_dispatch_actual_reward_manager_and_advantage_metadata(monkeypatch, tmp_path, arm, floor):
    from verl.utils.reward_score import rubric_judge_save_evolve_rubric as ev
    from verl.workers.reward_manager.batch import BatchRewardManager
    for key,value in dict(RUBRIC_EVOLVE='1', RUBRIC_EVOLVE_START_EPOCH='1',
        RUBRIC_EVOLVE_REWARD_MODE=arm, RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR=str(floor),
        RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE='0', RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE='0',
        RUBRIC_EVOLVE_GROUP_RANK_GUARD='0', RUBRIC_TARGETED_REFUSAL_ENABLE='0',
        RUBRIC_PRESENTATION_DAMPING_ENABLE='1', RUBRIC_PRESENTATION_REWARD_FACTOR='.2',
        GLOBAL_REASONING_RULE_ENABLE='0', REWARD_MODE='rubric_safe_helpful_balanced',
        RUBRIC_FORMAT_WEIGHT='.1', RUBRIC_EVOLVE_SAVE_DIR=str(tmp_path)).items():
        monkeypatch.setenv(key,value)
    monkeypatch.setattr(ev, '_write_runtime_status', lambda **kwargs: None)
    monkeypatch.setattr(ev, '_evolve_api_config', lambda: ('not-called', '', 'writer-not-called'))
    monkeypatch.setattr(ev, '_evolve_generation_due', lambda: False)
    monkeypatch.setattr(ev, '_get_evolved_cache', lambda epoch: {'benign': {'outcome_useful_rule':['Extra task fact.']}})
    class RecordedScores:
        def judge_batch(self, prompts, responses, rules, infos):
            return [dict({k:[int(text != STYLE or answer == 'A') for text in values]
                          for k,values in rule.all_rules.items()}, _parse_valid=True)
                    for rule,answer in zip(rules,responses)]
    monkeypatch.setattr(rj, '_get_judge_backend', lambda: RecordedScores())
    original = dict(index=0, uid='same', prompt='Question', reward_profile='benign',
        outcome_safety_rule=['No harm'], outcome_useful_rule=['Completes task.', STYLE],
        reasoning_process_rule=['Reason about task.'], refusal_rule=['Unnecessary refusal'])
    class Tokenizer:
        def decode(self, ids, **kwargs):
            return {1:'Question',2:'A',3:'B'}[int(ids[0])]
    class Batch:
        batch = dict(prompts=torch.tensor([[1],[1]]), responses=torch.tensor([[2],[3]]),
                     attention_mask=torch.ones(2,2, dtype=torch.long))
        non_tensor_batch = dict(data_source=['benign','benign'], uid=np.array(['same','same']),
                                extra_info=[deepcopy(original),deepcopy(original)])
        meta_info = {}
        def __len__(self): return 2
    batch = Batch()
    result = BatchRewardManager(Tokenizer(), 0, rj.compute_score_batch)(batch, return_dict=True)
    assert len(result['reward_trace_info']) == 2
    for key in (pr.COMPONENT, pr.SCORED, pr.CONTENT):
        assert len(result['reward_extra_info'][key]) == 2
    batch.batch.update(token_level_scores=result['reward_tensor'],
                       token_level_rewards=result['reward_tensor'], response_mask=torch.ones(2,1))
    batch.non_tensor_batch.update(result['reward_extra_info'])
    trainer_function()(batch, 'GRPO')
    assert np.abs(batch.non_tensor_batch['presentation_advantage_contribution']).max() <= .050001
    if arm == 'bounded_residual':
        assert all(r['presentation_evolution_anchor_accounted'] for r in result['reward_trace_info'])


def test_zero_presentation_credit_cannot_create_fp32_tie_advantages():
    result = grpo([[.3]] * 8, [0.] * 8)
    assert torch.equal(result, torch.zeros_like(result))


@pytest.mark.parametrize('effect, passed', [(0., True), (.06, False), (float('nan'), False)])
def test_real_first_batch_gate_requires_finite_bounded_actual_contribution(monkeypatch, tmp_path, effect, passed):
    from verl.utils.training_evidence import atomic_json
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / 'verl/trainer/ppo/ray_trainer.py').read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_write_training_batch_gate')
    env = dict(np=np, torch=torch, os=os, atomic_json=atomic_json,
               __file__=str(root / 'verl/trainer/ppo/ray_trainer.py'))
    exec(compile(ast.Module(body=[fn], type_ignores=[]), 'actual_first_batch_gate', 'exec'),env)
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_PRESENTATION_DAMPING_ENABLE','1')
    class Batch:
        batch = {k:torch.ones(64,1) for k in ['token_level_scores','token_level_rewards','advantages']}
        non_tensor_batch = dict(uid=[f'q{i//8}' for i in range(64)], presentation_advantage_contribution=[effect]*64)
        def __len__(self): return 64
    call = lambda: env[fn.name](Batch(), [dict(judge_parse_valid=True)]*64, 1, dict(train=3120,val=304))
    if passed:
        assert call()['passed'] is True
    else:
        with pytest.raises(ValueError, match='no optimizer update'):
            call()
        import json
        assert json.loads((tmp_path/'preflight/training_batch_gate.json').read_text())['passed'] is False
