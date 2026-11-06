from verl.utils import benign_contrast as bc


def report(flags, calibrated=True):
    return {}, dict(calibration=[{'passed': calibrated}], records=[dict(
        parse_valid=True, substantive_answer=s, coherent_answer=c, unnecessary_refusal=r)
        for s, c, r in flags])


def run(monkeypatch, flags, profiles=None, step=360, calibrated=True):
    monkeypatch.setenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '1')
    monkeypatch.setattr(bc, 'evaluate_validation', lambda *args: report(flags, calibrated))
    outputs = [{'score': .5} for _ in flags]
    bc.apply_benign_contrast(['same request'] * len(flags), ['response'] * len(flags),
                            profiles or ['benign'] * len(flags),
                            [{'uid': 'u'} for _ in flags], outputs, None, None, step)
    return outputs


def test_contrast_is_zero_sum_and_wrong_attempt_is_not_refusal(monkeypatch):
    outputs = run(monkeypatch, [(1, 1, 0), (0, 1, 1), (0, 0, 0)])
    assert [r['score'] for r in outputs] == [.6, .4, .5]
    assert sum(r['benign_contrast_delta'] for r in outputs) == 0


def test_all_bad_or_all_good_do_not_get_manufactured_signal(monkeypatch):
    for flags in ([(0, 1, 1)] * 2, [(1, 1, 0)] * 2, [(0, 0, 0)] * 2):
        assert all(r['score'] == .5 for r in run(monkeypatch, flags))


def test_harmful_and_pre_activation_rewards_unchanged(monkeypatch):
    flags = [(1, 1, 0), (0, 1, 1)]
    for kwargs in (dict(profiles=['ordinary_harm'] * 2), dict(step=359), dict(step=361)):
        assert all(r['score'] == .5 for r in run(monkeypatch, flags, **kwargs))


def test_calibration_failure_keeps_original_reward(monkeypatch):
    assert all(r['score'] == .5 for r in run(monkeypatch, [(1, 1, 0), (0, 1, 1)], calibrated=False))


def test_switch_off_does_not_call_judge(monkeypatch):
    monkeypatch.setenv('RUBRIC_BENIGN_CONTRAST_ENABLE', '0')
    monkeypatch.setattr(bc, 'evaluate_validation', lambda *a: (_ for _ in ()).throw(AssertionError()))
    outputs = [{'score': .5}] * 2
    bc.apply_benign_contrast(['p'] * 2, ['r'] * 2, ['benign'] * 2,
                            [{'uid': 'u'}] * 2, outputs, None, None, 360)
    assert all(r['score'] == .5 for r in outputs)


def test_selective_floor_preserves_untouched_grpo_groups(monkeypatch):
    import ast
    import collections
    import os
    from pathlib import Path
    from types import SimpleNamespace
    import torch
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    namespace = dict(torch=torch, np=np, defaultdict=collections.defaultdict, DataProto=object, os=os)
    names = ('GAE', 'GRPO', 'REINFORCE_PLUS_PLUS_BASELINE', 'REINFORCE_PLUS_PLUS', 'REMAX', 'RLOO')
    namespace['AdvantageEstimator'] = SimpleNamespace(**{name: name for name in names})
    for path, name in [('verl/trainer/ppo/core_algos.py', 'compute_grpo_outcome_advantage'),
                       ('verl/trainer/ppo/ray_trainer.py', 'compute_advantage')]:
        tree = ast.parse((root / path).read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), path, 'exec'), namespace)
        if name == 'compute_grpo_outcome_advantage':
            namespace['core_algos'] = SimpleNamespace(compute_grpo_outcome_advantage=namespace[name])
    class Batch:
        def __init__(self, deltas):
            self.batch = dict(token_level_rewards=torch.tensor([[.5], [.51], [.5], [.51]]),
                              response_mask=torch.ones(4, 1))
            self.non_tensor_batch = dict(uid=['a', 'a', 'b', 'b'], benign_contrast_delta=deltas)
        def __len__(self):
            return 4
    monkeypatch.setenv('RUBRIC_EVOLVE', '1')
    monkeypatch.setenv('RUBRIC_EVOLVE_REWARD_MODE', 'bounded_residual')
    monkeypatch.setenv('RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR', '.25')
    monkeypatch.setenv('RUBRIC_EVOLVE_STD_FLOOR_SCOPE', 'affected_groups')
    clean = namespace['compute_advantage'](Batch([0.0] * 4), 'GRPO').batch['advantages']
    changed = namespace['compute_advantage'](Batch([.1, -.1, None, None]), 'GRPO').batch['advantages']
    assert torch.equal(clean[2:], changed[2:])
    assert torch.all(changed[:2].abs() < clean[:2].abs())
