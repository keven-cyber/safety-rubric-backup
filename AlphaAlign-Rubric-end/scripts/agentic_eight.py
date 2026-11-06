"""Predeclared fresh comparisons; no model calls or benchmark-label access."""
import argparse
import json
import os
import shlex


ARMS = {
    'f1_g1': ('g1', 'legacy', '0', '0', '360', '4', 'False'),
    'f3_g3': ('g3', 'bounded_residual', '0.25', '1', '360', '4', 'False'),
    'p1_earlytarget': ('g3', 'bounded_residual', '0.25', '1', '180', '2', 'False'),
    'p2_klanchor': ('g3', 'bounded_residual', '0.25', '1', '360', '4', 'True'),
    'p3_combined': ('g3', 'bounded_residual', '0.25', '1', '180', '2', 'True'),
    'p4_midfloor': ('g3', 'bounded_residual', '0.10', '1', '180', '2', 'True'),
}


def environment(arm):
    baseline, mode, floor, gradual, start, interval, kl = ARMS[arm]
    return {
        'RUBRIC_AGENTIC_EIGHT_ARM': arm,
        'RUBRIC_TARGETED_EXPERIMENT_ID': 'agentic8_' + arm,
        'TRAIN_EXPERIMENT_NAME': 'rubric-agentic8-' + arm,
        'RUBRIC_EVOLVE_REWARD_MODE': mode,
        'RUBRIC_EVOLVE_ADVANTAGE_STD_FLOOR': floor,
        'RUBRIC_EVOLVE_STD_FLOOR_SCOPE': 'global',
        'RUBRIC_EVOLVE_GRADUAL_WEIGHT_ENABLE': gradual,
        'RUBRIC_TARGETED_REFUSAL_ENABLE': '1',
        'RUBRIC_TARGETED_REFUSAL_START_STEP': start,
        'RUBRIC_TARGETED_REFUSAL_INTERVAL': interval,
        'RUBRIC_TARGETED_REFUSAL_ADV_CAP': '0.05',
        'TRAIN_USE_KL_LOSS': kl,
        'TRAIN_KL_LOSS_COEF': '0.001',
        # Hold selection, reward contrast, and variance weighting fixed.
        'RUBRIC_BENIGN_CONTRAST_ENABLE': '0',
        'RUBRIC_BENIGN_CONTRAST_AUDIT_ENABLE': '0',
        'RUBRIC_EVOLVE_VARIANCE_WEIGHT_ENABLE': '0',
        'RUBRIC_EVOLVE_GROUP_RANK_GUARD': '0',
        'RUBRIC_SELECTION_V2_ENABLE': '0',
        'RUBRIC_SELECTION_ORDER_V5_ENABLE': '0',
        'RUBRIC_SELECTION_STRICT_BEST': '0',
        'TRAIN_BEST_METRIC_KEY': 'val-core/balanced/constraint_rank_score',
    }


def validation_record():
    arm = os.getenv('RUBRIC_AGENTIC_EIGHT_ARM', '')
    if not arm:
        return None
    expected = environment(arm)
    actual = {k: os.getenv(k) for k in expected}
    if actual != expected:
        raise ValueError('Declared agentic experiment settings were overwritten')
    return dict(arm=arm, expected=expected, actual=actual, settings_match=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('arm', choices=ARMS)
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--shell', action='store_true')
    a = p.parse_args()
    if a.baseline:
        print(ARMS[a.arm][0])
    elif a.shell:
        for k, v in environment(a.arm).items():
            print('export ' + k + '=' + shlex.quote(v))
    else:
        print(json.dumps(dict(arm=a.arm, baseline=ARMS[a.arm][0],
                              environment=environment(a.arm)), indent=2))


if __name__ == '__main__':
    main()
