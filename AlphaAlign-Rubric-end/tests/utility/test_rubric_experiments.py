import ast
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from scripts.rubric_experiments import ARMS, configuration, check_task_config
from scripts.safety_data_pipeline import rubric_review_prompt

ROOT = Path(__file__).resolve().parents[2]


def test_eight_distinct_controlled_arms():
    configs = [configuration(k) for k in ARMS]
    assert len(configs) == 8
    assert sum(c['api_revision'] for c in configs) == 3
    signatures = [tuple(c['environment'][k] for k in (
        'RUBRIC_REVISION_MODE', 'TRAIN_LEARNING_RATE', 'TRAIN_USE_KL_LOSS',
        'RUBRIC_CRITICAL_FAILURE_GATE')) for c in configs]
    assert len(set(signatures)) == 8
    for c in configs:
        assert c['resource_pool'] == 'wings-L20Z-T04-agentic'
        assert c['final_checkpoints'] == [900, 990, 'best']
        assert c['environment']['RUBRIC_GUARD_ENABLE'] == '0'
        assert not any('SEED' in k for k in c['environment'])


@pytest.mark.parametrize('arm', ARMS)
def test_dry_run_no_credentials_or_gpu(arm):
    result = subprocess.run(['bash', str(ROOT / 'scripts/train_rubric_experiment.sh'), arm,
                             '--dry-run'], text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == configuration(arm)


def test_invalid_arm_rejected():
    result = subprocess.run(['bash', str(ROOT / 'scripts/train_rubric_experiment.sh'), 'E8',
                             '--dry-run'], capture_output=True)
    assert result.returncode != 0


@pytest.mark.parametrize('config', [{}, {'rubric_eval': {'enabled': False}}])
def test_training_task_config_accepted(tmp_path, config):
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(config))
    check_task_config(path)


@pytest.mark.parametrize('config', [{'rubric_eval': {'enabled': True}},
                                   {'rubric_eval': {'enabled': 'false'}}, []])
def test_eval_or_malformed_task_config_rejected(tmp_path, config):
    path = tmp_path / 'task.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        check_task_config(path)


@pytest.mark.parametrize('arm', ARMS)
def test_launcher_routes_actual_environment(tmp_path, arm):
    # Execute the real launcher while replacing only heavyweight downstream entrypoints.
    scripts = tmp_path / 'scripts'
    scripts.mkdir()
    for name in ('train_rubric_experiment.sh', 'rubric_experiments.py'):
        shutil.copy2(ROOT / 'scripts' / name, scripts / name)
    data = tmp_path / 'Evaluation' / configuration(arm)['source_dataset']
    data.mkdir(parents=True)
    (data / 'manifest.json').write_text('{}')
    run = tmp_path / 'run'
    run.mkdir()
    (scripts / 'rubric_evidence_env.sh').write_text(
        f'export RUBRIC_RUN_DIR="{run}" RUBRIC_RUN_ID=test-run RUBRIC_EVIDENCE_INITIALIZED=1\n')
    dump = "python3 -c 'import os,json; print(json.dumps(dict(os.environ)))'\n"
    (scripts / 'train_api_rubric.sh').write_text('echo API\n' + dump)
    (tmp_path / 'train_val.sh').write_text('echo OLD\n' + dump)
    credential = tmp_path / 'dummy-config'
    credential.write_text('')  # Never load an actual credential during tests.
    env = dict(os.environ, SAFETY_PIPELINE_ENV_FILE=str(credential),
               TRAIN_LEARNING_RATE='999', RUBRIC_CRITICAL_FAILURE_GATE='1',
               RUBRIC_EVIDENCE_INITIALIZED='0')
    result = subprocess.run(['bash', str(scripts / 'train_rubric_experiment.sh'), arm],
                            text=True, capture_output=True, check=True, env=env)
    route, encoded = result.stdout.strip().splitlines()
    config = configuration(arm)
    assert route == ('API' if config['api_revision'] else 'OLD')
    effective = json.loads(encoded)
    for key, value in config['environment'].items():
        assert effective[key] == value
    assert json.loads((run / 'experiment.json').read_text()) == config
    if not config['api_revision']:
        assert effective['CKPT_PATH'].endswith(f'/{arm}/test-run')
        assert effective['FILE_PATH'] == str(data)


def test_api_arm_fails_before_start_without_config(tmp_path):
    env = dict(os.environ, SAFETY_PIPELINE_ENV_FILE=str(tmp_path / 'missing'))
    result = subprocess.run(['bash', str(ROOT / 'scripts/train_rubric_experiment.sh'), 'E1'],
                            text=True, capture_output=True, env=env)
    assert result.returncode != 0
    assert 'API config not found' in result.stderr


def test_repo_auth_default_and_external_override():
    for name in ('train_rubric_experiment.sh', 'train_api_rubric.sh'):
        script = (ROOT / 'scripts' / name).read_text()
        assert '${SAFETY_PIPELINE_ENV_FILE:-${PROJECT_ROOT}/configs/primus_api_auth.json}' in script


def reward_function():
    # Test actual reward function without importing CUDA, vLLM or HTTP clients.
    path = ROOT / 'verl/utils/reward_score/rubric_judge.py'
    tree = ast.parse(path.read_text())
    names = {'_compute_balanced_rubric_score', '_balanced_reward_profile'}
    subset = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name in names], type_ignores=[])
    scope = dict(os=os, Any=Any, Dict=Dict, Tuple=Tuple, _normalize_extra_info=lambda x: x,
                 _safe_str=str, _float_env=lambda k, default: float(os.getenv(k, default)))
    exec(compile(subset, str(path), 'exec'), scope)
    return scope['_compute_balanced_rubric_score']


@pytest.mark.parametrize('profile', ['benign', 'sensitive_benign', 'ordinary_harm', 'adaptive_harm'])
@pytest.mark.parametrize('critical', [0.0, 0.5, 1.0])
def test_critical_gate_scope_and_default(monkeypatch, profile, critical):
    score = reward_function()
    categories = dict(outcome_safety_rule=1, outcome_useful_rule=1,
                      reasoning_process_rule=1, refusal_rule=1)
    info = dict(reward_profile=profile, critical_safety_rule_count=3)
    judge = dict(outcome_safety_rule=[critical, 1, 1])
    monkeypatch.delenv('RUBRIC_CRITICAL_FAILURE_GATE', raising=False)
    original = score(categories, judge, info, False)[0]
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', '0')
    assert score(categories, judge, info, False)[0] == original
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', '1')
    changed = score(categories, judge, info, False)[0]
    if profile in ('ordinary_harm', 'adaptive_harm') and critical < 1:
        assert changed == -1
    else:
        assert changed == original


def test_critical_gate_missing_scores_fails(monkeypatch):
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', '1')
    with pytest.raises(ValueError, match='complete critical'):
        reward_function()({}, {}, dict(reward_profile='ordinary_harm', critical_safety_rule_count=3), True)


def test_revision_prompts_distinct_and_default(monkeypatch):
    job = dict(profile='ordinary_harm', request='Synthetic fixture', existing_rubric={})
    prompts = []
    for mode in ('old', 'general', 'safety_semantics', 'decision_boundaries', 'combined',
                 'critical_content'):
        monkeypatch.setenv('RUBRIC_REVISION_MODE', mode)
        prompts.append(rubric_review_prompt(job))
        assert f'policy ({mode})' in prompts[-1]
    assert len(set(prompts)) == 6
    monkeypatch.setenv('RUBRIC_REVISION_MODE', 'typo')
    with pytest.raises(ValueError, match='Unknown'):
        rubric_review_prompt(job)


def test_strict_gate_catches_noncritical_and_boundary_failures(monkeypatch):
    score = reward_function()
    categories = dict(outcome_safety_rule=0.75, outcome_useful_rule=1,
                      reasoning_process_rule=1, refusal_rule=1)
    info = dict(reward_profile='ordinary_harm', critical_safety_rule_count=3)

    # All three critical rules pass, but a later safety rule fails. The old
    # critical-only gate cannot see it; strict mode must.
    judge = dict(outcome_safety_rule=[1, 1, 1, 0], refusal_rule=[1, 1])
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', 'critical')
    critical_score, _, critical_components = score(categories, judge, info, False)
    assert critical_score > -1
    assert critical_components['balanced_critical_gate_applied'] == 0
    monkeypatch.setenv('RUBRIC_CRITICAL_FAILURE_GATE', 'strict')
    strict_score, _, strict_components = score(categories, judge, info, False)
    assert strict_score == -1
    assert strict_components['balanced_safety_threshold_failure'] == 1

    # Complete safety passes do not compensate for a failed harmful-boundary
    # rubric under strict mode.
    categories['outcome_safety_rule'] = 1
    categories['refusal_rule'] = 0.5
    judge = dict(outcome_safety_rule=[1, 1, 1, 1], refusal_rule=[1, 0])
    strict_score, _, strict_components = score(categories, judge, info, False)
    assert strict_score == -1
    assert strict_components['balanced_boundary_threshold_failure'] == 1


def test_targeted_experiment_matrix_is_comparable_and_guard_free():
    from scripts.rubric_targeted_experiments import ARMS as targeted_arms
    from scripts.rubric_targeted_experiments import configuration as targeted_configuration

    assert set(targeted_arms) == {'A', 'B', 'C', 'D1', 'D2', 'E'}
    for arm in targeted_arms:
        config = targeted_configuration(arm)
        env = config['environment']
        assert env['TRAIN_LEARNING_RATE'] == '1e-6'
        assert env['TRAIN_USE_KL_LOSS'] == 'False'
        assert env['RUBRIC_GUARD_ENABLE'] == '0'
        assert env['FINAL_SAFETY_EVAL_STEPS'] == '900,990,best'
        assert env['RUBRIC_INTERMEDIATE_RECORD_ENABLE'] == '0'
        assert env['RUBRIC_LIGHT_DIAGNOSTICS_ENABLE'] == '1'
        assert config['seed_policy'].startswith('unchanged repository default')


def test_light_diagnostics_are_bounded_and_reuse_validation_results(tmp_path, monkeypatch):
    path = ROOT / 'verl/trainer/ppo/ray_trainer.py'
    tree = ast.parse(path.read_text())
    trainer = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                   and node.name == 'RayPPOTrainer')
    method = next(node for node in trainer.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_write_light_validation_diagnostics')
    scope = {
        'os': os,
        'np': np,
        'fingerprint': lambda value: hashlib.sha256(value.encode()).hexdigest(),
        'atomic_json': lambda target, value: Path(target).write_text(json.dumps(value)),
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), scope)

    class Dummy:
        global_steps = 900

    (tmp_path / 'validation_diagnostics').mkdir()
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path))
    monkeypatch.setenv('RUBRIC_LIGHT_DIAGNOSTICS_ENABLE', '1')
    monkeypatch.setenv('RUBRIC_LIGHT_DIAGNOSTIC_STEPS', '900')
    monkeypatch.setenv('RUBRIC_LIGHT_DIAGNOSTIC_MAX_PER_PROFILE', '1')
    traces = [
        {'balanced_safety': 1.0, 'rubric_rules': {'outcome_safety_rule': ['safe']}},
        {'balanced_safety': 0.0, 'rubric_rules': {'outcome_safety_rule': ['unsafe']}},
    ]
    scope['_write_light_validation_diagnostics'](
        Dummy(), np.array(['ordinary_harm', 'ordinary_harm']),
        ['prompt-good', 'prompt-bad'], ['answer-good', 'answer-bad'], [1.0, -1.0],
        {'balanced_safety': [1.0, 0.0], 'balanced_critical_gate_applied': [0.0, 1.0],
         'judge_parse_valid': [True, True]},
        traces,
    )
    report = json.loads((tmp_path / 'validation_diagnostics/step_00000900.json').read_text())
    assert len(report['cases']) == 1
    assert report['cases'][0]['response'] == 'answer-bad'
    assert report['interpretation'] == 'rubric_validation_diagnostics_not_external_safety_labels'


def test_ppo_override_wiring():
    ppo = (ROOT / 'examples/ppo_trainer/qwen3B-instruct-GRPO.sh').read_text()
    assert 'actor.optim.lr="${TRAIN_LEARNING_RATE:-1e-6}"' in ppo
    assert 'actor.use_kl_loss="${TRAIN_USE_KL_LOSS:-False}"' in ppo
    assert 'actor.kl_loss_coef="${TRAIN_KL_LOSS_COEF:-0.001}"' in ppo
    subprocess.run(['bash', '-n', str(ROOT / 'scripts/train_rubric_experiment.sh')], check=True)
