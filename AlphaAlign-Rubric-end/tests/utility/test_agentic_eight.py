import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
import torch

from scripts import agentic_eight as arms
from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage
from verl.utils import targeted_refusal as targeted

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('arm', arms.ARMS)
def test_actual_launcher_chain_preserves_declared_config(tmp_path, arm):
    data = tmp_path / 'data'
    data.mkdir()
    (data / 'manifest.json').write_text(json.dumps(dict(
        version='updated_training_rubric_dataset_independent_v4_targeted_hard_negative',
        training_rows=3120, validation_rows=304)))
    for split in ['train', 'val']:
        (data / split).mkdir()
        for name in ['benign_sampled.parquet', 'sensitive_sampled.parquet',
                     'harm_sampled.parquet', 'adaptive_harm.parquet']:
            (data / split / name).write_bytes(b'metadata-only launcher test, not real parquet')
    auth = tmp_path / 'test-auth.json'
    auth.write_text('{}')
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    stub = bin_dir / 'bash'
    stub.write_text('''#!/usr/bin/python3
import os,sys,json
from pathlib import Path
if len(sys.argv)>1 and Path(sys.argv[1]).name=='train_val.sh':
 sys.path.insert(0,os.environ['TEST_ROOT'])
 from scripts.agentic_eight import validation_record
 v=validation_record()
 assert v['settings_match']
 keys=['TRAIN_LEARNING_RATE','TRAIN_TOTAL_EPOCHS','TRAIN_SAVE_FREQ','TRAIN_TEST_FREQ',
 'TRAIN_RESUME_MODE','RUBRIC_TARGETED_REFUSAL_ADV_CAP','RUBRIC_PRESENTATION_ADV_CAP',
 'RUBRIC_PRESENTATION_REWARD_FACTOR','RUBRIC_FIRST_BATCH_SNAPSHOT_ENABLE',
 'RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE','JUDGE_ENABLE_PREFIX_CACHING',
 'RUBRIC_JUDGE_MAX_NEW_TOKENS','RUBRIC_JUDGE_REPAIR_MAX_TOKENS',
 'RUBRIC_JUDGE_DROP_FAILED','RUBRIC_VALIDATION_QUARANTINE_ENABLE',
 'RUBRIC_JUDGE_API_CONCURRENCY','FINAL_SAFETY_EVAL_STEPS']
 v['invariants']={k:os.environ[k] for k in keys}
 Path(os.environ['TEST_CAPTURE']).write_text(json.dumps(v));sys.exit(0)
os.execv('/bin/bash',['/bin/bash']+sys.argv[1:])
''')
    stub.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ['PATH'],
        FILE_PATH=str(data), PRIMUS_TRAIN_DATASET_DIR=str(data),
        PRIMUS_SAVE_CHECKPOINT_DIR=str(tmp_path / 'checkpoints'),
        PRIMUS_OSS_WORKSPACE_DIR=str(tmp_path / 'workspace'),
        PRIMUS_JOB_SUBMIT_TIMESTAMP='local-arm-test', RUBRIC_EVOLVE_API_AUTH_FILE=str(auth),
        TEST_ROOT=str(ROOT), TEST_CAPTURE=str(tmp_path / 'captured.json'))
    env.pop('RUBRIC_RECOVERY_SOURCE_CHECKPOINT', None)
    # These inherited values must be replaced by the chosen registered arm.
    env.update(TRAIN_USE_KL_LOSS='wrong', RUBRIC_TARGETED_REFUSAL_INTERVAL='999')
    subprocess.run(['/bin/bash', str(ROOT / 'scripts/train_rubricevolve_agentic_eight.sh'), arm],
                   env=env, check=True, capture_output=True, text=True, timeout=30)
    actual = json.loads((tmp_path / 'captured.json').read_text())
    assert actual['actual'] == arms.environment(arm)
    assert actual['invariants'] == dict(
        TRAIN_LEARNING_RATE='1e-6', TRAIN_TOTAL_EPOCHS='3', TRAIN_SAVE_FREQ='30', TRAIN_TEST_FREQ='30',
        TRAIN_RESUME_MODE='disable', RUBRIC_TARGETED_REFUSAL_ADV_CAP='0.05',
        RUBRIC_PRESENTATION_ADV_CAP='0.05', RUBRIC_PRESENTATION_REWARD_FACTOR='0.20',
        RUBRIC_FIRST_BATCH_SNAPSHOT_ENABLE='1', RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE='1',
        JUDGE_ENABLE_PREFIX_CACHING='1', RUBRIC_JUDGE_MAX_NEW_TOKENS='1024',
        RUBRIC_JUDGE_REPAIR_MAX_TOKENS='2048', RUBRIC_JUDGE_DROP_FAILED='1',
        RUBRIC_VALIDATION_QUARANTINE_ENABLE='1', RUBRIC_JUDGE_API_CONCURRENCY='48',
        FINAL_SAFETY_EVAL_STEPS='900,990,1080,last,best')


def test_unknown_arm_fails_before_launch():
    p = subprocess.run(['/bin/bash', str(ROOT / 'scripts/train_rubricevolve_agentic_eight.sh'),
                        'unregistered;echo unsafe'], capture_output=True, text=True)
    assert p.returncode != 0 and 'invalid choice' in p.stderr


@pytest.mark.parametrize('arm', arms.ARMS)
def test_runtime_gate_rejects_overwritten_scientific_setting(monkeypatch, arm):
    for k, v in arms.environment(arm).items():
        monkeypatch.setenv(k, v)
    assert arms.validation_record()['settings_match']
    monkeypatch.setenv('RUBRIC_TARGETED_REFUSAL_ADV_CAP', '0.10')
    with pytest.raises(ValueError, match='overwritten'):
        arms.validation_record()


def test_unregistered_existing_jobs_are_unchanged(monkeypatch):
    monkeypatch.delenv('RUBRIC_AGENTIC_EIGHT_ARM', raising=False)
    assert arms.validation_record() is None


@pytest.mark.parametrize('floor', [0.0, 0.10, 0.25])
def test_actual_grpo_keeps_presentation_and_target_caps(floor, monkeypatch):
    mask = torch.ones(8, 2, dtype=torch.float64)
    content = torch.zeros_like(mask)
    content[:, -1] = torch.tensor([.10, .11, .12, .13, .14, .15, .16, .17])
    style = torch.tensor([.02, 0., .01, 0., .02, 0., .01, 0.], dtype=torch.float64)
    scored = content.clone()
    scored[:, -1] += style
    base, _ = compute_grpo_outcome_advantage(content, mask, ['p'] * 8, std_floor=floor)
    actual, _ = compute_grpo_outcome_advantage(scored, mask, ['p'] * 8, std_floor=floor,
        presentation_reward_components=style, presentation_adv_cap=.05)
    assert torch.isfinite(actual).all() and (actual - base).abs().max() <= .050001
    monkeypatch.setenv('RUBRIC_TARGETED_REFUSAL_ADV_CAP', '.05')
    corrected = targeted.apply_advantage_correction(actual, mask, [.05] * 4 + [-.05] * 4)
    assert (corrected - actual).abs().max() <= .050001


@pytest.mark.parametrize('arm', ['p1_earlytarget', 'p3_combined', 'p4_midfloor'])
def test_early_schedule_uses_actual_calibrated_correction(monkeypatch, tmp_path, arm):
    # Reuse the existing strict original/diagnostic agreement test transport.
    import sys
    spec = importlib.util.spec_from_file_location('targeted_test_fixture',
        ROOT / 'tests/utility/test_targeted_refusal.py')
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    for k, v in arms.environment(arm).items():
        monkeypatch.setenv(k, v)
    for step, expected_calls in [(179, 0), (180, 1), (181, 0), (182, 1)]:
        out, audit, judge = module.run(monkeypatch, tmp_path, step=step)
        assert judge.calls == expected_calls
        assert [r['score'] for r in out] == [.71, -.2]
        assert [r[targeted.DELTA] for r in out] == ([.05, -.05] if expected_calls else [0., 0.])
