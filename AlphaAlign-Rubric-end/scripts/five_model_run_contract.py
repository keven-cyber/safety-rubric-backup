"""Seal intended cross-model budget before entering a native method wrapper."""
import argparse
import json
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--key', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--method', choices=['alphaalign', 'nspo', 'p1'], required=True)
    args = p.parse_args()
    models = json.loads(Path(__file__).with_name('cross_model_twelve_models.json').read_text())
    contract = models[args.model]
    expected = dict(TRAIN_TOTAL_EPOCHS='3', TRAIN_SAVE_FREQ='30', TRAIN_TEST_FREQ='30',
                    TRAIN_BATCH_SIZE='8', ROLLOUT_N='8', ACTOR_LR='1e-6',
                    TRAIN_MAX_PROMPT_LENGTH='1536', TRAIN_MAX_RESPONSE_LENGTH='2048',
                    TRAIN_RESUME_MODE='disable', RESUME_TRAINING='0',
                    BASELINE_GENERAL_EVAL_SCOPE='selected', BASELINE_REQUIRED_FINAL_STEPS='900,990', TRAIN_TOTAL_TRAINING_STEPS='990', FINAL_EVAL_EXPECTED_LAST_STEP='990', FINAL_SAFETY_EVAL_STEPS='900,best,990')
    if args.method != 'p1':
        expected.update(BASELINE_LOG_VAL_GENERATIONS='0', BASELINE_DURABLE_TRACKING_ENABLE='1',
                        BASELINE_VALIDATION_ARCHIVE='1',
                        BASELINE_FIRST_BATCH_GATE='1', BASELINE_VAL_BATCH_SIZE='304')
    expected.update(TRAIN_LOGGER_BACKENDS='["console","tensorboard"]')
    actual = {name: os.environ.get(name) for name in expected}
    if actual != expected or os.environ.get('POLICY_MODEL_ID') != args.model:
        raise ValueError('Cross-model budget or policy overwritten')
    thinking = os.environ.get('POLICY_ENABLE_THINKING')
    if thinking != (None if contract['enable_thinking'] is None else 'false'):
        raise ValueError('Policy thinking mode differs from human choice')
    summary = Path(os.environ['PRIMUS_TENSORBOARD_LOG_DIR']).resolve()
    expected_logdir = summary / 'five_model_evidence990' / args.key / (os.environ['PRIMUS_JOB_UUID'] + '_' + os.environ['PRIMUS_JOB_SUBMIT_TIMESTAMP'])
    if Path(os.environ['TENSORBOARD_DIR']).resolve() != expected_logdir:
        raise ValueError('TensorBoard directory does not match this actual job attempt')
    from agentic_eight import environment
    payload = dict(key=args.key, method=args.method, policy_model_id=args.model,
                   source_contract=contract, expected=expected, actual=actual,
                   P1_configuration=environment('p1_evidence990') if args.method == 'p1' else None,
                   final_numbered_steps=[900, 990], native_best_only_if_available=True,
                   full_final_counts=dict(safety=1890, XSTest=250, MMLU=14042, MMLU_subjects=57, GSM8K=1319),
                   optional_wandb_validation_preview_count=0 if args.method != 'p1' else None,
                   mandatory_full_native304_archive=args.method != 'p1',
                   logging=dict(backends=['console', 'tensorboard'], tensorboard_directory=str(expected_logdir),
                                wandb_enabled=False, strict_original_JSON_archives_preserved=True),
                   fresh_initial_policy=True, native_method_reward_template_nullspace_preserved=True,
                   first_batch_gate='native_P1' if args.method == 'p1' else 'baseline_immutable64',
                   real_GPU_training_or_model_execution_verified=False)
    path = Path(os.environ['CROSS_MODEL_RUN_DIR']) / 'run_contract.json'
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    main()
