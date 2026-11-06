"""Opt-in remaining-stage recovery; never fit, regenerate or alter old evidence."""
import json
import math
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

from verl.utils.checkpoint_recovery import copy_file, digest, prepare, require_checkpoint
from verl.utils.frozen_validation_review import read, seal, verify_seal, actor_files, copy_checkpoint, KEY
from verl.utils.validation_recovery import write

ROLES = ('900', '990', '1080', 'last', 'best')


def copy_tree(source, target):
    source, target = Path(source), Path(target)
    if source.is_symlink() or target.exists() or target.is_relative_to(source):
        raise ValueError('Stage recovery requires a fresh physical destination')
    target.mkdir(parents=True)
    result = {}
    for p in sorted(source.rglob('*')):
        if p.is_symlink():
            raise ValueError('Stage recovery refuses links')
        if p.is_file():
            result[str(p.relative_to(source))] = copy_file(p, target / p.relative_to(source))
    return result


def check_safety(path):
    """Read the unchanged original summaries, retaining original denominators."""
    text = Path(path).read_text()
    decoder = json.JSONDecoder()
    rows = []
    expected = dict(wildjailbreak_adv=210, Jailbreaktrigger=400, WildGuardTest=329,
                    strongreject=313, advbench=520, gcg_official=47, pair_official=71)
    for hit in re.finditer(r'LlamaGuard Statistics for ([^:]+):', text):
        value, _ = decoder.raw_decode(text[text.index('{', hit.end()):])
        row = value['answer_only']
        if (hit[1] not in expected or row['total_responses'] != expected[hit[1]]
                or row['evaluation_complete'] is not True or row['invalid_judgment_count'] != 0):
            raise ValueError('Safety stage is incomplete; cannot skip it')
        rows.append(hit[1])
    hits = list(re.finditer(r'\{\s*"dataset"\s*:\s*"xstest"', text))
    if len(rows) != 7 or set(rows) != set(expected) or not hits:
        raise ValueError('Missing original seven-set safety evidence')
    xs, _ = decoder.raw_decode(text[hits[-1].start():])
    if (xs['evaluation_complete'] is not True or
            any(xs[k] != 250 for k in ('total_dataset_prompts', 'matched_prompts', 'valid_refusal_judgments')) or
            any(xs[k] != 0 for k in ('missing_prompts_in_results', 'ignored_none_judgments', 'invalid_refusal_judgments'))):
        raise ValueError('XSTest is incomplete; cannot skip it')
    return seal(path)


def check_general(path, task):
    v = read(path)
    counts = v['n-samples']
    expected, subjects = (1319, 1) if task == 'gsm8k_cot_modelscope' else (14042, 57)
    if (v['config'].get('limit') is not None or len(counts) != subjects or
            any(x['original'] != x['effective'] for x in counts.values()) or
            sum(x['effective'] for x in counts.values()) != expected):
        raise ValueError('Incomplete original general benchmark denominator')
    metric = v['results'][task]['exact_match,flexible-extract' if subjects == 1 else 'acc,none']
    if not math.isfinite(metric):
        raise ValueError('Nonfinite final metric')
    return dict(artifact=seal(path), task=task, count=expected, subjects=subjects, metric=metric,
                task_hashes=v.get('task_hashes'), exact_input_bytes_verified=False)


def general_resume():
    """Only MMLU/GSM8K after all designated safety stages are already complete."""
    run = Path(os.environ['RUBRIC_RUN_DIR'])
    source_run = Path(os.environ['RUBRIC_FINAL_RESUME_SOURCE_RUN'])
    source = Path(os.environ['RUBRIC_RECOVERY_SOURCE_CHECKPOINT'])
    root = source.parent
    job = os.environ['PRIMUS_JOB_UUID']
    if not source_run.name.startswith(job + '_') or source.name != 'global_step_1170':
        raise ValueError('Remaining-stage recovery must bind this exact completed job')
    cfg = read(source_run / 'effective_config.json')
    if cfg['trainer']['default_local_dir'] != str(root):
        raise ValueError('Source run/checkpoint root differs')
    # Same pinned verifier used before the original final merger. Reads all seals
    # and actual model bytes; no cached scores or model calls in this verification.
    from verl.utils.frozen_validation_review import selection
    chosen = selection(root / 'new_reevaluation/selected_checkpoints', ROLES, 1170)
    if read(source_run / 'eval_outputs/checkpoint_selection.json') != chosen:
        raise ValueError('Original final selector differs from the sealed review')
    original_outputs = source_run / 'eval_outputs'
    model_names = ['Qwen2.5-3B-Instruct-' + Path(x['checkpoint']).name for x in chosen['checkpoints']]
    safety = [check_safety(original_outputs / (name + '_summary.txt')) for name in model_names]
    # An already attempted/completed general task cannot be silently re-run.
    if any(original_outputs.glob('*_general_scores.txt')) or any(original_outputs.glob('**/results_*.json')):
        raise ValueError('Existing general results require explicit per-task continuation review')
    destination = Path(os.environ['PRIMUS_SAVE_CHECKPOINT_DIR']) / ('presentation_v8_final_general_' +
                         os.environ['PRIMUS_JOB_SUBMIT_TIMESTAMP'])
    prepare(source, destination, Path(os.environ['PRIMUS_SAVE_CHECKPOINT_DIR']))
    model_copies = {}
    for name in model_names:
        src = root / 'models/Qwen' / name
        config = read(src / 'config.json')
        dtypes = [config[k] for k in ('dtype', 'torch_dtype') if k in config]
        if not dtypes or any(value != 'bfloat16' for value in dtypes) or not list(src.glob('*.safetensors')):
            raise ValueError('Missing original BF16 merged model')
        model_copies[name] = copy_tree(src, destination / 'models/Qwen' / name)
    write(run / 'general_resume_ready.json', dict(status='ready', optimizer_updates=0,
        original_safety=safety, original_selector=seal(original_outputs / 'checkpoint_selection.json'),
        source_checkpoint=str(source), copy_root=str(destination), model_copies=model_copies,
        new_generations_in_validation=0, repeated_safety_evaluations=0,
        historical_unknowns_unchanged=True))
    out = run / 'eval_outputs'
    out.mkdir(exist_ok=True)
    write(out / 'checkpoint_selection.json', chosen)
    script = Path(__file__).resolve().parents[2] / 'Evaluation/lm_eval_last3_general.sh'
    proofs = []
    for i, name in enumerate(model_names):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7', EVAL_OUTPUT_DIR=str(out),
                   LM_EVAL_DP_SIZE='1', LM_EVAL_TP_SIZE='1', LM_EVAL_EXTRA_MODEL_ARGS='max_model_len=4096',
                   SKIP_LM_EVAL_INSTALL=str(int(i > 0)), SKIP_LM_EVAL_DATA_PREPARE=str(int(i > 0)))
        subprocess.run(['bash', str(script), '--ckpt-path', str(destination), '--models-dir',
                        str(destination / 'models/Qwen'), '--glob', name, '--num-models', '1',
                        '--tasks', 'gsm8k_cot_modelscope,mmlu_modelscope', '--backend', 'vllm',
                        '--device', 'cuda:0'], env=env, check=True)
        for task in ('gsm8k_cot_modelscope', 'mmlu_modelscope'):
            files = list((out / name / task).glob('**/results_*.json'))
            if len(files) != 1:
                raise ValueError('Missing or ambiguous original general result')
            proofs.append(check_general(files[0], task))
    for name, items in model_copies.items():
        for rel, item in items.items():
            if digest(destination / 'models/Qwen' / name / rel) != item['sha256']:
                raise ValueError('Evaluation copy mutated')
    for item in safety:
        verify_seal(item)
    write(run / 'general_resume_completed.json', dict(status='complete', optimizer_updates=0,
          requested_roles=list(ROLES), original_safety=safety, raw_general_results=proofs,
          source_ready=seal(run / 'general_resume_ready.json'), historical_unknowns_unchanged=True))


def replay_rank(records):
    if [r['step'] for r in records] != list(range(30, 1171, 30)):
        raise ValueError('Original chronological validation coverage must include all39 scheduled steps')
    if any(not math.isfinite(r['metrics'][KEY]) for r in records):
        raise ValueError('Unknown/nonfinite validation cannot rank best')
    best = records[0]
    for record in records[1:]:
        if record['metrics'][KEY] > best['metrics'][KEY]:
            best = record
    return best


def verify_replay_scores(result):
    """Use the existing strict trace verifier without mislabelling old generations."""
    from verl.utils.frozen_validation_review import verify_result, LABEL
    if result.get('label') != 'original_validation_replay' or result.get('new_generation_calls') != 0:
        raise ValueError('Replay cannot reconstruct or regenerate old responses')
    for item in [result['original_event'], result['original_generation'], result['rescored_snapshot']] + result['frozen_policy_files']:
        verify_seal(item)
    if read(result['original_generation']['path'])['snapshot'] != read(result['rescored_snapshot']['path'])['snapshot']:
        raise ValueError('Replay changed original304 tokens/text/masks/metadata')
    verify_result(dict(result, label=LABEL, original_generations=[result['original_generation']]))


def validation_replay(trainer):
    """Feed preserved original generations through the untouched real validator."""
    import torch
    from verl import DataProto
    source_run = Path(os.environ['RUBRIC_VALIDATION_REPLAY_SOURCE_RUN'])
    source = Path(os.environ['RUBRIC_RECOVERY_SOURCE_CHECKPOINT'])
    root, run = Path(trainer.config.trainer.default_local_dir), Path(os.environ['RUBRIC_RUN_DIR'])
    if (source.name != 'global_step_1170' or not source_run.name.startswith(os.environ['PRIMUS_JOB_UUID'] + '_')
            or read(source_run / 'effective_config.json')['trainer']['default_local_dir'] != str(source.parent)
            or len(trainer.val_dataset) != 304 or trainer.config.actor_rollout_ref.rollout.val_kwargs.n != 1
            or trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample
            or os.getenv('RUBRIC_FROZEN_REVIEW_ENABLE', '0') != '0'):
        raise ValueError('Original validation replay requires exact completed source and unchanged304 greedy protocol')
    report = read(root / 'checkpoint_recovery.json')
    if report['status'] != 'ready' or report['source_checkpoint'] != str(source) or report['restored_step'] != 1170:
        raise ValueError('Original replay requires complete physical1170 preservation first')
    if (source.parent / 'scoring_protocol_transition.json').exists() or list((source.parent / 'protocol_history').glob('*.json')):
        raise ValueError('Original replay cannot resolve a scoring protocol transition')
    pending = list((source.parent / 'validation_pending').glob('*.json'))
    if len(pending) != 1:
        raise ValueError('This continuation requires exactly one fully bound original validation gap')
    event_path = pending[0]
    event = read(event_path)
    step = event['global_step']
    frozen = source.parent / ('global_step_' + str(step))
    generation = Path(event['generated_batch'])
    if (event['status'] != 'incomplete' or event['aggregate_reward'] is not None or event['best_eligible'] is not False
            or event['checkpoint'] != str(frozen) or event['batch_index'] != 1
            or event_path.read_bytes() != (root / 'validation_pending' / event_path.name).read_bytes()):
        raise ValueError('Unverified original gap binding')
    require_checkpoint(frozen)
    original_record = read(generation)
    if (not generation.is_absolute() or generation.is_symlink() or original_record['global_step'] != step
            or original_record['batch_index'] != event['batch_index']
            or generation.parent != source_run / 'validation_pending/generations'):
        raise ValueError('Original generation is bound to a different event/run/policy')
    snapshot = original_record['snapshot']
    if len(snapshot['outputs']) != 304 or len(snapshot['inputs']) != 304:
        raise ValueError('Original complete304 generation is required')
    for name, text_key in [('prompts', 'inputs'), ('responses', 'outputs')]:
        decoded = [trainer.tokenizer.decode(torch.tensor(row, dtype=torch.long), skip_special_tokens=True) for row in snapshot['tensors'][name]]
        if decoded != snapshot[text_key]:
            raise ValueError('Original archived tokens and decoded text disagree')
    signatures = [seal(p) for p in actor_files(frozen)]
    out = run / 'original_validation_replay'
    out.mkdir(exist_ok=False)
    original_event, original_generation = seal(event_path), seal(generation)
    worker_calls, scoring = [], []
    class OriginalReplayWorker:
        world_size = 4
        def generate_sequences(self, batch):
            if worker_calls:
                raise ValueError('Original validation replay may execute once only')
            prompts = torch.tensor(snapshot['tensors']['prompts'], dtype=torch.long)
            if not torch.equal(batch.batch['input_ids'], prompts):
                raise ValueError('Current original304 question/order differs from archived prompt tokens')
            if not torch.equal(batch.batch['attention_mask'], torch.tensor(snapshot['tensors']['attention_mask'], dtype=torch.long)[:, :prompts.shape[1]]):
                raise ValueError('Original prompt masks differ')
            worker_calls.append(True)
            return DataProto.from_dict(tensors={k:torch.tensor(v, dtype=torch.long) for k,v in snapshot['tensors'].items()},
                                       meta_info=snapshot['meta_info'])
    reward = trainer.val_reward_fn
    def capture(batch, return_dict=True):
        actual = trainer._rollout_trace_json_value(dict(batch.batch.items()))
        metadata = trainer._rollout_trace_json_value(batch.non_tensor_batch)
        if actual != snapshot['tensors'] or metadata != snapshot['non_tensor_batch']:
            raise ValueError('Original replay tensor/rule/metadata alignment mismatch before any Judge score')
        if scoring:
            raise ValueError('Refuse deterministic re-scoring retry')
        value = reward(batch, return_dict=return_dict)
        scoring.append(True)
        write(out / 'scoring.json', trainer._rollout_trace_json_value(value))
        return value
    trainer.actor_rollout_wg = OriginalReplayWorker()
    trainer.val_reward_fn = capture
    trainer.global_steps = step
    trainer.config.trainer.log_val_generations = 0
    previous_environment = {k:os.environ.get(k) for k in ('RUBRIC_RUN_DIR', 'RUBRIC_INTERMEDIATE_RECORD_ENABLE', 'RUBRIC_BASE_JUDGE_TRACE_DIR')}
    os.environ['RUBRIC_RUN_DIR'] = str(out)
    os.environ['RUBRIC_INTERMEDIATE_RECORD_ENABLE'] = '1'
    os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = str(out / 'base_judge_attempts')
    try:
        metrics = trainer._rollout_trace_json_value(trainer._validate())
    finally:
        trainer.val_reward_fn = reward
        for key, value in previous_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    originals = list((out / 'validation_pending/generations').glob('*.json'))
    if len(originals) != 1:
        raise ValueError('Missing single immutable replay archive')
    result = dict(label='original_validation_replay', step=step, rows=304, metrics=metrics,
        optimizer_updates=0, new_generation_calls=0, original_records_reconstructed=False,
        original_event=original_event, original_generation=original_generation, frozen_policy_files=signatures,
        rescored_snapshot=seal(originals[0]), scoring_results=[seal(out / 'scoring.json')] if scoring else [],
        raw_judge_attempts=[seal(p) for p in sorted((out / 'base_judge_attempts').glob('*.jsonl'))])
    write(out / 'result.json', result)
    verify_replay_scores(result)  # unknown whole validation blocks completion, never partial mean
    for item in signatures:
        verify_seal(item)
    records = []
    for n in range(30, 1171, 30):
        if n == step:
            records.append(dict(step=n, metrics=metrics, evidence=seal(out / 'result.json')))
        else:
            p = source_run / 'metrics' / f'step_{n:08d}.json'
            v = read(p)
            records.append(dict(step=n, metrics=v.get('metrics', v), evidence=seal(p)))
    old = read(source.parent / 'best_checkpoint_info.json')
    original_records = [x for x in records if x['step'] != step]
    oldwinner = max(original_records, key=lambda x:x['metrics'][KEY])
    if oldwinner['step'] != old['best_metric_step'] or oldwinner['metrics'][KEY] != old['best_metric_value']:
        raise ValueError('Existing best is inconsistent with original known chronological validation')
    winner = replay_rank(records)
    selected = root / 'original_validation_replay_selected'
    selected.mkdir(exist_ok=False)
    copies = {}
    best_step = winner['step']
    from verl.utils.best_checkpoint_reference import resolve
    for n in sorted({900,990,1080,1170,best_step}):
        src = resolve(source.parent, old) if n == old['best_metric_step'] else source.parent / f'global_step_{n}'
        copies[f'global_step_{n}'] = copy_checkpoint(src, selected / f'global_step_{n}')
    info = dict(old, best_metric_step=best_step, best_metric_value=winner['metrics'][KEY],
                best_checkpoint_relative_path=f'global_step_{best_step}', checkpoint_storage='numbered_checkpoint_reference_v1',
                selection_evidence=winner['metrics'], selection_status='feasible' if winner['metrics'].get('val-core/balanced/selection_feasible')==1 else 'fallback_no_feasible_checkpoint',
                original_training_best_preserved=seal(source.parent / 'best_checkpoint_info.json'))
    write(selected / 'best_checkpoint_info.json', info)
    completion = dict(status='complete', label='original_validation_replay', optimizer_updates=0, new_generation_calls=0,
        original_pending_preserved=[seal(p) for p in pending], replay=seal(out / 'result.json'), records=records,
        original_best=seal(source.parent / 'best_checkpoint_info.json'), best_step=best_step, copies=copies)
    write(selected / 'validation_replay_completed.json', completion)
    verify_replay_selection(selected, ROLES, 1170)


def verify_replay_selection(root, requested, expected_last_step):
    root = Path(root)
    if tuple(map(str, requested)) != ROLES or expected_last_step != 1170:
        raise ValueError('Original replay cannot weaken required final checkpoint roles')
    done = read(root / 'validation_replay_completed.json')
    if done['status'] != 'complete' or done['optimizer_updates'] != 0 or done['new_generation_calls'] != 0:
        raise ValueError('Original replay completion missing')
    for item in done['original_pending_preserved'] + [done['replay'], done['original_best']]:
        verify_seal(item)
    verify_replay_scores(read(done['replay']['path']))
    for record in done['records']:
        verify_seal(record['evidence'])
        evidence = read(record['evidence']['path'])
        if record['metrics'] != evidence.get('metrics', evidence):
            raise ValueError('Chronological metric differs from its sealed original evidence')
    best = replay_rank(done['records'])
    info = read(root / 'best_checkpoint_info.json')
    if done['best_step'] != best['step'] or info['best_metric_step'] != best['step'] or info['best_metric_value'] != best['metrics'][KEY]:
        raise ValueError('Recovered chronological best mismatch')
    if set(done['copies']) != {f'global_step_{n}' for n in {900,990,1080,1170,best['step']}}:
        raise ValueError('Recovered final checkpoint role missing')
    for name, files in done['copies'].items():
        for rel, item in files.items():
            p = root / name / rel
            if p.is_symlink() or p.stat().st_size != item['bytes'] or digest(p) != item['sha256']:
                raise ValueError('Recovered selected policy copy changed')
    rows = [dict(step=n, checkpoint=str(root / f'global_step_{n}'), roles=[r for r in ROLES if
            (r == 'best' and n == best['step']) or (r == 'last' and n == 1170) or r == str(n)])
            for n in sorted({900,990,1080,1170,best['step']})]
    return dict(requested=list(ROLES), last_step=1170, best_metric=info, unique_checkpoint_count=len(rows),
                checkpoints=rows, label='original_validation_replay', historical_pending_files_preserved=True)


if __name__ == '__main__':
    general_resume()
