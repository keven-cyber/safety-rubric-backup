"""Explicitly authorized NEW validation of preserved policies, never old answers.

This opt-in evaluation path does not call fit/update_actor/save_checkpoint. It
uses the existing validation generator, reward manager and aggregation verbatim.
Old pending/transition records remain historical unknowns, byte-for-byte intact.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

from verl.utils.checkpoint_recovery import copy_file, digest
from verl.utils.validation_recovery import write

LABEL = '新复评 / new re-evaluation'
KEY = 'val-core/balanced/constraint_rank_score'
AUTHORIZATION = 'user_authorized_new_generations_20261102'


def read(path):
    return json.loads(Path(path).read_text())


def seal(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
        raise ValueError('Missing or linked review evidence: ' + str(path))
    before = path.stat()
    sha = digest(path)
    if signature(path) != [before.st_size, before.st_mtime_ns]:
        raise ValueError('Source changed while reading: ' + str(path))
    return dict(path=str(path), bytes=before.st_size, mtime_ns=before.st_mtime_ns, sha256=sha)


def signature(path):
    value = Path(path).stat()
    return [value.st_size, value.st_mtime_ns]


def verify_seal(value, full=True):
    path = Path(value['path'])
    if path.is_symlink() or signature(path) != [value['bytes'], value['mtime_ns']]:
        raise ValueError('Review evidence changed: ' + str(path))
    if full and digest(path) != value['sha256']:
        raise ValueError('Review evidence hash changed: ' + str(path))


def actor_files(path):
    """Require the actual four-rank load layout; no optimizer fallback."""
    path = Path(path)
    required = [path / 'actor' / f'{kind}_world_size_4_rank_{rank}.pt'
                for kind in ('model', 'optim', 'extra_state') for rank in range(4)]
    for item in required:
        if item.is_symlink() or not item.is_file() or item.stat().st_size == 0:
            raise ValueError('Incomplete frozen actor: ' + str(item))
    if any(p.is_symlink() for p in path.rglob('*')):
        raise ValueError('Frozen policies must not contain links')
    return required


def inventory(source_root, persistent_root, last=1170, scope="complete_history"):
    """Follow only this job's explicit recovery lineage, not directory mtimes."""
    if scope not in ("complete_history", "final_checkpoints"):
        raise ValueError("Unknown frozen review scope")
    source_root, persistent_root = Path(source_root).resolve(), Path(persistent_root).resolve()
    todo, roots, evidence = [source_root], [], []
    while todo:
        root = todo.pop()
        if root in roots:
            continue
        if not root.is_relative_to(persistent_root) or not root.is_dir() or root.is_symlink():
            raise ValueError('Frozen lineage escapes this job checkpoint mount')
        roots.append(root)
        recovery = root / 'checkpoint_recovery.json'
        if recovery.exists():
            evidence.append(seal(recovery))
            value = read(recovery)
            if value.get('status') != 'ready' or Path(value['destination']).resolve() != root:
                raise ValueError('Unverified checkpoint lineage')
            todo.append(Path(value['source_checkpoint']).parent.resolve())
        for path in sorted(root.glob('protocol_history/*.json')) + sorted(root.glob('validation_pending/*.json')):
            evidence.append(seal(path))
        for name in ('scoring_protocol_transition.json', 'best_checkpoint_info.json', 'best_checkpointed_iteration.txt', 'latest_checkpointed_iteration.txt'):
            if (root / name).exists():
                evidence.append(seal(root / name))
    targets, original_best = {}, None
    if scope == 'final_checkpoints':
        info = read(source_root / 'best_checkpoint_info.json')
        step = info.get('best_metric_step')
        if type(step) is not int or not 0 < step <= last or info.get('best_metric_key') != KEY or info.get('best_metric_mode') != 'max':
            raise ValueError('Original training best metadata is invalid')
        from verl.utils.best_checkpoint_reference import resolve
        best = resolve(source_root, info)
        actor_files(best)
        tracker = source_root / 'best_checkpointed_iteration.txt'
        if tracker.exists() and int(tracker.read_text().strip()) != step:
            raise ValueError('Original training best tracker disagrees')
        original_best = dict(step=step, checkpoint=str(best), metadata=seal(source_root / 'best_checkpoint_info.json'))
        for requested in (900, 990, 1080, last):
            path = next((root / f'global_step_{requested}' for root in roots
                         if (root / f'global_step_{requested}').is_dir()), None)
            if path is None:
                raise ValueError('Missing designated final checkpoint: ' + str(requested))
            targets[str(requested)] = str(path)
        targets['best'] = str(best)
        candidates = [(role, Path(path)) for role, path in targets.items()]
    else:
        # Ancestors first; retain distinct historical variants and exact aliases.
        candidates = [(None, path) for root in reversed(roots)
                      for path in sorted(root.glob('global_step_*'), key=lambda p: p.name)]
    policies, seen = [], {}
    for role, path in candidates:
        match = re.fullmatch(r'global_step_(?:best_)?([1-9][0-9]*)', path.name)
        if not match:
            continue
        step = int(match.group(1))
        if not 0 < step <= last or path.is_symlink() or not path.is_dir():
            raise ValueError('Unexpected frozen checkpoint in completed lineage')
        required = actor_files(path)
        models = [seal(p) for p in required if p.name.startswith('model_')]
        identity = hashlib.sha256(json.dumps([p['sha256'] for p in models]).encode()).hexdigest()
        key = (step, identity)
        if key in seen:
            seen[key]['aliases'].append(str(path))
            seen[key]['alias_model_files'].extend(models)
            if role is not None:
                seen[key]['roles'].append(role)
            continue
        item = dict(step=step, identity=identity, checkpoint=str(path), aliases=[str(path)],
                    model_files=models, alias_model_files=[], roles=[] if role is None else [role],
                    load_file_signatures={str(p): signature(p) for p in required})
        policies.append(item)
        seen[key] = item
    if scope == 'complete_history' and set(range(30, last + 1, 30)) - {p['step'] for p in policies}:
        raise ValueError('Historical checkpoint coverage is incomplete; refuse partial best selection')
    policies.sort(key=lambda p: p['step'])
    for index, item in enumerate(policies):
        item['id'] = f"{index:03d}_step_{item['step']:08d}_{item['identity'][:16]}"
    return dict(schema_version=1, label=LABEL, authorization=AUTHORIZATION,
                scope=scope, final_targets=targets, original_training_best=original_best,
                original_missing_records_not_reconstructed=True,
                original_pending_remain_historical_unknown=True,
                source_root=str(source_root), roots=list(map(str, roots)),
                expected_last_step=last, expected_rows=304, policies=policies, original_evidence=evidence)


def validate_scope(plan):
    scope = plan.get('scope', 'complete_history')
    if scope == 'complete_history':
        if set(range(30, 1171, 30)) - {p['step'] for p in plan['policies']}:
            raise ValueError('Completed review lacks a historical validation checkpoint')
        return
    if scope != 'final_checkpoints':
        raise ValueError('Unknown frozen review scope')
    roles = ('900', '990', '1080', '1170', 'best')
    if set(plan['final_targets']) != set(roles):
        raise ValueError('Designated final review roles are incomplete')
    assigned = [role for p in plan['policies'] for role in p['roles']]
    if sorted(assigned) != sorted(roles) or any(not p['roles'] for p in plan['policies']):
        raise ValueError('Designated review omitted or added a policy')
    best = plan['original_training_best']
    verify_seal(best['metadata'])
    info = read(best['metadata']['path'])
    from verl.utils.best_checkpoint_reference import resolve
    if info.get('best_metric_step') != best['step'] or best['checkpoint'] != str(resolve(plan['source_root'], info)):
        raise ValueError('Original training best binding changed')
    for p in plan['policies']:
        for role in p['roles']:
            if plan['final_targets'][role] not in p['aliases'] or p['step'] != (best['step'] if role == 'best' else int(role)):
                raise ValueError('Designated review policy/role mismatch')
    if plan['final_targets']['best'] != best['checkpoint']:
        raise ValueError('Designated review lost original training best')


def selection_best(plan, records):
    # Validate every requested result; a subset is never called a full-history best.
    ranked = best_record(records)
    if plan.get('scope', 'complete_history') == 'complete_history':
        return ranked
    policy = next(p for p in plan['policies'] if 'best' in p['roles'])
    return next(r for r in records if r['policy_id'] == policy['id'])


def best_record(records):
    if not records:
        raise ValueError('No complete new validation results')
    best = None
    for record in records:
        score = record['metrics'].get(KEY)
        if record.get('status') != 'complete' or record.get('rows') != 304 or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Incomplete/nonfinite new validation cannot rank best')
        # Original max selection: strict improvement; chronological ties keep first.
        if best is None or score > best['metrics'][KEY]:
            best = record
    return best


def verify_result(result):
    if result.get('label') != LABEL or result.get('optimizer_updates') != 0 or result.get('original_records_reconstructed') is not False:
        raise ValueError('New review provenance is missing')
    rows, score_rows = 0, 0
    for item in result['original_generations']:
        verify_seal(item)
        snapshot = read(item['path'])['snapshot']
        count = len(snapshot['outputs'])
        tensors = snapshot['tensors']
        if len(snapshot['inputs']) != count or any(len(tensors.get(k, [])) != count for k in
                ('prompts', 'responses', 'input_ids', 'attention_mask', 'position_ids')):
            raise ValueError('New validation original tokens/masks/positions are incomplete')
        if not snapshot.get('non_tensor_batch') or any(len(v) != count for v in snapshot['non_tensor_batch'].values()):
            raise ValueError('New validation metadata is incomplete')
        rows += count
    for item in result['scoring_results']:
        verify_seal(item)
        scored = read(item['path'])
        rewards, traces = scored['reward_tensor'], scored['reward_trace_info']
        if len(rewards) != len(traces) or any(not row or any(not math.isfinite(v) for v in row) for row in rewards):
            raise ValueError('New validation scoring is incomplete/nonfinite')
        if any(trace.get('judge_failed') or trace.get('judge_parse_valid') is False for trace in traces):
            raise ValueError('Unavailable Judge result cannot rank a new best')
        score_rows += len(rewards)
    if rows != 304 or score_rows != 304 or result['rows'] != 304:
        raise ValueError('New validation must score all original304 rows')
    if not result['metrics'] or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in result['metrics'].values()):
        raise ValueError('New validation metrics are incomplete/nonfinite')
    attempts = []
    for item in result['raw_judge_attempts']:
        verify_seal(item)
        attempts.extend(json.loads(line) for line in Path(item['path']).read_text().splitlines() if line.strip())
    actual = [a for a in attempts if a.get('event') == 'base_judge_attempt' and a.get('error_type') is None]
    ids = [(a.get('request_audit') or {}).get('response_id') for a in actual]
    if len(actual) < 304 or any(not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('New review lacks distinct original Judge request/return evidence')


def prompt_identity(result):
    value = hashlib.sha256()
    for item in result['original_generations']:
        snapshot = read(item['path'])['snapshot']
        original = dict(inputs=snapshot['inputs'], prompts=snapshot['tensors']['prompts'],
                        metadata=snapshot['non_tensor_batch'])
        value.update(json.dumps(original, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())
    return value.hexdigest()


def copy_checkpoint(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists() or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Final review copies require a fresh destination')
    actor_files(source)
    destination.mkdir(parents=True)
    result = {}
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Refuse linked frozen state')
        if path.is_file():
            result[str(path.relative_to(source))] = copy_file(path, destination / path.relative_to(source))
    return result


def run(trainer):
    """Generate once per distinct preserved policy; unknowns stop final selection."""
    if os.getenv('RUBRIC_FROZEN_REVIEW_AUTHORIZATION') != AUTHORIZATION:
        raise ValueError('New historical generations require explicit recorded authorization')
    if os.getenv('RUBRIC_VALIDATION_QUARANTINE_ENABLE') != '1' or os.getenv('RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE') != '1':
        raise ValueError('New review requires original archival containment and concise protocol')
    if trainer.config.trainer.del_local_ckpt_after_load or trainer.use_critic:
        raise ValueError('Frozen review cannot delete checkpoints or use a critic')
    if len(trainer.val_dataset) != 304 or trainer.config.actor_rollout_ref.rollout.val_kwargs.n != 1:
        raise ValueError('New review requires the complete unchanged 304-row validation')
    if trainer.config.actor_rollout_ref.rollout.val_kwargs.do_sample:
        raise ValueError('New review uses the original deterministic validation generation')
    from omegaconf import OmegaConf
    import torch
    import numpy as np
    import random
    root = Path(trainer.config.trainer.default_local_dir)
    review = root / 'new_reevaluation'
    review.mkdir(exist_ok=False)
    # File evidence replaces sample-table logging; no uninitialized training tracker.
    trainer.config.trainer.log_val_generations = 0
    plan = inventory(os.environ['RUBRIC_FROZEN_REVIEW_SOURCE_ROOT'], os.environ['PRIMUS_SAVE_CHECKPOINT_DIR'],
                     scope=os.getenv('RUBRIC_FROZEN_REVIEW_SCOPE', 'complete_history'))
    validate_scope(plan)
    write(review / 'inventory.json', plan)
    data_paths = trainer.config.data.val_files
    data_paths = [data_paths] if isinstance(data_paths, str) else list(data_paths)
    protocol = dict(label=LABEL, scope=plan['scope'], full_history_best_reselected=plan['scope']=='complete_history', validation_files=[seal(p) for p in data_paths],
                    rollout=OmegaConf.to_container(trainer.config.actor_rollout_ref.rollout, resolve=True),
                    seed=0, optimizer_updates=0,
                    scientific_settings={k:os.environ.get(k) for k in (
                        'REWARD_MODE','RUBRIC_JUDGE_LEGACY_COMPACT_ENABLE','RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE',
                        'RUBRIC_SELECTION_CONSTRAINT_ENABLE','RUBRIC_SELECTION_SAFETY_MIN','RUBRIC_SELECTION_REFUSAL_MAX',
                        'RUBRIC_SELECTION_USEFUL_MIN','RUBRIC_SELECTION_V2_ENABLE','RUBRIC_SELECTION_ORDER_V5_ENABLE',
                        'RUBRIC_BALANCED_BENIGN_OVER_REFUSAL_PENALTY','RUBRIC_BALANCED_BOUNDARY_OVER_REFUSAL_PENALTY')},
                    source_manifest=seal(Path(__file__).resolve().parents[2] / 'configs/presentation_v8_source_manifest.json'))
    write(review / 'protocol.json', protocol)
    records = []
    original_prompt_identity = None
    reward_fn = trainer.val_reward_fn
    initial_run = os.environ.get('RUBRIC_RUN_DIR')
    initial_base_trace = os.environ.get('RUBRIC_BASE_JUDGE_TRACE_DIR')
    try:
        for policy in plan['policies']:
            out = review / policy['id']
            out.mkdir()
            os.environ['RUBRIC_RUN_DIR'] = str(out)
            os.environ['RUBRIC_INTERMEDIATE_RECORD_ENABLE'] = '1'
            os.environ['RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT'] = policy['checkpoint']
            os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = str(out / 'base_judge_attempts')
            trainer.global_steps = policy['step']
            trainer.actor_rollout_wg.load_checkpoint(str(Path(policy['checkpoint']) / 'actor'), del_local_after_load=False)
            # This is frozen-policy validation, NOT a completed training restore.
            write(out / 'policy_loaded.json', dict(label=LABEL, policy=policy['id'], step=policy['step'],
                  load_returned=True, optimizer_updates=0, training_resume=False))
            torch.manual_seed(0)
            np.random.seed(0)
            random.seed(0)
            calls = []

            def capture(batch, return_dict=True):
                value = reward_fn(batch, return_dict=return_dict)
                file = out / 'scoring_results' / f'batch_{len(calls)+1:05d}.json'
                write(file, trainer._rollout_trace_json_value(value))
                calls.append(seal(file))
                return value

            trainer.val_reward_fn = capture
            metrics = trainer._rollout_trace_json_value(trainer._validate())
            snapshots = sorted((out / 'validation_pending/generations').glob('*.json'))
            rows = sum(len(read(p)['snapshot']['outputs']) for p in snapshots)
            result = dict(label=LABEL, policy_id=policy['id'], step=policy['step'],
                          identity=policy['identity'], status='complete' if metrics else 'unknown',
                          rows=rows, metrics=metrics, optimizer_updates=0,
                          original_records_reconstructed=False,
                          original_generations=[seal(p) for p in snapshots], scoring_results=calls,
                          raw_judge_attempts=[seal(p) for p in sorted((out / 'base_judge_attempts').glob('*.jsonl'))])
            if rows != 304:
                write(out / 'result.json', result)
                raise ValueError('Incomplete new original generation archive')
            result['prompt_identity_sha256'] = prompt_identity(result)
            if original_prompt_identity is None:
                original_prompt_identity = result['prompt_identity_sha256']
            elif original_prompt_identity != result['prompt_identity_sha256']:
                raise ValueError('Original304 question/rule/order changed between historical policies')
            # Include the fixed-input identity before sealing the immutable result.
            write(out / 'result.json', result)
            if metrics:
                verify_result(result)
            for name, value in policy['load_file_signatures'].items():
                if signature(name) != value:
                    raise ValueError('Frozen source mutated while evaluating')
            records.append(result)
            print(f"[NEW_REEVALUATION] {policy['id']} status={result['status']} rows={rows} rank={metrics.get(KEY)} optimizer_updates=0", flush=True)
    finally:
        trainer.val_reward_fn = reward_fn
        os.environ.pop('RUBRIC_VALIDATION_FROZEN_POLICY_CHECKPOINT', None)
        if initial_base_trace is None:
            os.environ.pop('RUBRIC_BASE_JUDGE_TRACE_DIR', None)
        else:
            os.environ['RUBRIC_BASE_JUDGE_TRACE_DIR'] = initial_base_trace
        if initial_run is None:
            os.environ.pop('RUBRIC_RUN_DIR', None)
        else:
            os.environ['RUBRIC_RUN_DIR'] = initial_run
    write(review / 'status.json', dict(label=LABEL, optimizer_updates=0,
        complete_policy_count=sum(r['status'] == 'complete' for r in records),
        unknown_policy_count=sum(r['status'] != 'complete' for r in records),
        original_pending_remain_historical_unknown=True))
    best = selection_best(plan, records)
    for item in plan['original_evidence']:
        verify_seal(item)
    for policy in plan['policies']:
        for item in policy['model_files'] + policy['alias_model_files']:
            verify_seal(item, full=False)
    selected = review / 'selected_checkpoints'
    selected.mkdir()
    paths, copies = [], {}
    selected_records = {}
    for role in (900, 990, 1080, 1170, 'best'):
        if plan['scope'] == 'final_checkpoints':
            policy = next(p for p in plan['policies'] if str(role) in p['roles'])
            record = next(r for r in records if r['policy_id'] == policy['id'])
        else:
            record = best if role == 'best' or role == best['step'] else next(r for r in records if r['step'] == role)
        name = f"global_step_best_{record['step']}" if record['policy_id'] == best['policy_id'] else f"global_step_{record['step']}"
        # Also provide an ordinary 1170 path for unchanged final last-step discovery.
        selected_records[name] = record['policy_id']
        if name not in copies:
            policy = next(p for p in plan['policies'] if p['id'] == record['policy_id'])
            copies[name] = copy_checkpoint(policy['checkpoint'], selected / name)
            for item in policy['model_files']:
                if copies[name]['actor/' + Path(item['path']).name]['sha256'] != item['sha256']:
                    raise ValueError('Selected copy is not the policy that was newly evaluated')
            paths.append(str(selected / name))
    if best['step'] == 1170 and 'global_step_1170' not in copies:
        copies['global_step_1170'] = copy_checkpoint(next(p['checkpoint'] for p in plan['policies'] if p['id'] == best['policy_id']), selected / 'global_step_1170')
    info = dict(best_metric_key=KEY, best_metric_mode='max', best_metric_step=best['step'],
                best_metric_value=best['metrics'][KEY], label=LABEL,
                selection_status=('feasible' if best['metrics'].get('val-core/balanced/selection_feasible') == 1 else 'fallback_no_feasible_checkpoint'),
                original_validation_gaps_remain_unknown=True,
                review_scope=plan['scope'], full_history_best_reselected=plan['scope']=='complete_history',
                selection_basis=('original_training_best_and_designated_final_checkpoints' if plan['scope']=='final_checkpoints'
                                 else 'complete_new_uniform_validation'))
    write(selected / 'best_checkpoint_info.json', info)
    completion = dict(schema_version=1, status='complete', label=LABEL, authorization=AUTHORIZATION,
                      expected_last_step=1170, optimizer_updates=0, review_scope=plan['scope'],
                      full_history_best_reselected=plan['scope']=='complete_history',
                      original_missing_records_not_reconstructed=True,
                      original_pending_remain_historical_unknown=True,
                      inventory=seal(review / 'inventory.json'), protocol=seal(review / 'protocol.json'),
                      results=[seal(review / p['id'] / 'result.json') for p in plan['policies']],
                      best_policy_id=best['policy_id'], best_metric=info,
                      selected_paths=paths, selected_records=selected_records, copies=copies)
    write(selected / 'new_reevaluation_completed.json', completion)  # ready LAST
    verify_completion(selected)


def verify_completion(root):
    root = Path(root)
    report = read(root / 'new_reevaluation_completed.json')
    if report.get('status') != 'complete' or report.get('label') != LABEL or report.get('authorization') != AUTHORIZATION or report.get('optimizer_updates') != 0:
        raise ValueError('Invalid new re-evaluation completion')
    if report.get('original_pending_remain_historical_unknown') is not True or report.get('original_missing_records_not_reconstructed') is not True:
        raise ValueError('New generation must retain original historical limitations')
    for item in [report['inventory'], report['protocol']] + report['results']:
        verify_seal(item)
    plan = read(report['inventory']['path'])
    records = [read(p['path']) for p in report['results']]
    validate_scope(plan)
    if report.get('review_scope', 'complete_history') != plan.get('scope', 'complete_history'):
        raise ValueError('Completion changed the authorized review scope')
    if [r['policy_id'] for r in records] != [p['id'] for p in plan['policies']]:
        raise ValueError('New review omitted or reordered historical policies')
    if any((r['step'],r['identity']) != (p['step'],p['identity']) for r,p in zip(records,plan['policies'])):
        raise ValueError('New review result is bound to the wrong frozen policy')
    best = selection_best(plan, records)
    scope = plan.get('scope', 'complete_history')
    if scope == 'final_checkpoints' and (report.get('full_history_best_reselected') is not False
            or report['best_metric'].get('selection_basis') != 'original_training_best_and_designated_final_checkpoints'
            or report['best_metric'].get('full_history_best_reselected') is not False):
        raise ValueError('Designated review cannot claim a full-history best')
    if best['policy_id'] != report['best_policy_id'] or report['best_metric']['best_metric_value'] != best['metrics'][KEY]:
        raise ValueError('New review best differs from chronological ranking')
    if scope == 'final_checkpoints':
        protocol = read(report['protocol']['path'])
        if protocol.get('scope') != scope or protocol.get('full_history_best_reselected') is not False:
            raise ValueError('Protocol changed the designated review scope')
        expected_records = {}
        for role in ('900', '990', '1080', '1170', 'best'):
            p = next(p for p in plan['policies'] if role in p['roles'])
            name = f"global_step_best_{p['step']}" if p['id'] == best['policy_id'] else f"global_step_{p['step']}"
            expected_records[name] = p['id']
        if report['selected_records'] != expected_records or report['best_metric']['best_metric_step'] != best['step']:
            raise ValueError('Selected copies omitted or changed a designated role')
        allowed_copies = set(expected_records)
        if best['step'] == 1170:
            allowed_copies.add('global_step_1170')
        if set(report['copies']) != allowed_copies:
            raise ValueError('Selected copies do not cover the designated checkpoints')
    for item in plan['original_evidence']:
        verify_seal(item)
    for result in records:
        verify_result(result)
        if result.get('prompt_identity_sha256') != prompt_identity(result) or result['prompt_identity_sha256'] != records[0]['prompt_identity_sha256']:
            raise ValueError('New review changed validation inputs/rules/order')
    for name, files in report['copies'].items():
        for relative, item in files.items():
            path = root / name / relative
            if path.is_symlink() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
                raise ValueError('Selected frozen copy integrity mismatch')
    return report


def selection(root, requested, expected_last_step):
    report = verify_completion(root)
    if tuple(map(str, requested)) != ('900', '990', '1080', 'last', 'best') or expected_last_step != 1170:
        raise ValueError('New review cannot weaken the common final evaluation selection')
    rows = []
    for name, policy_id in report['selected_records'].items():
        step = int(name.split('_')[-1])
        roles = [str(role) for role in requested if (role == 'best' and policy_id == report['best_policy_id']) or str(role) == str(step) or (role == 'last' and step == 1170)]
        rows.append(dict(checkpoint=str(Path(root) / name), step=step, roles=roles))
    return dict(requested=list(requested), last_step=1170, best_metric=report['best_metric'],
                unique_checkpoint_count=len(rows), checkpoints=rows, label=LABEL,
                review_scope=report.get('review_scope', 'complete_history'),
                full_history_best_reselected=report.get('full_history_best_reselected', True),
                original_validation_gaps_remain_unknown=True)
