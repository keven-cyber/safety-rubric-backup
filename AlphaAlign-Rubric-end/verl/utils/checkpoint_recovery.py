"""Explicit failed-run recovery into a new, byte-verified checkpoint directory.

No model conversion or tensor arithmetic. Original files are read only, never
linked, pruned or replaced. Missing state, pending validation or storage errors
stop recovery; there is no silent fresh-policy fallback.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, data):
    path = Path(path)
    raw = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def require_checkpoint(path, world_size=4):
    path = Path(path)
    match = re.fullmatch(r'global_step_(?:best_)?([1-9][0-9]*)', path.name)
    if not match or not path.is_dir() or path.is_symlink():
        raise ValueError('Recovery requires an explicit ordinary checkpoint directory')
    step = int(match.group(1))
    required = ['data.pt', 'rubric_evolution/manifest.json'] + [
        f'actor/{prefix}_world_size_{world_size}_rank_{rank}.pt'
        for prefix in ('model', 'optim', 'extra_state') for rank in range(world_size)]
    for name in required:
        f = path / name
        if not f.is_file() or f.is_symlink() or f.stat().st_size == 0:
            raise ValueError('Missing complete recovery state: ' + name)
    # Inspect only the small synchronization payloads; never deserialize or
    # reencode the multi-gigabyte model/optimizer shards in this copy helper.
    import torch
    data_state = torch.load(path / 'data.pt', map_location='cpu', weights_only=False)
    if not isinstance(data_state, dict) or not data_state:
        raise ValueError('Recovery requires a nonempty data-loader state')
    for rank in range(world_size):
        extra = torch.load(path / f'actor/extra_state_world_size_{world_size}_rank_{rank}.pt',
                           map_location='cpu', weights_only=False)
        rng = extra.get('rng', {}) if isinstance(extra, dict) else {}
        if (not isinstance(extra, dict) or not isinstance(extra.get('lr_scheduler'), dict)
                or not isinstance(rng, dict) or not {'cpu', 'cuda', 'numpy', 'random'} <= rng.keys()
                or not all(torch.is_tensor(rng[k]) and rng[k].dtype == torch.uint8 for k in ('cpu', 'cuda'))
                or not isinstance(rng['numpy'], tuple) or not isinstance(rng['random'], tuple)):
            raise ValueError('Missing synchronized scheduler/RNG state for an actor rank')
    snapshot = json.loads((path / 'rubric_evolution/manifest.json').read_text())
    if (snapshot.get('schema_version') != 1 or snapshot.get('global_step') != step
            or type(snapshot.get('epoch')) is not int or snapshot['epoch'] < 1
            or not isinstance(snapshot.get('files'), dict)):
        raise ValueError('Rubric snapshot is not aligned with the policy checkpoint')
    for name, sha in snapshot['files'].items():
        f = path / 'rubric_evolution' / name
        if Path(name).name != name or not f.is_file() or digest(f) != sha:
            raise ValueError('Incomplete or corrupt original rubric state')
    for f in path.rglob('*'):
        if f.is_symlink():
            raise ValueError('Checkpoint recovery refuses links')
    return step


def copy_file(source, destination):
    before = source.stat()
    destination.parent.mkdir(parents=True, exist_ok=True)
    value = hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        for block in iter(lambda: src.read(8 * 1024 * 1024), b''):
            dst.write(block)
            value.update(block)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    sha = value.hexdigest()
    if (before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns
            or destination.stat().st_size != before.st_size or digest(destination) != sha):
        raise ValueError('Checkpoint changed during copy or copy content mismatch')
    return dict(bytes=before.st_size, sha256=sha, copied_mtime_ns=destination.stat().st_mtime_ns)


def prepare(source, destination, persistent_root, world_size=4):
    if Path(source).is_symlink() or not re.fullmatch(r'global_step_[1-9][0-9]*', Path(source).name):
        raise ValueError('Recovery requires a numbered original checkpoint, not a link or best alias')
    source, destination, persistent_root = map(lambda p: Path(p).resolve(),
                                               (source, destination, persistent_root))
    if not source.is_relative_to(persistent_root) or not destination.is_relative_to(persistent_root):
        raise ValueError('Recovery paths must stay in the current persistent checkpoint root')
    if destination.is_relative_to(source.parent) or source.parent.is_relative_to(destination) or destination.exists():
        raise ValueError('Recovery requires a new destination; existing evidence is read only')
    step = require_checkpoint(source, world_size)
    original_root = source.parent
    pending_files = sorted((original_root / 'validation_pending').glob('*.json'))
    gaps = []
    if pending_files:
        if os.getenv('RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING', '0') != '1':
            raise ValueError('Unresolved frozen validation must not be lost during checkpoint recovery')
        # Preserve unknown validations, their exact original generations and
        # frozen policies. No gap is marked resolved or allowed into best.
        for event_file in pending_files:
            event = json.loads(event_file.read_bytes())
            gap_step = event.get('global_step')
            frozen = original_root / f'global_step_{gap_step}'
            generation = Path(event.get('generated_batch', ''))
            if (event_file.is_symlink() or type(gap_step) is not int or not 0 < gap_step <= step
                    or event.get('status') != 'incomplete' or event.get('aggregate_reward') is not None
                    or event.get('best_eligible') is not False or event.get('final_selection_blocked') is not True
                                        or not generation.is_absolute() or generation.is_symlink()
                    or not generation.resolve().is_relative_to(persistent_root.parent)
                    or 'validation_pending' not in generation.parts or 'generations' not in generation.parts):
                raise ValueError('Unverified pending validation cannot be preserved')
            if event.get('checkpoint') != str(frozen):
                # A previous recovery retained the ORIGINAL immutable event.
                # Bind it to the previous copy manifest and original frozen policy;
                # copy from that unchanged original, without rewriting the event.
                previous_path = original_root / 'checkpoint_recovery.json'
                previous = json.loads(previous_path.read_text())
                origin = Path(event.get('checkpoint',''))
                original_event = origin.parent / 'validation_pending' / event_file.name
                binding = next((g for g in previous.get('preserved_validation_gaps', [])
                    if g['copied_pending'] == 'validation_pending/' + event_file.name), None)
                stored = previous.get('files', {}).get('validation_pending/' + event_file.name, {})
                if (previous.get('status') != 'ready' or not binding
                        or not origin.is_absolute() or origin.is_symlink()
                        or not origin.resolve().is_relative_to(persistent_root)
                        or origin.name != f'global_step_{gap_step}'
                        or binding['original_pending'] != str(original_event)
                        or original_event.is_symlink()
                        or original_event.read_bytes() != event_file.read_bytes()
                        or stored.get('sha256') != digest(event_file)):
                    raise ValueError('Copied immutable validation gap has no verified origin')
                frozen = origin
            require_checkpoint(frozen, world_size)
            raw_generation = json.loads(generation.read_bytes())
            snapshot = raw_generation.get('snapshot', {})
            if (raw_generation.get('global_step') != gap_step
                    or raw_generation.get('batch_index') != event.get('batch_index')
                    or not isinstance(snapshot.get('inputs'), list)
                    or not snapshot['inputs'] or len(snapshot['inputs']) != len(snapshot.get('outputs', []))
                    or not isinstance(snapshot.get('tensors'), dict)
                    or not isinstance(snapshot.get('non_tensor_batch'), dict)):
                raise ValueError('Pending validation original generation is incomplete')
            gaps.append((event_file, generation, frozen))
    best_info_path = original_root / 'best_checkpoint_info.json'
    if best_info_path.is_symlink():
        raise ValueError('Recovery refuses linked best metadata')
    if int((original_root / 'latest_checkpointed_iteration.txt').read_text().strip()) != step:
        raise ValueError('Explicit recovery source is not the last completed checkpoint')
    best = json.loads(best_info_path.read_text())
    best_step = best.get('best_metric_step')
    if (type(best_step) is not int or not 0 < best_step <= step
            or not isinstance(best.get('best_metric_value'), (float, int))
            or not math.isfinite(best['best_metric_value'])):
        raise ValueError('Missing aligned original best-checkpoint metadata')
    from verl.utils.best_checkpoint_reference import resolve
    roots = [source, resolve(original_root, best)] + [g[2] for g in gaps]
    # Numbered final-evaluation sources already reached must also survive.
    roots += [original_root / f'global_step_{s}' for s in (900, 990, 1080) if s <= step]
    roots = list(dict.fromkeys(roots))
    for path in roots:
        require_checkpoint(path, world_size)
    best_iteration = original_root / 'best_checkpointed_iteration.txt'
    if best_iteration.is_symlink() or int(best_iteration.read_text().strip()) != best_step:
        raise ValueError('Best checkpoint tracker and metadata differ')
    protocol_markers = list((original_root / 'protocol_history').glob('*.json'))
    marker = original_root / 'scoring_protocol_transition.json'
    if marker.exists():
        protocol_markers.append(marker)
    if protocol_markers and os.getenv('RUBRIC_RESUME_PRESERVE_VALIDATION_PENDING','0') != '1':
        raise ValueError('Scoring protocol review cannot disappear during recovery')
    for marker in protocol_markers:
        value=json.loads(marker.read_text())
        if marker.is_symlink() or value.get('status') != 'requires_same_protocol_review' or value.get('final_best_blocked') is not True:
            raise ValueError('Invalid scoring protocol transition evidence')
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    layouts = {}
    for root in roots:
        layout={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        if root.name in layouts and layouts[root.name] != layout:
            raise ValueError('Frozen policy aliases have different file layouts')
        layouts[root.name]=layout
        for src in sorted(root.rglob('*')):
            if src.is_file():
                relative = str(Path(root.name) / src.relative_to(root))
                if relative in files:
                    if digest(src) != files[relative]['sha256']:
                        raise ValueError('Frozen policy aliases contain different original bytes')
                else:
                    files[relative] = copy_file(src, destination / relative)
    preserved_gaps = []
    for event_file, generation, frozen in gaps:
        event_relative = 'validation_pending/' + event_file.name
        generation_relative = 'validation_pending/preserved_generations/' + event_file.name
        files[event_relative] = copy_file(event_file, destination / event_relative)
        files[generation_relative] = copy_file(generation, destination / generation_relative)
        preserved_gaps.append(dict(original_pending=str(frozen.parent / 'validation_pending' / event_file.name), source_pending=str(event_file), copied_pending=event_relative,
            original_generation=str(generation), copied_generation=generation_relative,
            frozen_checkpoint=frozen.name, unresolved=True, final_selection_blocked=True))
    for marker in protocol_markers:
        relative='protocol_history/' + digest(marker) + '.json'
        files[relative]=copy_file(marker,destination/relative)
    for src in [best_info_path, best_iteration]:
        files[src.name] = copy_file(src, destination / src.name)
    with (destination / 'latest_checkpointed_iteration.txt').open('x') as tracker:
        tracker.write(str(step) + '\n')
        tracker.flush()
        os.fsync(tracker.fileno())
    report = dict(schema_version=1, status='ready', source_checkpoint=str(source),
        destination=str(destination), restored_step=step, world_size=world_size,
        best_step=best_step, source_files_modified=False, links_used=False,
        source_and_copy_stream_sha256_equal=True, files=files,
        model_precision_changed=False, model_state_reencoded=False,
        preserved_validation_gaps=preserved_gaps,
        state_includes=['model', 'optimizer', 'scheduler', 'rng', 'data_loader', 'rubric_evolution', 'best_selection'],
        identity_note='Fresh local copy SHA comparison, not an invented original-server integrity sidecar')
    write(destination / 'checkpoint_recovery.json', report)
    return report


def copied_context(root, actual_step):
    root = Path(root).resolve()
    report_path = root / 'checkpoint_recovery.json'
    report = json.loads(report_path.read_text())
    if (report.get('schema_version') != 1 or report.get('status') != 'ready'
            or report.get('destination') != str(root) or report.get('restored_step') != actual_step
            or report.get('world_size') != 4 or report.get('links_used') is not False
            or report.get('source_files_modified') is not False
            or report.get('source_and_copy_stream_sha256_equal') is not True):
        raise ValueError('Resume source and verified checkpoint-copy manifest do not match')
    require_checkpoint(root / f'global_step_{actual_step}', 4)
    for name, item in report['files'].items():
        rel = Path(name)
        f = root / rel
        if rel.is_absolute() or '..' in rel.parts or f.is_symlink() or not f.is_file() or f.stat().st_size != item['bytes'] or f.stat().st_mtime_ns != item['copied_mtime_ns']:
            raise ValueError('Verified copied checkpoint file is missing or changed')
    return dict(status='copied', restored_step=actual_step,
                source_checkpoint=report['source_checkpoint'], checkpoint=str(root / f'global_step_{actual_step}'),
                checkpoint_root=str(root), copy_manifest=str(report_path), copy_manifest_sha256=digest(report_path),
                original_best_step=report['best_step'],
                original_best_metric_key=json.loads((root / 'best_checkpoint_info.json').read_text())['best_metric_key'],
                original_best_metric_mode=json.loads((root / 'best_checkpoint_info.json').read_text())['best_metric_mode'],
                original_best_metric_value=json.loads((root / 'best_checkpoint_info.json').read_text())['best_metric_value'])


def verified_context(context):
    if not isinstance(context, dict) or context.get('status') != 'restored':
        return False
    expected = copied_context(context['checkpoint_root'], context['restored_step'])
    return (all(context.get(k) == v for k, v in expected.items() if k != 'status')
            and context.get('actor_optimizer_scheduler_rng_load_returned') is True
            and context.get('data_loader_loaded') is True and context.get('rubric_state_loaded') is True
            and context.get('best_selection_loaded') is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--persistent-root', required=True)
    args = parser.parse_args()
    report = prepare(args.source, args.destination, args.persistent_root)
    print(json.dumps(dict(status=report['status'], restored_step=report['restored_step'],
                         source_checkpoint=report['source_checkpoint'], destination=report['destination'],
                         verified_files=len(report['files']))), flush=True)


if __name__ == '__main__':
    main()
