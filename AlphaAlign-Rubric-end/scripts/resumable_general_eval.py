"""Resume complete lm-eval tasks only when model/configuration/result match.

No GPU imports. Partial attempts are archived, not counted as successful results.
An interrupted task is rerun in full; completed tasks can be skipped safely.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def identity(command, model, data, harness):
    model, data, harness = map(Path, (model, data, harness))
    # Checkpoint path + shard size/mtime identify weights without repeatedly
    # streaming many GB from OSS. Changed shards/config invalidate the cache.
    weights = {str(p.relative_to(model)): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in sorted(model.rglob('*')) if p.is_file()
               and p.suffix in {'.safetensors', '.bin', '.pt'}}
    if not weights:
        raise ValueError(f'No model weight files found under {model}')
    configs = {str(p.relative_to(model)): digest(p) for p in sorted(model.glob('*'))
               if p.is_file() and p.suffix in {'.json', '.jinja', '.jinja2'}}
    sources = {}
    for root in (data, harness / 'lm_eval'):
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in {'.py', '.jsonl', '.json', '.yaml', '.yml'}:
                sources[str(path)] = digest(path)
    packages = {}
    for package in ('torch', 'vllm', 'transformers', 'datasets', 'numpy'):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    settings = {name: os.environ.get(name, '') for name in
                ('CUDA_VISIBLE_DEVICES', 'VLLM_WORKER_MULTIPROC_METHOD', 'VLLM_USE_V1')}
    if 'POLICY_ENABLE_THINKING' in os.environ:
        settings['POLICY_ENABLE_THINKING'] = os.environ['POLICY_ENABLE_THINKING']
    payload = dict(command=list(command), model=str(model.resolve()), weights=weights,
                   configs=configs, sources=sources, packages=packages, settings=settings,
                   runner=digest(__file__))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_result(path, task):
    try:
        payload = json.loads(Path(path).read_text())
        bucket = next((payload.get(name, {}).get(task) for name in ('groups', 'results')
                       if isinstance(payload.get(name, {}).get(task), dict)), {})
        metrics = [value for key, value in bucket.items()
                   if (key.startswith('acc') or key.startswith('exact_match'))
                   and 'stderr' not in key and type(value) in (int, float)
                   and math.isfinite(value) and 0 <= value <= 1]
        return bool(metrics)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def run(command, model, data, harness, output, task):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = identity(command, model, data, harness)
    marker_dir = output.parent / '.general_eval_completion'
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f'{task}.json'
    try:
        cached = json.loads(marker.read_text())
        result = output / cached['result']
        if (cached['identity'] == key and result.is_file()
                and digest(result) == cached['sha256'] and valid_result(result, task)):
            print(f'[RESUME] Verified complete task={task}; skipping inference.', flush=True)
            return 0
    except (OSError, ValueError, KeyError, TypeError):
        pass
    # Keep earlier evidence outside the task's result tree so the summarizer
    # cannot mistake an old/partial result for the current successful attempt.
    history = output.parent / '.general_eval_history' / task / str(time.time_ns())
    for previous in sorted(output.rglob('*.json')):
        target = history / previous.relative_to(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        previous.rename(target)
    print(f'[RESUME] Running task={task}; completed marker will be written only on success.', flush=True)
    code = subprocess.run(command, check=False).returncode
    if code:
        return code if code > 0 else 128 - code
    candidates = sorted(output.rglob('*.json'), key=lambda p: p.stat().st_mtime_ns)
    if not candidates or not valid_result(candidates[-1], task):
        raise RuntimeError(f'Process exited successfully but no complete result exists for {task}')
    result = candidates[-1]
    temporary = marker.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(dict(identity=key, result=str(result.relative_to(output)),
                                         sha256=digest(result), completed_ns=time.time_ns()), sort_keys=True) + '\n')
    temporary.replace(marker)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'data', 'harness', 'output', 'task'):
        parser.add_argument(f'--{name}', required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('A command after -- is required')
    raise SystemExit(run(command, args.model, args.data, args.harness, args.output, args.task))


if __name__ == '__main__':
    main()
