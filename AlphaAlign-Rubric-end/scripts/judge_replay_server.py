"""Controlled Judge serving profile; never a replacement for real gates."""
import argparse
import json
import os
from pathlib import Path
import sys

PROFILE = dict(name='single_sequence_eager_v7', seed=0, max_num_seqs=1,
               max_num_batched_tokens=16384, enforce_eager=True,
               enable_prefix_caching=False, enable_chunked_prefill=False)
BATCHED_PROFILE = dict(name='batched_v8', seed=0, max_num_seqs=256,
                       max_num_batched_tokens=16384, enforce_eager=False,
                       enable_prefix_caching=False, enable_chunked_prefill=False)
# Full-rubric requests retain the configured output budget per predicate. The
# real21-rule replay used1024 each:2021+21504=23525 cannot fit batched_v8.
# The real32768 batch profile left only5.22GiB KV cache (8GiB required).
# The proven16384 batch profile had34624 KV tokens on the same L20Z GPUs.
# Chunked prefill permits32K context with the original16K scheduling budget;
# no RoPE override, precision change,
# omitted predicate, shorter reason or reduction of the completion budget.
GROUPED_PROFILE = dict(BATCHED_PROFILE, name='grouped_context_v8',
                       max_model_len=32768, enable_chunked_prefill=True)
LEGACY_PROFILE = dict(BATCHED_PROFILE, name='legacy_compact_v8', enable_prefix_caching=True)
# train_val.sh uses WARNING globally. This child must retain the server's INFO
# parsed-configuration evidence; process argv alone is insufficient for the gate.
ENVIRONMENT = dict(PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8',
                   NVIDIA_TF32_OVERRIDE='0', VLLM_LOGGING_LEVEL='INFO')


def selected_profile():
    name = os.getenv('RUBRIC_JUDGE_SERVING_PROFILE', PROFILE['name'])
    profiles = {profile['name']: profile for profile in (PROFILE, BATCHED_PROFILE, GROUPED_PROFILE, LEGACY_PROFILE)}
    if name not in profiles:
        raise ValueError(f'Unknown Judge serving profile: {name}')
    return dict(profiles[name])


def endpoint_limit():
    return 48 if selected_profile()['max_num_seqs'] == 256 else 1


def command(model, served, port):
    profile = selected_profile()
    if profile['name'] == GROUPED_PROFILE['name']:
        config = json.loads((Path(model) / 'config.json').read_text())
        native = config.get('max_position_embeddings')
        if (type(native) is not int or native < profile['max_model_len']
                or config.get('rope_scaling') is not None):
            raise ValueError('Grouped context must fit the unchanged native model; no RoPE extension')
    argv = [sys.executable, '-m', 'vllm.entrypoints.openai.api_server',
            '--model', model, '--served-model-name', served, '--port', str(port),
            '--tensor-parallel-size', '1', '--max-model-len', str(profile.get('max_model_len', 16384)),
            '--gpu-memory-utilization', '0.93', '--seed', '0',
            '--max-num-seqs', str(profile['max_num_seqs']), '--max-num-batched-tokens', str(profile['max_num_batched_tokens']),
            '--enable-prefix-caching' if profile['enable_prefix_caching'] else '--no-enable-prefix-caching',
            '--enable-chunked-prefill' if profile['enable_chunked_prefill'] else '--no-enable-chunked-prefill']
    if profile['enforce_eager']:
        argv.append('--enforce-eager')
    return argv


def observed_profile(pid, log_path):
    """Use the real process command AND server-parsed values; missing proof fails."""
    argv = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    env = dict(item.split('=', 1) for item in
               Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in item)
    # Retain only the three non-secret reproducibility settings.
    env = {key: env.get(key) for key in ENVIRONMENT}
    log = Path(log_path).read_text(errors='replace')
    profile = selected_profile()
    checks = {key: f'{key}={value!r}' in log for key, value in profile.items() if key != 'name'}
    chunk_flag = '--enable-chunked-prefill' if profile['enable_chunked_prefill'] else '--no-enable-chunked-prefill'
    opposite_chunk = '--no-enable-chunked-prefill' if profile['enable_chunked_prefill'] else '--enable-chunked-prefill'
    prefix_flag = '--enable-prefix-caching' if profile['enable_prefix_caching'] else '--no-enable-prefix-caching'
    opposite_prefix = '--no-enable-prefix-caching' if profile['enable_prefix_caching'] else '--enable-prefix-caching'
    checks['argv'] = (prefix_flag in argv and opposite_prefix not in argv and chunk_flag in argv
        and opposite_chunk not in argv and ('--enforce-eager' in argv) == profile['enforce_eager'])
    checks['environment'] = env == ENVIRONMENT
    checks['sequence_capacity'] = ('--max-num-seqs' in argv and
        argv[argv.index('--max-num-seqs') + 1] == str(profile['max_num_seqs']))
    if profile['name'] == GROUPED_PROFILE['name']:
        checks['context_capacity'] = ('--max-model-len' in argv and
            argv[argv.index('--max-model-len') + 1] == str(profile['max_model_len']))
        checks['batch_token_capacity'] = ('--max-num-batched-tokens' in argv and
            argv[argv.index('--max-num-batched-tokens') + 1] == str(profile['max_num_batched_tokens']))
    return dict(passed=all(checks.values()), expected=profile, checks=checks,
                observed_argv=argv, observed_environment=env,
                note='Controlled scheduling profile, not a claim that backend determinism is proved.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--served', required=True)
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--training', action='store_true')
    args = p.parse_args()
    argv = command(args.model, args.served, args.port)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(profile=selected_profile(), argv=argv,
        environment=ENVIRONMENT, validation_only=not args.training,
        purpose='training_with_mandatory_graded_gate' if args.training else 'validation'), indent=2))
    env = dict(os.environ, **ENVIRONMENT)
    os.execve(sys.executable, argv, env)


if __name__ == '__main__':
    main()
