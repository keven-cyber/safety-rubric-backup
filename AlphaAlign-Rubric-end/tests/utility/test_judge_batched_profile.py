import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import re

import pytest

from scripts import judge_replay_server as server


def test_batched_profile_restores_scheduler_without_changing_judge_or_seed(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'batched_v8')
    argv = server.command('/unchanged/model', 'qwen3_32B_instruct', 8000)
    assert argv[argv.index('--max-num-seqs') + 1] == '256'
    assert argv[argv.index('--seed') + 1] == '0'
    assert '--enforce-eager' not in argv
    assert '--no-enable-prefix-caching' in argv
    assert '--no-enable-chunked-prefill' in argv
    assert server.endpoint_limit() == 48
    assert server.PROFILE['max_num_seqs'] == 1


@pytest.mark.parametrize('wrong_key', [None, 'enforce_eager', 'max_num_seqs'])
def test_batched_profile_requires_real_server_parsed_configuration(monkeypatch, tmp_path, wrong_key):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'batched_v8')
    argv = server.command('/model', 'qwen3_32B_instruct', 8000)
    original_read = Path.read_bytes
    def read_bytes(path):
        if str(path) == '/proc/123/cmdline':
            return '\0'.join(argv).encode()
        if str(path) == '/proc/123/environ':
            return '\0'.join(f'{key}={value}' for key, value in server.ENVIRONMENT.items()).encode()
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    profile = server.selected_profile()
    if wrong_key:
        profile[wrong_key] = 'WRONG'
    log = tmp_path / 'server.log'
    log.write_text('Namespace(' + ', '.join(f'{key}={value!r}' for key, value in profile.items()) + ')')
    assert server.observed_profile(123, log)['passed'] is (wrong_key is None)


def test_unknown_serving_profile_never_falls_back_silently(monkeypatch):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'unknown')
    with pytest.raises(ValueError, match='Unknown Judge'):
        server.command('/model', 'qwen3_32B_instruct', 8000)


def test_real_grouped_23525_token_request_fits_without_shortening_original_budget(monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'grouped_context_v8')
    (tmp_path / 'config.json').write_text(json.dumps(dict(
        max_position_embeddings=40960, torch_dtype='bfloat16', rope_scaling=None)))
    argv = server.command(str(tmp_path), 'qwen3_32B_instruct', 8000)
    context = int(argv[argv.index('--max-model-len') + 1])
    batch_tokens = int(argv[argv.index('--max-num-batched-tokens') + 1])
    # Original server400:2021 prompt+21 unchanged1024-token budgets=23525.
    assert 2021 + 21 * 1024 == 23525 > 16384
    assert 23525 <= context <= 40960
    assert context == 32768 and batch_tokens == 16384
    assert argv[argv.index('--max-num-seqs') + 1] == '256'
    assert argv[argv.index('--gpu-memory-utilization') + 1] == '0.93'
    assert argv[argv.index('--seed') + 1] == '0'
    assert '--enable-chunked-prefill' in argv and '--no-enable-prefix-caching' in argv
    assert '--no-enable-chunked-prefill' not in argv
    assert '--enforce-eager' not in argv and '--rope-scaling' not in argv
    assert server.endpoint_limit() == 48
    assert server.BATCHED_PROFILE['max_num_batched_tokens'] == 16384


@pytest.mark.parametrize('config', [dict(max_position_embeddings=16384), {},
    dict(max_position_embeddings=40960, rope_scaling={'factor': 2})])
def test_grouped_capacity_never_extends_or_guesses_native_model(monkeypatch, tmp_path, config):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'grouped_context_v8')
    (tmp_path / 'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError, match='unchanged native model'):
        server.command(str(tmp_path), 'judge', 8000)


@pytest.mark.parametrize('corruption', [None, 'context', 'batch_tokens', 'parsed_context', 'chunk_flag', 'parsed_chunk'])
def test_grouped_capacity_requires_actual_argv_and_parsed_configuration(monkeypatch, tmp_path, corruption):
    monkeypatch.setenv('RUBRIC_JUDGE_SERVING_PROFILE', 'grouped_context_v8')
    (tmp_path / 'config.json').write_text(json.dumps(dict(max_position_embeddings=40960)))
    argv = server.command(str(tmp_path), 'judge', 8000)
    profile = server.selected_profile()
    if corruption == 'context': argv[argv.index('--max-model-len') + 1] = '16384'
    if corruption == 'batch_tokens': argv[argv.index('--max-num-batched-tokens') + 1] = '32768'
    if corruption == 'parsed_context': profile['max_model_len'] = 16384
    if corruption == 'chunk_flag': argv[argv.index('--enable-chunked-prefill')] = '--no-enable-chunked-prefill'
    if corruption == 'parsed_chunk': profile['enable_chunked_prefill'] = False
    original_read = Path.read_bytes
    def read_bytes(path):
        if str(path) == '/proc/123/cmdline': return '\0'.join(argv).encode()
        if str(path) == '/proc/123/environ':
            return '\0'.join(f'{key}={value}' for key,value in server.ENVIRONMENT.items()).encode()
        return original_read(path)
    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    log = tmp_path / 'server.log'
    log.write_text('Namespace(' + ', '.join(f'{k}={v!r}' for k,v in profile.items()) + ')')
    assert server.observed_profile(123, log)['passed'] is (corruption is None)


@pytest.mark.parametrize('grouped', [False, True])
def test_fresh_fast_launch_has_isolated_checkpoints_and_keeps_semantic_safeguards(tmp_path, grouped):
    root = Path(__file__).resolve().parents[2]
    for name in ('train_rubricevolve_presentation_v8_fast.sh', 'train_rubricevolve_presentation_v8.sh'):
        shutil.copy2(root / 'scripts' / name, tmp_path / name)
    keys = ('RUBRIC_JUDGE_SERVING_PROFILE', 'RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT',
            'RUBRIC_JUDGE_API_CONCURRENCY', 'TRAIN_RESUME_MODE', 'CKPT_PATH',
            'RUBRIC_JUDGE_DROP_FAILED', 'RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE',
            'RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE',
            'RUBRIC_PRESENTATION_ADV_CAP', 'RUBRIC_PRESENTATION_FIRST_BATCH_GATE')
    program = 'import json,os;print(json.dumps({key:os.environ[key] for key in ' + repr(keys) + '}))'
    import shlex
    (tmp_path / 'train_rubricevolve_targeted_refusal.sh').write_text(
        'exec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(program) + '\n')
    environment = dict(os.environ, PRIMUS_SAVE_CHECKPOINT_DIR='/existing/checkpoints',
                       PRIMUS_JOB_SUBMIT_TIMESTAMP='new-submission',
                       RUBRIC_JUDGE_GROUPED_GROUNDING_ENABLE=str(int(grouped)))
    result = subprocess.run(['bash', str(tmp_path / 'train_rubricevolve_presentation_v8_fast.sh'), 'g1'],
                            env=environment, check=True, capture_output=True, text=True)
    values = json.loads(result.stdout)
    assert values['RUBRIC_JUDGE_SERVING_PROFILE'] == ('grouped_context_v8' if grouped else 'batched_v8')
    assert values['RUBRIC_JUDGE_ENDPOINT_MAX_INFLIGHT'] == '48'
    assert values['RUBRIC_JUDGE_API_CONCURRENCY'] == '48'
    assert values['TRAIN_RESUME_MODE'] == 'disable'
    assert values['CKPT_PATH'].startswith('/existing/checkpoints/presentation_v8_batched_new-submission_')
    assert len(values['CKPT_PATH'].split('_')[-2]) == 16  # UTC startup stamp, distinct from stale platform env.
    assert values['CKPT_PATH'].split('_')[-1].isdigit()
    for key in ('RUBRIC_JUDGE_DROP_FAILED', 'RUBRIC_LITERAL_CONSTRAINT_GUARD_ENABLE',
                'RUBRIC_JUDGE_CONFLICT_QUARANTINE_ENABLE',
                'RUBRIC_PRESENTATION_FIRST_BATCH_GATE'):
        assert values[key] == '1'
    assert values['RUBRIC_PRESENTATION_ADV_CAP'] == '0.05'


@pytest.mark.parametrize('state', ['', 'Z', ' Z+'])
def test_actual_readiness_loop_rejects_exited_child_without_waiting_or_calling_model(state):
    root=Path(__file__).resolve().parents[2]
    function=re.search(r'^wait_for_api\(\) \{.*?^\}',(root/'train_val.sh').read_text(),re.M|re.S).group()
    import shlex
    script=function+'\nps() { printf "%s\\n" '+shlex.quote(state)+'; }\n'+(
        'curl() { echo UNEXPECTED_CURL; return 0; }\n'
        'sleep() { echo UNEXPECTED_SLEEP; exit 88; }\n'
        'wait_for_api 8000 "test Judge" 123\n')
    result=subprocess.run(['bash','-c',script],capture_output=True,text=True,timeout=3)
    assert result.returncode==1 and 'exited before readiness' in result.stderr
    assert 'UNEXPECTED' not in result.stdout+result.stderr


def test_actual_readiness_loop_accepts_live_ready_service_without_sleep():
    root=Path(__file__).resolve().parents[2]
    function=re.search(r'^wait_for_api\(\) \{.*?^\}',(root/'train_val.sh').read_text(),re.M|re.S).group()
    script=function+('\nps() { echo S; }\n'
        'curl() { [[ "$*" == *"--connect-timeout 2 --max-time 5"* ]] || exit 89; printf 200; }\n'
        'sleep() { exit 88; }\nwait_for_api 8000 "test Judge" 123\n')
    result=subprocess.run(['bash','-c',script],capture_output=True,text=True,timeout=3)
    assert result.returncode==0 and 'is ready' in result.stdout


