import importlib.util
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('resume_eval', ROOT / 'scripts/resumable_general_eval.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_completed_task_reused_but_changed_inputs_and_corrupt_results_rerun(tmp_path):
    model, data, harness, output = [tmp_path / p for p in ('model', 'data', 'harness', 'out/task')]
    for path in (model, data, harness / 'lm_eval', output):
        path.mkdir(parents=True, exist_ok=True)
    weights = model / 'model.safetensors'
    weights.write_bytes(b'weights')
    (model / 'config.json').write_text('{}')
    (data / 'test.jsonl').write_text('{"q": "test"}\n')
    counter, result = tmp_path / 'count', output / 'results.json'
    command = [sys.executable, '-c',
               'from pathlib import Path; import json; '
               f'p=Path({str(counter)!r}); p.write_text(str(int(p.read_text())+1) if p.exists() else "1"); '
               f'Path({str(result)!r}).write_text(json.dumps({{"results": {{"gsm8k_cot_modelscope": {{"exact_match,none": 0.5}}}}}}))']
    args = (command, model, data, harness, output, 'gsm8k_cot_modelscope')
    assert module.run(*args) == 0
    assert module.run(*args) == 0
    assert counter.read_text() == '1'
    result.write_text('corrupt')
    assert module.run(*args) == 0
    assert counter.read_text() == '2'
    weights.write_bytes(b'changed weights')
    assert module.run(*args) == 0
    assert counter.read_text() == '3'
    (data / 'test.jsonl').write_text('{"q": "changed"}\n')
    assert module.run(*args) == 0
    assert counter.read_text() == '4'
    assert list(output.parent.glob('.general_eval_history/gsm8k_cot_modelscope/*/results.json'))


def test_failed_or_empty_result_never_marks_completion(monkeypatch, tmp_path):
    output = tmp_path / 'task'
    monkeypatch.setattr(module, 'identity', lambda *args: 'same')
    args = (tmp_path, tmp_path, tmp_path, output, 'mmlu_modelscope')
    assert module.run([sys.executable, '-c', 'raise SystemExit(7)'], *args) == 7
    assert not list(tmp_path.glob('.general_eval_completion/*.json'))
    with pytest.raises(RuntimeError, match='no complete result'):
        module.run([sys.executable, '-c', 'pass'], *args)
    assert not list(tmp_path.glob('.general_eval_completion/*.json'))


def test_results_accept_aggregate_not_partial_subtask(tmp_path):
    result = tmp_path / 'result.json'
    result.write_text(json.dumps({'groups': {'mmlu_modelscope': {'acc,none': .5}}}))
    assert module.valid_result(result, 'mmlu_modelscope')
    assert not module.valid_result(result, 'gsm8k_cot_modelscope')
    result.write_text(json.dumps({'results': {'mmlu_modelscope': {'acc_stderr,none': .1}}}))
    assert not module.valid_result(result, 'mmlu_modelscope')
