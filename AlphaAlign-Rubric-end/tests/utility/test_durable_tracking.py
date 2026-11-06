import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from verl.utils.tracking import Tracking, ValidationGenerationsLogger


class SDKError(RuntimeError):
    pass


@pytest.fixture
def sdk(monkeypatch):
    calls = []

    class Table:
        def __init__(self, columns, data=None):
            self.columns, self.data = columns, list(data or [])

        def add_data(self, *row):
            self.data.append(list(row))

    obj = SimpleNamespace(Table=Table)
    obj.init = lambda **kw: calls.append(('init', kw))
    obj.log = lambda *args, **kw: calls.append(('log', dict(data=args[0], **kw) if args else kw))
    obj.finish = lambda **kw: calls.append(('finish', kw))
    monkeypatch.setitem(sys.modules, 'wandb', obj)
    monkeypatch.delenv('RUBRIC_DURABLE_TRACKING_ENABLE', raising=False)
    return obj, calls


def enable(monkeypatch, tmp_path):
    root = tmp_path / 'persistent' / 'run'
    monkeypatch.setenv('RUBRIC_DURABLE_TRACKING_ENABLE', '1')
    monkeypatch.setenv('PRIMUS_OSS_WORKSPACE_DIR', str(tmp_path / 'persistent'))
    monkeypatch.setenv('RUBRIC_RUN_DIR', str(root))
    return root / 'tracking'


def records(root, kind):
    return [json.loads(p.read_text()) for p in root.glob(kind + '_*.json')]


def fail(message):
    def boom(*args, **kwargs):
        raise SDKError(message)
    return boom


def test_init_403_does_not_lose_config_or_later_metrics(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    obj.init = fail('Error uploading run: returned error 403')
    with pytest.warns(RuntimeWarning):
        log = Tracking('p', 'n', 'wandb', {'seed': 0, 'api_key': 'never-store-me'})
    log.log({'actor/grad_norm': 2.0422768592834473}, 29)
    assert records(root, 'config')[0]['payload']['config'] == {'seed': 0, 'api_key': '[REDACTED]'}
    assert records(root, 'metrics')[0]['payload']['data'] == {'actor/grad_norm': 2.0422768592834473}
    assert records(root, 'backend_error')[0]['payload']['message'].endswith('403')
    assert not calls


def test_step30_closed_transport_preserves_table_and_continues_through31(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    log = Tracking('p', 'n', 'wandb', {'total_steps': 1170})
    val = ValidationGenerationsLogger()
    samples = [('original prompt', 'original answer', -0.532)]
    original_log = obj.log
    def table_failure(*args, **kw):
        data = args[0] if args else kw.get('data', {})
        if 'val/generations' in data:
            raise SDKError('unable to perform operation on <UnixTransport closed=True>; the handler is closed')
        original_log(*args, **kw)
    obj.log = table_failure
    for step in range(1, 32):
        if step == 30:
            with pytest.warns(RuntimeWarning):
                val.log(['wandb'], samples, step)
        log.log({'actor/lr': 1e-6, 'step': step}, step)
    # A later table is still retained, without retrying the failed SDK table.
    val.log(['wandb'], [('p2', 'a2', 0.25)], 60)
    assert sorted(x['step'] for x in records(root, 'metrics')) == list(range(1, 32))
    tables = {x['step']: x['payload']['samples'] for x in records(root, 'validation_generations')}
    assert tables[30] == [list(samples[0])] and tables[60] == [['p2', 'a2', 0.25]]
    assert len(records(root, 'backend_error')) == 1
    assert len([c for c in calls if c[0] == 'log']) == 31


def test_metric_transport_failure_is_bounded_and_every_original_is_saved(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    log = Tracking('p', 'n', 'wandb')
    obj.log = fail('closed transport')
    with pytest.warns(RuntimeWarning):
        log.log({'x': 1}, 1)
    log.log({'x': 2}, 1)
    assert len(records(root, 'metrics')) == 2  # repeated step never overwrites evidence
    assert len(records(root, 'backend_error')) == 1


def test_successful_sdk_still_receives_unchanged_values(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    log = Tracking('p', 'n', 'wandb')
    values = {'reward': -0.532, 'expression_increment': 0.002}
    log.log(values, 17)
    ValidationGenerationsLogger().log(['wandb'], [('q', 'a', -0.532)], 30)
    assert [c[1] for c in calls if c[0] == 'log'][0] == {'data': values, 'step': 17}
    assert records(root, 'metrics')[0]['payload']['data'] == values
    assert len(records(root, 'validation_generations')) == 1
    assert len([c for c in calls if c[0] == 'log']) == 2


@pytest.mark.parametrize('operation', ['init', 'metric', 'table'])
def test_default_off_does_not_swallow_sdk_errors(sdk, operation):
    obj, calls = sdk
    if operation == 'init':
        obj.init = fail('403')
        with pytest.raises(SDKError):
            Tracking('p', 'n', 'wandb')
    elif operation == 'metric':
        log = Tracking('p', 'n', 'wandb')
        obj.log = fail('closed transport')
        with pytest.raises(SDKError):
            log.log({'x': 1}, 1)
    else:
        obj.log = fail('closed table transport')
        with pytest.raises(SDKError):
            ValidationGenerationsLogger().log(['wandb'], [('q', 'a', 1)], 30)


def test_disk_error_propagates_before_sdk_side_effect(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    log = Tracking('p', 'n', 'wandb')
    before = list(calls)
    monkeypatch.setattr(log._durable, 'record', lambda *a, **kw: (_ for _ in ()).throw(OSError('disk full')))
    with pytest.raises(OSError, match='disk full'):
        log.log({'x': 1}, 1)
    assert calls == before


@pytest.mark.parametrize('bad_root', ['missing', 'outside', 'workspace_itself'])
def test_archive_must_have_isolated_persistent_path(sdk, monkeypatch, tmp_path, bad_root):
    obj, calls = sdk
    enable(monkeypatch, tmp_path)
    if bad_root == 'missing':
        monkeypatch.delenv('RUBRIC_RUN_DIR')
    elif bad_root == 'outside':
        monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'ephemeral'))
    else:
        monkeypatch.setenv('RUBRIC_RUN_DIR', str(tmp_path / 'persistent'))
    with pytest.raises((KeyError, ValueError)):
        Tracking('p', 'n', 'wandb')
    assert not calls


def test_nonfinite_event_is_not_silently_repaired_or_sent(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    enable(monkeypatch, tmp_path)
    log = Tracking('p', 'n', 'wandb')
    before = list(calls)
    with pytest.raises(ValueError):
        log.log({'reward': float('nan')}, 1)
    assert calls == before


def test_actual_hydra_logger_list_and_resolved_config_are_preserved(sdk, monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    actual = OmegaConf.create(dict(trainer=dict(logger=['wandb'],total_training_steps=1170),
        actor=dict(lr=1e-6), data=dict(train=3120,val=304),seed=0))
    # Exact argument types supplied by RayPPOTrainer.fit, not a plain-list stub.
    resolved = OmegaConf.to_container(actual, resolve=True)
    log = Tracking('p', 'n', actual.trainer.logger, resolved)
    log.log({'actor/lr':1e-6,'reward':-.532},1,backend=actual.trainer.logger)
    ValidationGenerationsLogger().log(actual.trainer.logger,[('question','answer',-.532)],30)
    config = records(root,'config')[0]['payload']
    assert config['backends'] == ['wandb'] and config['config'] == resolved
    assert records(root,'metrics')[0]['payload']['backend_filter'] == ['wandb']
    assert records(root,'validation_generations')[0]['payload']['samples'] == [['question','answer',-.532]]
    assert calls[0][1]['config'] is resolved


def test_nested_omegaconf_configuration_redaction_retains_original_values(sdk, monkeypatch, tmp_path):
    from omegaconf import OmegaConf
    obj, calls = sdk
    root = enable(monkeypatch, tmp_path)
    cfg = OmegaConf.create(dict(actor=dict(lr=1e-6),api_key='never-store-this',
        references=[dict(password='also-private',model='unchanged')]))
    log = Tracking('p','n',OmegaConf.create(['wandb']),cfg)
    archived = records(root,'config')[0]['payload']['config']
    assert archived == dict(actor=dict(lr=1e-6),api_key='[REDACTED]',
        references=[dict(password='[REDACTED]',model='unchanged')])
    assert calls[0][1]['config'] is cfg and cfg.api_key == 'never-store-this'


def test_unknown_event_object_still_raises_before_sdk_delivery(sdk, monkeypatch, tmp_path):
    obj, calls = sdk
    enable(monkeypatch,tmp_path)
    log = Tracking('p','n','wandb')
    before = list(calls)
    with pytest.raises(TypeError):
        log.log({'reward':object()},1)
    assert calls == before
