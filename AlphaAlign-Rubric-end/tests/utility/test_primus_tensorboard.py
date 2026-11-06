"""Read actual native SummaryWriter events; never initialize W&B or a model."""
import builtins
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('name', ['verl/utils/tracking.py', 'AlphaAlign/verl/utils/tracking.py',
                                  'NSPO/verl/verl/utils/tracking.py'])
def test_actual_event_write_read_without_wandb(name, monkeypatch, tmp_path):
    monkeypatch.setenv('RUBRIC_DURABLE_TRACKING_ENABLE', '0')
    monkeypatch.setenv('BASELINE_DURABLE_TRACKING_ENABLE', '1')
    monkeypatch.setenv('PRIMUS_OSS_WORKSPACE_DIR', str(tmp_path))
    monkeypatch.setenv('BASELINE_RUN_DIR', str(tmp_path/'original_run'))
    monkeypatch.setenv('PRIMUS_TENSORBOARD_LOG_DIR', str(tmp_path/'summary'))
    monkeypatch.setenv('TENSORBOARD_DIR', str(tmp_path/'summary'/'test_attempt'))
    monkeypatch.setenv('CROSS_MODEL_EXPERIMENT_KEY', 'test_key')
    monkeypatch.setenv('PRIMUS_JOB_UUID', 'test_job')
    monkeypatch.setenv('PRIMUS_JOB_SUBMIT_TIMESTAMP', 'test_attempt')
    original_import=builtins.__import__
    def guarded(module, *a, **kw):
        if module=='wandb' or module.startswith('wandb.'):
            raise AssertionError('Unexpected W&B import')
        return original_import(module, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', guarded)
    key='native_tb_'+name.split('/')[0]
    spec=importlib.util.spec_from_file_location(key, ROOT/name)
    mod=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,key,mod)
    spec.loader.exec_module(mod)
    config={'trainer': {'logger': ['tensorboard']}, 'api_key': 'private-fixture'}
    logger=mod.Tracking('project', 'experiment', ['tensorboard'], config)
    try:
        logger.log({'actor/loss': .25, 'actor/lr': 1e-6}, 30)
        mod.ValidationGenerationsLogger().log(['tensorboard'], [('input', 'original output', .5)], 30)
        # Flush is observable before finish: active jobs must publish events.
        accumulator=EventAccumulator(str(tmp_path/'summary'/'test_attempt')).Reload()
        scalars=accumulator.Scalars('actor/loss')
        assert len(scalars)==1 and scalars[0].step==30 and scalars[0].value==.25
        tags=accumulator.Tags()['tensors']
        assert 'run/config/text_summary' in tags and 'val/generations/text_summary' in tags
        raw=accumulator.Tensors('run/config/text_summary')[0].tensor_proto.string_val[0].decode()
        assert '[REDACTED]' in raw and 'private-fixture' not in raw
        preview=accumulator.Tensors('val/generations/text_summary')[0]
        assert preview.step==30 and 'original output' in preview.tensor_proto.string_val[0].decode()
        assert config['api_key']=='private-fixture'
        if not name.startswith('verl/'):
            metrics=list((tmp_path/'original_run'/'tracking').glob('metrics_*.json'))
            assert len(metrics)==1 and json.loads(metrics[0].read_text())['step']==30
    finally:
        logger.__del__();logger.logger={};logger._durable=None
