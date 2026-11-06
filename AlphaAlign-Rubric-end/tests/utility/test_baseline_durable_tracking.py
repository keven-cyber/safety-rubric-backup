"""Exercise both real native logger modules with failed SDK transports."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
CLOSED = "unable to perform operation on <UnixTransport closed=True reading=False 0x123>; the handler is closed"


@pytest.fixture(params=["AlphaAlign/verl/utils/tracking.py", "NSPO/verl/verl/utils/tracking.py"])
def native(request, monkeypatch, tmp_path):
    name = "baseline_tracking_" + request.param.split("/")[0]
    spec = importlib.util.spec_from_file_location(name, ROOT / request.param)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    monkeypatch.setenv("BASELINE_DURABLE_TRACKING_ENABLE", "1")
    monkeypatch.setenv("PRIMUS_OSS_WORKSPACE_DIR", str(tmp_path))
    monkeypatch.setenv("BASELINE_RUN_DIR", str(tmp_path / "unique_run"))
    calls = []
    sdk = SimpleNamespace(init=lambda **kw: calls.append(("init", kw)),
                          log=lambda **kw: calls.append(("log", kw)),
                          finish=lambda **kw: calls.append(("finish", kw)))
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    instances = []

    def make(**kwargs):
        obj = module.Tracking("project", "experiment", "wandb", config={"trainer": {}}, **kwargs)
        instances.append(obj)
        return obj

    yield SimpleNamespace(module=module, make=make, sdk=sdk, calls=calls,
                          directory=tmp_path / "unique_run" / "tracking")
    for obj in instances:
        obj.logger = {}
        obj._durable = None


def events(native, kind):
    return [json.loads(p.read_text()) for p in native.directory.glob(kind + "_*.json")]


@pytest.mark.parametrize("operation", ["init", "log", "finish"])
def test_transport_failure_after_complete_event(native, operation):
    def broken(**kwargs):
        expected = {"init": "config", "log": "metrics", "finish": "finish"}[operation]
        assert len(events(native, expected)) == 1
        raise RuntimeError(CLOSED)

    setattr(native.sdk, operation, broken)
    with pytest.warns(RuntimeWarning, match="transport failed"):
        logger = native.make()
        if operation == "log":
            logger.log({"actor/loss": 0.2}, 30)
        if operation == "finish":
            logger.__del__()
            logger.logger = {}
    failures = events(native, "backend_error")
    assert len(failures) == 1 and failures[0]["payload"]["operation"] == operation
    if operation in {"init", "log"}:
        logger.log({"actor/loss": 0.3}, 31)
        assert any(e["step"] == 31 for e in events(native, "metrics"))


def test_preview_archive_before_failure_and_no_repeat(native):
    logger = native.module.ValidationGenerationsLogger()
    samples = [("original input", "original output", 0.25)]
    calls = []

    def broken(value, step):
        assert events(native, "validation_generations")[0]["payload"]["samples"] == [list(samples[0])]
        calls.append(step)
        raise ConnectionResetError("native W&B connection reset")

    logger.log_generations_to_wandb = broken
    with pytest.warns(RuntimeWarning, match="transport failed"):
        logger.log(["wandb"], samples, 30)
    logger.log(["wandb"], samples, 60)
    assert calls == [30]
    assert sorted(e["step"] for e in events(native, "validation_generations")) == [30, 60]
    assert samples == [("original input", "original output", 0.25)]


@pytest.mark.parametrize("error", [ValueError("bad programming input"), TypeError("SDK bug"),
                                   RuntimeError("unrecognized failure")])
def test_unknown_sdk_errors_propagate(native, error):
    logger = native.make()
    native.sdk.log = lambda **kw: (_ for _ in ()).throw(error)
    with pytest.raises(type(error), match=str(error)):
        logger.log({"loss": 0.2}, 1)
    assert not events(native, "backend_error")


@pytest.mark.parametrize("value,error_type", [(float("nan"), ValueError), (object(), TypeError)])
def test_strict_serialization_failure_before_sdk(native, value, error_type):
    logger = native.make()
    before = len(native.calls)
    with pytest.raises(error_type):
        logger.log({"loss": value}, 1)
    assert len(native.calls) == before and not events(native, "metrics")


def test_disk_failure_propagates_before_sdk(native, monkeypatch):
    logger = native.make()
    before = len(native.calls)
    monkeypatch.setattr("os.fsync", lambda fd: (_ for _ in ()).throw(OSError("disk failure")))
    with pytest.raises(OSError, match="disk failure"):
        logger.log({"loss": 0.2}, 1)
    assert len(native.calls) == before and not events(native, "metrics")


def test_non_wandb_failure_propagates(native):
    logger = native.make()
    logger.logger["console"] = SimpleNamespace(log=lambda **kw: (_ for _ in ()).throw(ConnectionResetError("console")))
    with pytest.raises(ConnectionResetError, match="console"):
        logger.log({"loss": 0.2}, 1)


def test_disabled_opt_in_retains_failure_propagation(native, monkeypatch):
    monkeypatch.setenv("BASELINE_DURABLE_TRACKING_ENABLE", "0")
    logger = native.make()
    native.sdk.log = lambda **kw: (_ for _ in ()).throw(RuntimeError(CLOSED))
    with pytest.raises(RuntimeError, match="UnixTransport"):
        logger.log({"loss": 0.2}, 1)
    assert not events(native, "metrics")


def test_real_hydra_config_archive_and_sdk_input_unchanged(native):
    from omegaconf import OmegaConf
    config = OmegaConf.create({"trainer": {"logger": ["wandb"]}, "api_key": "private-fixture", "token": "test-token"})
    # Use the constructor with its actual Hydra container rather than a shadow class.
    logger = native.module.Tracking("project", "experiment", "wandb", config=config)
    try:
        assert native.calls[0][1]["config"] is config
        saved = events(native, "config")[0]["payload"]["config"]
        assert saved["trainer"]["logger"] == ["wandb"] and saved["api_key"] == "[REDACTED]"
        assert saved["token"] == "[REDACTED]"
        assert config.api_key == "private-fixture" and config.token == "test-token"
    finally:
        logger.logger = {}
        logger._durable = None
