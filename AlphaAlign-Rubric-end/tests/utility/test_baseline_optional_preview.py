"""Execute the native preview method without importing GPU trainers."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(params=["AlphaAlign/verl", "NSPO/verl/verl"])
def preview(request):
    path = ROOT / request.param / "trainer/ppo/ray_trainer.py"
    tree = ast.parse(path.read_text())
    method = next(node for node in ast.walk(tree)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "_maybe_log_val_generations")
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
    return scope[method.name]


def test_zero_preview_never_calls_failed_artifact_backend(preview):
    class BrokenArtifactBackend:
        def log(self, *args):
            raise RuntimeError("offline artifact transport closed")

    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(
            log_val_generations=0, logger=["wandb"])),
        validation_generations_logger=BrokenArtifactBackend(), global_steps=30)
    # The native method returns before slicing, shuffling, or invoking W&B.
    preview(trainer, None, None, None)


def test_enabled_preview_failure_still_propagates(preview):
    class BrokenArtifactBackend:
        def log(self, *args):
            raise RuntimeError("offline artifact transport closed")

    trainer = SimpleNamespace(
        config=SimpleNamespace(trainer=SimpleNamespace(
            log_val_generations=10, logger=["wandb"])),
        validation_generations_logger=BrokenArtifactBackend(), global_steps=30)
    with pytest.raises(RuntimeError, match="artifact transport closed"):
        preview(trainer, ["input"], ["output"], [0.5])
