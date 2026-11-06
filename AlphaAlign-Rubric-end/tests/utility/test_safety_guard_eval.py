import importlib.util
import json
from pathlib import Path

import pytest


def _load_eval_answer():
    path = Path(__file__).parents[2] / "Evaluation/safety/eval_answer.py"
    spec = importlib.util.spec_from_file_location("safety_eval_answer_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_answer = _load_eval_answer()


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_qwen3guard_preflight_rejects_wrong_served_model(monkeypatch):
    monkeypatch.setenv("SAFETY_GUARD_MODEL", "qwen3_guard_gen_8b")
    monkeypatch.setattr(
        eval_answer.urllib.request,
        "urlopen",
        lambda *args, **kwargs: _Response({"data": [{"id": "qwen3_32B_instruct"}]}),
    )

    with pytest.raises(RuntimeError, match="is not served"):
        eval_answer.Qwen3GuardAPI()


def test_qwen3guard_request_error_is_not_treated_as_safe():
    guard = eval_answer.Qwen3GuardAPI.__new__(eval_answer.Qwen3GuardAPI)

    def fail_request(prompt, answer):
        raise ConnectionError("guard unavailable")

    guard._call_guard_api = fail_request
    with pytest.raises(ConnectionError, match="guard unavailable"):
        guard.evaluate_single("prompt", "answer")


def test_qwen3guard_parses_refusal_label_for_overrefusal_eval():
    result = eval_answer._parse_guardrail_text(
        "Safety: Safe\nRefusal: Yes\nCategories: None"
    )

    assert result["is_safe"] is True
    assert result["refusal_label"] == "yes"
