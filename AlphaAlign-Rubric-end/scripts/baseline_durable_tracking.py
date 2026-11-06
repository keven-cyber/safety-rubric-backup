"""Optional W&B transport containment after durable, strict event recording.

The completed Qwen2.5-3B runs used the same record-before-SDK ordering.
This adapter is opt-in for native baselines; it never handles trainer/reward
exceptions, and unknown SDK/program errors still propagate.
"""
import errno
import json
import os
import uuid
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path


def json_container(value, *, redact=False):
    if isinstance(value, Mapping):
        return {key: ("[REDACTED]" if redact and (str(key).lower() in {"token", "api_base", "api_url"} or any(term in str(key).lower()
                    for term in ("api_key", "access_key", "secret", "password",
                                 "credential", "authorization")))
                    else json_container(item, redact=redact))
                for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [json_container(item, redact=redact) for item in value]
    # Unknown objects/nonfinite numbers fail strict JSON; no stringification.
    return value


def transport_error(error):
    if isinstance(error, (ConnectionError, BrokenPipeError, EOFError)):
        return True
    if isinstance(error, OSError) and error.errno in {
            errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN}:
        return True
    # Exact family of messages in the archived native W&B crash. Do not turn
    # arbitrary RuntimeError, TypeError, ValueError or numeric errors into success.
    message = str(error)
    return (isinstance(error, RuntimeError) and "UnixTransport" in message
            and "closed=True" in message and "the handler is closed" in message)


class DurableTracking:
    def __init__(self):
        raw = Path(os.environ["BASELINE_RUN_DIR"])
        workspace = Path(os.environ["PRIMUS_OSS_WORKSPACE_DIR"]).resolve()
        root = raw.resolve()
        if raw.is_symlink() or root == workspace or workspace not in root.parents:
            raise ValueError("Baseline tracking requires an isolated persistent run")
        self.directory = root / "tracking"
        if self.directory.is_symlink():
            raise ValueError("Tracking directory must not be a symbolic link")
        self.directory.mkdir(parents=True, exist_ok=True)

    def record(self, kind, payload, step=None):
        raw = json.dumps(json_container(dict(kind=kind, step=step, payload=payload),
                                       redact=kind == "config"),
                         ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        path = self.directory / f"{kind}_{step}_{os.getpid()}_{uuid.uuid4().hex}.json"
        temporary = path.with_suffix(".tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def failure(self, operation, error, step=None):
        if not transport_error(error):
            raise error
        self.record("backend_error", dict(backend="wandb", operation=operation,
                    error_type=type(error).__name__, message=str(error)), step)
        warnings.warn(f"W&B {operation} transport failed; complete baseline events "
                      "remain in persistent tracking files.", RuntimeWarning)
