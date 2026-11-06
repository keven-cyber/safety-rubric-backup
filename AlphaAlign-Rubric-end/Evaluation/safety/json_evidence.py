"""Complete JSON handoffs on local disks and asynchronously mounted storage.

Retries only storage visibility/integrity, never model verdicts. Each rejected
read is preserved; an exhausted or permanently corrupt file remains an error.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _publish(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.tmp.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _record_rejected_read(path, attempt, raw, manifest_raw, error):
    directory = Path(os.getenv('SAFETY_JSON_EVIDENCE_DIR', str(path.parent / 'json_read_errors')))
    directory.mkdir(parents=True, exist_ok=True)
    stem = f'{path.name}.{os.getpid()}.{attempt}.{uuid.uuid4().hex}'
    # Preserve the failed bytes without attempting to repair or reinterpret them.
    _publish(directory / (stem + '.bin'), raw)
    if manifest_raw is not None:
        _publish(directory / (stem + '.manifest.bin'), manifest_raw)
    record = dict(path=str(path), attempt=attempt, bytes=len(raw), sha256=_digest(raw),
                  manifest_sha256=_digest(manifest_raw) if manifest_raw is not None else None,
                  error_type=type(error).__name__, error=str(error), timestamp=time.time())
    _publish(directory / (stem + '.json'), json.dumps(record).encode('utf-8'))


def read_json_complete(path, expected_sha256=None, max_attempts=8):
    path = Path(path)
    if max_attempts < 1:
        raise ValueError('A JSON read needs at least one attempt')
    manifest_path = path.with_name(path.name + '.integrity.json')
    for attempt in range(1, max_attempts + 1):
        raw = b''
        manifest_raw = None
        try:
            raw = path.read_bytes()
            if manifest_path.exists():
                manifest_raw = manifest_path.read_bytes()
                manifest = json.loads(manifest_raw)
                if (set(manifest) != {'sha256', 'bytes'} or type(manifest['bytes']) is not int
                        or manifest['bytes'] != len(raw) or manifest['sha256'] != _digest(raw)):
                    raise ValueError('JSON payload does not match its committed integrity manifest')
            if expected_sha256 is not None and _digest(raw) != expected_sha256:
                raise ValueError('JSON payload does not match the just-written generation')
            return json.loads(raw)
        except (OSError, UnicodeError, ValueError) as error:
            _record_rejected_read(path, attempt, raw, manifest_raw, error)
            if attempt == max_attempts:
                raise
            time.sleep(min(0.2 * 2 ** (attempt - 1), 1.0))


def write_json_atomic(path, value):
    path = Path(path)
    raw = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8')
    digest = _digest(raw)
    _publish(path, raw)
    manifest = json.dumps(dict(sha256=digest, bytes=len(raw))).encode('utf-8')
    _publish(path.with_name(path.name + '.integrity.json'), manifest)
    # Do not hand a subsequent stage a path until these exact bytes are readable.
    read_json_complete(path, expected_sha256=digest)
    return digest
