"""Download exact native Judge/Guard bytes without a mutable branch alias.

The ModelScope SDK accepts named tags/branches but rejects raw repository commit
IDs. Its public file endpoint accepts the pinned commit, with SHA256 verification.
"""
import json
import os
from pathlib import Path
import sys

from cross_model_policy import download_modelscope

if __name__ == '__main__':
    model, destination, report = sys.argv[1:]
    contracts = json.loads((Path(__file__).resolve().parents[1] / 'configs/cross_model_twelve_dependency_models.json').read_text())
    contract = contracts[model]
    root = Path(destination)
    # train_val.sh creates the empty directory first. Never remove old content.
    if root.exists():
        if root.is_symlink() or root.resolve() != root or any(root.iterdir()):
            raise FileExistsError('Dependency destination must be fresh and empty')
        root.rmdir()
    download_modelscope(root, contract)
    with Path(report).open('x') as stream:
        json.dump(dict(model=model, revision=contract['ModelScope_revision'],
                       files=contract['required_snapshot_files'],
                       actual_stream_SHA256_and_sizes_verified=True,
                       GPU_model_load_verified=False), stream, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
