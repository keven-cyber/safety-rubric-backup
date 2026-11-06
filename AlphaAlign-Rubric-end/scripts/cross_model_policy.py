"""Prepare a fixed, own-initial policy snapshot before any training service.

Only official model repositories are downloaded. A locally supplied authorized
snapshot must satisfy the same fixed metadata and complete-shard contract.
This command performs no policy inference and loads no policy weight tensors.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_metadata(root, contract):
    root = Path(root)
    for name, expected in contract['metadata_sha256'].items():
        if expected is None:
            raise ValueError('Unresolved authorized model metadata: ' + name)
        if digest(root / name) != expected:
            raise ValueError('Policy snapshot differs from fixed metadata: ' + name)
    config = json.loads((root / 'config.json').read_text())
    if (config.get('model_type') != contract['expected_model_type']
            or config.get('architectures') != [contract['expected_architecture']]):
        raise ValueError('Wrong policy architecture/model family')
    return config


def verify_weights(root):
    root = Path(root)
    if root.is_symlink() or root.resolve() != root or not root.is_dir():
        raise ValueError('Policy root must be an absolute physical directory')
    if any(root.rglob('*.incomplete')):
        raise ValueError('Incomplete policy snapshot')
    indices = list(root.glob('*.safetensors.index.json')) + list(root.glob('pytorch_model.bin.index.json'))
    if len(indices) > 1:
        raise ValueError('Ambiguous policy weight indices')
    if indices:
        names = set(json.loads(indices[0].read_text())['weight_map'].values())
    else:
        names = {p.name for p in root.glob('*.safetensors')} or {p.name for p in root.glob('pytorch_model*.bin')}
    if not names:
        raise ValueError('No policy weight shards')
    records = {}
    for name in sorted(names):
        path = root / name
        if (Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink()
                or not path.is_file() or path.stat().st_size <= 0):
            raise ValueError('Missing or unsafe policy shard: ' + name)
        before = path.stat()
        sha = digest(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Policy shard changed during verification')
        records[name] = {'bytes': before.st_size, 'sha256': sha}
    return records


def tokenizer_contract(root, contract):
    from transformers import AutoConfig, AutoTokenizer
    config = AutoConfig.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    if config.model_type != contract['expected_model_type'] or not tokenizer.chat_template:
        raise ValueError('Native policy config/tokenizer unavailable')
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise ValueError('Native policy EOS must resolve to a valid token')
    # Matches each native trainer's existing fallback, using this model's EOS.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    opts = {} if contract['enable_thinking'] is None else {'enable_thinking': contract['enable_thinking']}
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': 'Hello.'}],
        tokenize=False, add_generation_prompt=True, **opts)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids or max(ids) >= len(tokenizer):
        raise ValueError('Native chat-template/tokenizer mismatch')
    return {'class': type(tokenizer).__name__, 'eos_token_id': eos,
            'pad_token_id': tokenizer.pad_token_id, 'bos_token_id': tokenizer.bos_token_id,
            'config_eos_token_id': config.eos_token_id, 'vocab_size': len(tokenizer),
            'chat_template_sha256': hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
            'probe_prompt_token_count': len(ids), 'enable_thinking': contract['enable_thinking']}


def download_modelscope(root, contract):
    """Stream the reviewed fixed mirror files and verify every published byte."""
    import requests
    root.mkdir(parents=True, exist_ok=False)
    def fetch(item):
        name, expected = item
        if Path(name).name != name:
            raise ValueError('Mirror contract only permits reviewed root files')
        target = root / name
        partial = target.with_name(target.name + '.incomplete')
        url = 'https://modelscope.cn/api/v1/models/' + quote(contract['download_repo'], safe='/') + '/repo'
        h = hashlib.sha256(); size = 0
        with requests.get(url, params={'Revision': contract['revision'], 'FilePath': name},
                          stream=True, timeout=(30, 300)) as response:
            response.raise_for_status()
            with partial.open('xb') as stream:
                for block in response.iter_content(8 * 1024 * 1024):
                    if block:
                        stream.write(block); h.update(block); size += len(block)
                stream.flush(); os.fsync(stream.fileno())
        if h.hexdigest() != expected['sha256'] or size != expected['bytes']:
            raise ValueError('Fixed mirror snapshot integrity mismatch: ' + name)
        partial.rename(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch, contract['required_snapshot_files'].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--existing-authorized-snapshot')
    args = parser.parse_args()
    models = json.loads(Path(__file__).with_name('cross_model_twelve_models.json').read_text())
    contract = models[args.model]
    if any(value is None for value in contract['metadata_sha256'].values()):
        raise ValueError('Authorized model metadata/access must be resolved before submission: ' + args.model)
    report = Path(args.report)
    if report.exists() or report.is_symlink():
        raise FileExistsError('Refusing to replace model provenance: ' + str(report))
    root = Path(args.existing_authorized_snapshot or args.destination)
    if not args.existing_authorized_snapshot:
        if root.exists():
            raise FileExistsError('Use a fresh policy download destination')
        if contract.get('download_source') == 'modelscope':
            download_modelscope(root, contract)
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=args.model, revision=contract['revision'], local_dir=str(root),
                              endpoint='https://huggingface.co', max_workers=8)
    config = verify_metadata(root, contract)
    weights = verify_weights(root)
    if contract.get('download_source') == 'modelscope':
        for name, actual in weights.items():
            if actual != contract['required_snapshot_files'][name]:
                raise ValueError('Policy weights differ from the reviewed mirror and official LFS metadata')
    tokenizer = tokenizer_contract(root, contract)
    payload = dict(model_id=args.model, revision=contract['revision'], policy_path=str(root),
                   model_type=config['model_type'], architecture=config['architectures'],
                   tokenizer=tokenizer, metadata_sha256=contract['metadata_sha256'],
                   weights=weights, source='provided_authorized_snapshot' if args.existing_authorized_snapshot else contract.get('download_source', 'official_HF_fixed_revision'),
                   GPU_inference_verified=False, training_restore=False)
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open('x') as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(str(root))


if __name__ == '__main__':
    main()
