import hashlib
import importlib.util
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    'targeted_bootstrap', ROOT / 'scripts/prepare_targeted_refusal_validation.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def setup_tree(tmp_path):
    patch = []
    before, after = {}, {}
    for name in sorted(bootstrap.CHECKED):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        initial = b'original source\n'
        if name == bootstrap.FROZEN:
            initial = (ROOT / name[len(bootstrap.PROJECT):]).read_bytes()
        path.write_bytes(initial)
        after[name] = digest(initial)
        if name in bootstrap.CHANGED:
            before[name] = digest(initial)
            after[name] = digest(b'candidate source\n')
            patch += [f'diff --git a/{name} b/{name}', f'--- a/{name}', f'+++ b/{name}',
                      '@@ -1 +1 @@', '-original source', '+candidate source']
    patch = ('\n'.join(patch) + '\n').encode()
    manifest = dict(before=before, after=after, patch_sha256=digest(patch))
    return patch, manifest


@pytest.mark.parametrize('empty_git_index', [False, True])
def test_exported_or_untracked_primus_files_do_not_need_checkout(tmp_path, empty_git_index):
    patch, manifest = setup_tree(tmp_path)
    if empty_git_index:
        subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
        assert subprocess.check_output(['git', 'ls-files'], cwd=tmp_path) == b''
    unrelated = tmp_path / 'primus_user_file.txt'
    unrelated.write_text('must survive')
    result = bootstrap.prepare(tmp_path, patch, manifest)
    assert result['status'] == 'ready' and not result['already_prepared']
    assert unrelated.read_text() == 'must survive'
    assert len(result['source_sha256']) == len(bootstrap.CHECKED) - 1
    assert bootstrap.prepare(tmp_path, patch, manifest)['already_prepared'] is True
    if empty_git_index:
        assert subprocess.check_output(['git', 'ls-files'], cwd=tmp_path) == b''


@pytest.mark.parametrize('problem', ['wrong_patch_hash', 'wrong_base', 'altered_fixture',
                                    'missing_manifest_file', 'unexpected_patch_path', 'symlink'])
def test_invalid_input_never_overwrites_mounted_files(tmp_path, problem):
    patch, manifest = setup_tree(tmp_path)
    if problem == 'wrong_patch_hash':
        manifest['patch_sha256'] = '0' * 64
    elif problem == 'wrong_base':
        (tmp_path / sorted(bootstrap.CHANGED)[0]).write_text('unrelated user edit\n')
    elif problem == 'altered_fixture':
        (tmp_path / bootstrap.FROZEN).write_text('{}')
    elif problem == 'missing_manifest_file':
        manifest['after'].pop(sorted(bootstrap.CHANGED)[0])
    elif problem == 'unexpected_patch_path':
        patch += b'diff --git a/unrelated b/unrelated\n--- a/unrelated\n+++ b/unrelated\n'
        manifest['patch_sha256'] = digest(patch)
    else:
        name = sorted(bootstrap.CHANGED)[0]
        path = tmp_path / name
        target = tmp_path / 'private_source.txt'
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        bootstrap.prepare(tmp_path, patch, manifest)
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


def test_bootstrap_never_runs_git_checkout_or_modifies_index(monkeypatch, tmp_path):
    patch, manifest = setup_tree(tmp_path)
    real_run = subprocess.run
    commands = []
    def record(command, **kwargs):
        commands.append(command)
        return real_run(command, **kwargs)
    monkeypatch.setattr(bootstrap.subprocess, 'run', record)
    bootstrap.prepare(tmp_path, patch, manifest)
    assert commands == [['git', 'apply', '--unidiff-zero', '--check', '-'],
                        ['git', 'apply', '--unidiff-zero', '-']]
