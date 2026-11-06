"""Prepare a validation-only patch without checking out Primus-mounted files.

Primus may supply an exported working tree with an empty Git index. Never
fetch/switch/reset/clean that tree. Verify every input, then use working-tree
git apply (no --index) only for the reviewed correction and preflight files.
"""
import hashlib
from pathlib import Path
import subprocess

PROJECT = 'AlphaAlign-Rubric-end/'
CHANGED = frozenset(PROJECT + path for path in (
    'verl/utils/reward_score/rubric_judge.py', 'scripts/smoke_targeted_refusal.py',
    'verl/trainer/ppo/ray_trainer.py', 'train_val.sh'))
FROZEN = PROJECT + 'tests/fixtures/targeted_refusal_existing_answers.json'
FIXTURE_SHA256 = 'b3f35d0eba94455ae4c3c4646d7e2f8e9749cdde4f9358fc3b451f3fff2b6dc8'
CHECKED = CHANGED | {FROZEN} | {PROJECT + path for path in (
    'verl/utils/targeted_refusal.py', 'verl/utils/validation_selection.py',
    'verl/trainer/ppo/core_algos.py', 'verl/trainer/ppo/ray_trainer.py',
    'verl/utils/reward_score/rubric_judge_save_evolve_rubric.py')}


def prepare(repo, patch, manifest):
    repo = Path(repo).resolve()
    if not (repo / PROJECT).is_dir():
        raise ValueError('Expected a dedicated verl source directory')
    if set(manifest['before']) != CHANGED or set(manifest['after']) != CHECKED:
        raise ValueError('Candidate manifest must cover all reviewed and frozen files')
    if manifest['after'][FROZEN] != FIXTURE_SHA256:
        raise ValueError('Frozen case expectations must not change')
    if hashlib.sha256(patch).hexdigest() != manifest['patch_sha256']:
        raise ValueError('Candidate patch hash mismatch')
    paths = []
    for line in patch.decode('utf-8').splitlines():
        if line.startswith('diff --git '):
            fields = line.split()
            if len(fields) != 4 or fields[2] != 'a/' + fields[3][2:]:
                raise ValueError('Only ordinary same-path file edits are allowed')
            paths.append(fields[3][2:])
    if len(paths) != len(CHANGED) or set(paths) != CHANGED:
        raise ValueError('Patch contains an unexpected or missing source file')
    if any(line.startswith(('new file mode', 'deleted file mode', 'rename from',
                            'rename to', 'old mode', 'new mode', 'GIT binary patch'))
           for line in patch.decode('utf-8').splitlines()):
        raise ValueError('Patch must not add, remove, rename or change file modes')

    def hashes():
        output = {}
        for name in CHECKED:
            path = repo / name
            if path.resolve() != path or not path.is_file():
                raise ValueError('Expected regular non-symlink candidate source files')
            output[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return output

    current = hashes()
    already_prepared = current == manifest['after']
    if not already_prepared:
        expected = {**manifest['after'], **manifest['before']}
        if current != expected:
            raise ValueError('Mounted sources differ from reviewed base; refusing to overwrite')
        # No branch switch and no index requirement: exported/untracked files are valid.
        subprocess.run(['git', 'apply', '--unidiff-zero', '--check', '-'], cwd=repo, input=patch, check=True)
        subprocess.run(['git', 'apply', '--unidiff-zero', '-'], cwd=repo, input=patch, check=True)
    actual = hashes()
    if actual != manifest['after']:
        raise ValueError('Prepared sources do not match the validated candidate')
    return dict(status='ready', already_prepared=already_prepared,
                patch_sha256=manifest['patch_sha256'], fixture_sha256=actual[FROZEN],
                source_sha256={name[len(PROJECT):]: digest for name, digest in actual.items()
                               if name != FROZEN})
