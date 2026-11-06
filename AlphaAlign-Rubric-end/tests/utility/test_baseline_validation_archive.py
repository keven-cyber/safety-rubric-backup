"""Finite CPU archive tests; no policy generation or reward/model call."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
ROOT=Path(__file__).resolve().parents[2]

def module():
    p=ROOT/'AlphaAlign/verl/utils/baseline_validation_archive.py';spec=importlib.util.spec_from_file_location('native_archive_fixture',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def batch(n=304):
    return SimpleNamespace(batch={'prompts':torch.ones((n,3),dtype=torch.long),'responses':torch.ones((n,3),dtype=torch.long),'attention_mask':torch.ones((n,6),dtype=torch.long),'position_ids':torch.arange(6).repeat(n,1)},non_tensor_batch={'source':['CPU fixture']*n},meta_info={'untouched':True})

def test_originals_persist_before_scoring_and_cannot_be_replaced(tmp_path,monkeypatch):
    m=module();monkeypatch.setenv('BASELINE_VALIDATION_ARCHIVE','1');monkeypatch.setenv('BASELINE_RUN_DIR',str(tmp_path));t=SimpleNamespace(global_steps=30);b=batch();original=b.batch['responses'].clone();root=m.original(t,b)
    assert (root/'original304.json').is_file() and not (root/'scored304.json').exists()
    assert torch.equal(b.batch['responses'],original)
    with pytest.raises(FileExistsError):m.original(t,b)
    m.scored(root,[.5]*304);assert (root/'scored304.json').is_file()

@pytest.mark.parametrize('scores',[[.2]*303,[.2]*303+[float('nan')],[.2]*303+[None]])
def test_incomplete_or_nonfinite_score_never_publishes_complete(tmp_path,monkeypatch,scores):
    m=module();monkeypatch.setenv('BASELINE_VALIDATION_ARCHIVE','1');monkeypatch.setenv('BASELINE_RUN_DIR',str(tmp_path));root=m.original(SimpleNamespace(global_steps=30),batch())
    with pytest.raises(ValueError):m.scored(root,scores)
    assert (root/'original304.json').is_file() and not (root/'scored304.json').exists()

def test_partial_validation_generation_rejected_before_reward(tmp_path,monkeypatch):
    m=module();monkeypatch.setenv('BASELINE_VALIDATION_ARCHIVE','1');monkeypatch.setenv('BASELINE_RUN_DIR',str(tmp_path))
    with pytest.raises(ValueError):m.original(SimpleNamespace(global_steps=30),batch(303))

def test_both_native_copies_identical():
    assert (ROOT/'AlphaAlign/verl/utils/baseline_validation_archive.py').read_bytes()==(ROOT/'NSPO/verl/verl/utils/baseline_validation_archive.py').read_bytes()
