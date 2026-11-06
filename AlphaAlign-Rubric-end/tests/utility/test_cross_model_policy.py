"""CPU contract tests; no authorized model weights or model calls."""
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]

def load(name, path):
    spec=importlib.util.spec_from_file_location(name, ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

policy=load('policy_contract_test','scripts/cross_model_policy.py')
helper=load('policy_template_test','verl/utils/policy_template.py')

@pytest.mark.parametrize('value,expected',[(None,{}),('true',{'enable_thinking':True}),('false',{'enable_thinking':False})])
def test_optional_template_preserves_default_and_explicit_bool(monkeypatch,value,expected):
    monkeypatch.delenv('POLICY_ENABLE_THINKING',raising=False)
    if value is not None:monkeypatch.setenv('POLICY_ENABLE_THINKING',value)
    original={'other':'unchanged'}
    assert helper.policy_chat_template_kwargs(original)==dict(original,**expected)
    assert original=={'other':'unchanged'}

@pytest.mark.parametrize('value',['0','1','FALSE','','yes'])
def test_invalid_mode_fails(monkeypatch,value):
    monkeypatch.setenv('POLICY_ENABLE_THINKING',value)
    with pytest.raises(ValueError):helper.policy_chat_template_kwargs()

def test_conflicting_native_mode_rejected(monkeypatch):
    monkeypatch.setenv('POLICY_ENABLE_THINKING','false')
    with pytest.raises(ValueError):helper.policy_chat_template_kwargs({'enable_thinking':True})

def test_independent_helpers_identical():
    paths=['verl/utils/policy_template.py','AlphaAlign/verl/utils/policy_template.py','NSPO/verl/verl/utils/policy_template.py','Evaluation/safety/policy_template.py','Evaluation/lm-evaluation-harness/lm_eval/models/policy_template.py']
    assert len({(ROOT/p).read_bytes() for p in paths})==1

def test_all_policy_template_call_sites_receive_setting():
    paths=['verl/utils/dataset/rl_dataset.py','AlphaAlign/verl/utils/dataset/rl_dataset.py','Evaluation/safety/generate.py','Evaluation/lm-evaluation-harness/lm_eval/models/huggingface.py','Evaluation/lm-evaluation-harness/lm_eval/models/vllm_causallms.py']
    for path in paths:
        calls=[n for n in ast.walk(ast.parse((ROOT/path).read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='apply_chat_template']
        assert calls,path
        for call in calls:
            assert any(k.arg is None and isinstance(k.value,ast.Call) and getattr(k.value.func,'id','')=='policy_chat_template_kwargs' for k in call.keywords),path
    assert 'policy_chat_template_kwargs(config.get("apply_chat_template_kwargs", {}))' in (ROOT/'NSPO/verl/verl/utils/dataset/rl_dataset.py').read_text()

@pytest.mark.parametrize('backend',['huggingface','vllm_causallms'])
def test_actual_general_method_forwards_explicit_policy_setting(monkeypatch,backend):
    import jinja2
    from typing import List,Dict
    tree=ast.parse((ROOT/f'Evaluation/lm-evaluation-harness/lm_eval/models/{backend}.py').read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='apply_chat_template')
    module=ast.Module(body=[method],type_ignores=[])
    scope=dict(List=List,Dict=Dict,jinja2=jinja2,policy_chat_template_kwargs=helper.policy_chat_template_kwargs)
    exec(compile(ast.fix_missing_locations(module),'actual_general_method','exec'),scope)
    class Tokenizer:
        def apply_chat_template(self,*args,**kwargs):return kwargs
    from types import SimpleNamespace
    monkeypatch.setenv('POLICY_ENABLE_THINKING','false')
    result=scope['apply_chat_template'](SimpleNamespace(tokenizer=Tokenizer()),[{'role':'user','content':'hello'}])
    assert result['enable_thinking'] is False and result['tokenize'] is False

def test_complete_weight_index_requires_every_shard(tmp_path):
    root=tmp_path.resolve();(root/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'a':'a.safetensors','b':'b.safetensors'}}));(root/'a.safetensors').write_bytes(b'CPU fixture')
    with pytest.raises(ValueError):policy.verify_weights(root)
    (root/'b.safetensors').write_bytes(b'CPU fixture2')
    records=policy.verify_weights(root);assert set(records)=={'a.safetensors','b.safetensors'}
    (root/'b.safetensors').unlink();(root/'b.safetensors').symlink_to(root/'a.safetensors')
    with pytest.raises(ValueError):policy.verify_weights(root)

def test_unresolved_llama_metadata_fails_closed(tmp_path):
    contracts=json.loads((ROOT/'scripts/cross_model_twelve_models.json').read_text())
    contracts['meta-llama/Llama-3.2-3B-Instruct']['metadata_sha256']['config.json'] = None
    with pytest.raises(ValueError,match='Unresolved authorized'):
        policy.verify_metadata(tmp_path,contracts['meta-llama/Llama-3.2-3B-Instruct'])

def test_wrong_architecture_cannot_pass_even_with_matching_file_hash(tmp_path):
    files={name:'{}' for name in ['config.json','tokenizer.json','tokenizer_config.json','generation_config.json']};files['config.json']=json.dumps({'model_type':'qwen2','architectures':['Qwen2ForCausalLM']})
    for name,text in files.items():(tmp_path/name).write_text(text)
    contract={'metadata_sha256':{name:policy.digest(tmp_path/name) for name in files},'expected_model_type':'llama','expected_architecture':'LlamaForCausalLM'}
    with pytest.raises(ValueError,match='Wrong policy'):policy.verify_metadata(tmp_path,contract)

@pytest.mark.parametrize('corrupt',[False,True])
def test_mirror_downloader_verifies_stream_before_publication(tmp_path,monkeypatch,corrupt):
    import hashlib,requests
    raw=b'CPU metadata fixture only'
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def raise_for_status(self):pass
        def iter_content(self,*args):return iter([raw[:5],raw[5:]])
    monkeypatch.setattr(requests,'get',lambda *a,**kw:Response())
    expected={'bytes':len(raw),'sha256':'0'*64 if corrupt else hashlib.sha256(raw).hexdigest()}
    root=tmp_path/'snapshot'
    contract={'download_repo':'LLM-Research/Llama-3.2-3B-Instruct','revision':'fixed-test-revision','required_snapshot_files':{'config.json':expected}}
    if corrupt:
        with pytest.raises(ValueError,match='integrity mismatch'):policy.download_modelscope(root,contract)
        assert not (root/'config.json').exists()
        assert (root/'config.json.incomplete').is_file()
    else:
        policy.download_modelscope(root,contract)
        assert (root/'config.json').read_bytes()==raw
