"""CPU-only original validation replay and remaining-stage contracts."""
import importlib.util
import json
from pathlib import Path
import os
import shutil
import pytest
from verl.utils import final_stage_resume as stage
from verl.utils.validation_recovery import write, JudgeScoringUnavailable

spec=importlib.util.spec_from_file_location('actual_review_fixture',Path(__file__).with_name('test_frozen_validation_review.py'))
f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)


def setup_replay(tmp_path, monkeypatch):
    obj, loads=f.make_trainer(tmp_path,monkeypatch)
    spec2=importlib.util.spec_from_file_location('complete_checkpoint_fixture',Path(__file__).with_name('test_checkpoint_recovery.py'))
    cp=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(cp)
    source=tmp_path/'source'
    for n in (180,900,990,1080,1170):
        shutil.rmtree(source/f'global_step_{n}')
        cp.original(source,n)
    root=Path(obj.config.trainer.default_local_dir);run=tmp_path/'run';run.mkdir()
    original=tmp_path/'job_original';original.mkdir()
    write(original/'effective_config.json',dict(trainer=dict(default_local_dir=str(source))))
    write(root/'checkpoint_recovery.json',dict(status='ready',source_checkpoint=str(source/'global_step_1170'),restored_step=1170))
    snapshot=dict(inputs=['decoded text 1']*304,outputs=['decoded text 1']*304,
      tensors=dict(prompts=[[1,1]]*304,responses=[[1,1]]*304,input_ids=[[1,1,1,1]]*304,
                   attention_mask=[[1,1,1,1]]*304,position_ids=[[0,0,0,0]]*304),
      non_tensor_batch=obj._rollout_trace_json_value({k:v for k,v in obj.val_dataloader[0].items() if k in ('data_source','extra_info')}),meta_info={})
    generation=original/'validation_pending/generations/gap.json';write(generation,dict(global_step=180,batch_index=1,snapshot=snapshot))
    event=dict(global_step=180,batch_index=1,status='incomplete',aggregate_reward=None,best_eligible=False,
      generated_batch=str(generation),checkpoint=str(source/'global_step_180'))
    write(source/'validation_pending/gap.json',event);write(root/'validation_pending/gap.json',event)
    for n in range(30,1171,30):
        if n!=180:write(original/'metrics'/f'step_{n:08d}.json',dict(metrics={stage.KEY:-2.+n/100000}))
    write(source/'best_checkpoint_info.json',dict(best_metric_step=1170,best_metric_value=-2.+1170/100000,
      checkpoint_storage='numbered_checkpoint_reference_v1',best_checkpoint_relative_path='global_step_1170'))
    monkeypatch.setenv('RUBRIC_FROZEN_REVIEW_ENABLE','0');monkeypatch.setenv('PRIMUS_JOB_UUID','job')
    monkeypatch.setenv('RUBRIC_VALIDATION_REPLAY_SOURCE_RUN',str(original));monkeypatch.setenv('RUBRIC_RECOVERY_SOURCE_CHECKPOINT',str(source/'global_step_1170'))
    return obj,loads,source,generation


def test_replays_original304_once_through_real_validator_manager_without_policy_load_or_update(tmp_path,monkeypatch):
    obj,loads,source,generation=setup_replay(tmp_path,monkeypatch)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    previous={k:os.environ.get(k) for k in ('RUBRIC_RUN_DIR','RUBRIC_INTERMEDIATE_RECORD_ENABLE','RUBRIC_BASE_JUDGE_TRACE_DIR')}
    stage.validation_replay(obj)
    assert previous=={k:os.environ.get(k) for k in previous}
    assert loads==[]
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    selected=Path(obj.config.trainer.default_local_dir)/'original_validation_replay_selected'
    value=stage.verify_replay_selection(selected,stage.ROLES,1170)
    assert value['best_metric']['best_metric_step']==180
    assert any(x['roles']==['best'] and x['step']==180 for x in value['checkpoints'])
    assert (source/'validation_pending/gap.json').exists()
    raw=generation.read_bytes();generation.write_bytes(raw+b' ')
    with pytest.raises(ValueError,match='changed'):stage.verify_replay_selection(selected,stage.ROLES,1170)


@pytest.mark.parametrize('change',['prompt','rules','tokens','missing304'])
def test_alignment_and_missing_originals_fail_before_scoring(tmp_path,monkeypatch,change):
    obj,loads,source,generation=setup_replay(tmp_path,monkeypatch)
    v=json.loads(generation.read_text())
    if change=='prompt':v['snapshot']['tensors']['prompts'][0][0]=2
    if change=='rules':v['snapshot']['non_tensor_batch']['extra_info'][0]['new_rule']='unexpected'
    if change=='tokens':v['snapshot']['tensors']['responses'][0][0]=2
    if change=='missing304':v['snapshot']['outputs'].pop()
    write(generation,v)
    if change=='tokens':
        # A changed original token archive cannot pass the original scored text
        # comparison; make real manager inspect the non-matching decoded output.
        obj.tokenizer.decode=lambda ids,**kw:'different '+str(int(ids[0]))
    with pytest.raises(ValueError):stage.validation_replay(obj)
    assert not list(Path(obj.config.trainer.default_local_dir).rglob('validation_replay_completed.json'))
    assert loads==[]


def test_known_unavailable_keeps_whole_validation_unknown_and_does_not_retry(tmp_path,monkeypatch):
    obj,loads,source,generation=setup_replay(tmp_path,monkeypatch)
    calls=[]
    def unavailable(*a,**kw):calls.append(1);raise JudgeScoringUnavailable('one finite unavailable attempt')
    obj.val_reward_fn=unavailable
    with pytest.raises(ValueError):stage.validation_replay(obj)
    assert calls==[1] and loads==[]
    assert not list(Path(obj.config.trainer.default_local_dir).rglob('validation_replay_completed.json'))
    assert json.loads((tmp_path/'run/original_validation_replay/result.json').read_text())['metrics']=={}
    assert (source/'validation_pending/gap.json').exists()


@pytest.mark.parametrize('error',[OSError('disk'),RuntimeError('program')])
def test_storage_program_errors_propagate(tmp_path,monkeypatch,error):
    obj,loads,_,_=setup_replay(tmp_path,monkeypatch)
    obj.val_reward_fn=lambda *a,**kw:(_ for _ in ()).throw(error)
    with pytest.raises(type(error),match=str(error)):stage.validation_replay(obj)


def test_full39_chronology_and_strict_ties():
    records=[dict(step=n,metrics={stage.KEY:1.}) for n in range(30,1171,30)]
    assert stage.replay_rank(records)['step']==30
    with pytest.raises(ValueError):stage.replay_rank(records[1:])
    records[5]['metrics'][stage.KEY]=float('nan')
    with pytest.raises(ValueError):stage.replay_rank(records)


def test_complete_original_safety_summary_required_and_empty_hash_not_input_identity(tmp_path):
    rows=[]
    counts=dict(wildjailbreak_adv=210,Jailbreaktrigger=400,WildGuardTest=329,strongreject=313,advbench=520,gcg_official=47,pair_official=71)
    for name,n in counts.items():rows.append('LlamaGuard Statistics for '+name+':\n'+json.dumps(dict(answer_only=dict(total_responses=n,evaluation_complete=True,invalid_judgment_count=0))))
    rows.append(json.dumps(dict(dataset='xstest',evaluation_complete=True,total_dataset_prompts=250,matched_prompts=250,valid_refusal_judgments=250,missing_prompts_in_results=0,ignored_none_judgments=0,invalid_refusal_judgments=0)))
    p=tmp_path/'summary.txt';p.write_text('\n'.join(rows));assert stage.check_safety(p)['path']==str(p)
    p.write_text('\n'.join(rows[1:]))
    with pytest.raises(ValueError):stage.check_safety(p)
    p=tmp_path/'raw.json';write(p,dict(config=dict(limit=None),task_hashes={},**{'n-samples':{'gsm8k':dict(original=1319,effective=1319)}},results=dict(gsm8k_cot_modelscope={'exact_match,flexible-extract':.5})))
    assert stage.check_general(p,'gsm8k_cot_modelscope')['exact_input_bytes_verified'] is False
    v=json.loads(p.read_text());v['config']['limit']=10;write(p,v)
    with pytest.raises(ValueError):stage.check_general(p,'gsm8k_cot_modelscope')


@pytest.mark.parametrize('dtype_field', ['dtype', 'torch_dtype'])
def test_general_only_continuation_runs_exact_four_models_and_only_two_original_tasks(tmp_path,monkeypatch,dtype_field):
    from verl.utils import frozen_validation_review as review
    source=tmp_path/'source';source.mkdir();source_run=tmp_path/'job_old';source_run.mkdir();run=tmp_path/'newrun';run.mkdir()
    chosen=dict(requested=list(stage.ROLES),last_step=1170,checkpoints=[dict(step=n,checkpoint=str(source/'new_reevaluation/selected_checkpoints'/f'global_step_{n}'),roles=[str(n)]) for n in (900,990,1080,1170)])
    chosen['checkpoints'][0]['roles'].append('best')
    write(source_run/'effective_config.json',dict(trainer=dict(default_local_dir=str(source))))
    write(source_run/'eval_outputs/checkpoint_selection.json',chosen)
    monkeypatch.setattr(review,'selection',lambda *a:chosen)
    monkeypatch.setattr(stage,'check_safety',lambda p:stage.seal(p))
    for n in (900,990,1080,1170):
        name=f'Qwen2.5-3B-Instruct-global_step_{n}'
        write(source/'models/Qwen'/name/'config.json',{dtype_field:'bfloat16'})
        (source/'models/Qwen'/name/'model.safetensors').write_bytes(b'CPU tiny placeholder, not GPU proof')
        (source_run/'eval_outputs'/(name+'_summary.txt')).write_text('CPU only intercepted completed safety fixture')
    monkeypatch.setattr(stage,'prepare',lambda source,dest,*a:dest.mkdir())
    calls=[]
    def general(argv,env,check):
        calls.append(argv);assert '--tasks' in argv and argv[argv.index('--tasks')+1]=='gsm8k_cot_modelscope,mmlu_modelscope'
        name=argv[argv.index('--glob')+1]
        for task in ('gsm8k_cot_modelscope','mmlu_modelscope'):
            counts={str(i):dict(original=246,effective=246) for i in range(57)} if task=='mmlu_modelscope' else {'gsm':dict(original=1319,effective=1319)}
            if task=='mmlu_modelscope':counts['56']=dict(original=266,effective=266)
            metric='acc,none' if task=='mmlu_modelscope' else 'exact_match,flexible-extract'
            write(Path(env['EVAL_OUTPUT_DIR'])/name/task/'model/results_fixture.json',dict(config=dict(limit=None),**{'n-samples':counts},results={task:{metric:.5}},task_hashes={}))
    monkeypatch.setattr(stage.subprocess,'run',general)
    for key,value in dict(RUBRIC_RUN_DIR=str(run),RUBRIC_FINAL_RESUME_SOURCE_RUN=str(source_run),RUBRIC_RECOVERY_SOURCE_CHECKPOINT=str(source/'global_step_1170'),PRIMUS_JOB_UUID='job',PRIMUS_SAVE_CHECKPOINT_DIR=str(tmp_path),PRIMUS_JOB_SUBMIT_TIMESTAMP='new').items():monkeypatch.setenv(key,value)
    before={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    stage.general_resume()
    assert len(calls)==4 and len(json.loads((run/'general_resume_completed.json').read_text())['raw_general_results'])==8
    assert before=={str(p):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    # Existing general result: refuse an unchanged replay, do not rerun until pass.
    (source_run/'eval_outputs/a_general_scores.txt').write_text('already attempted')
    with pytest.raises(ValueError,match='Existing general results'):stage.general_resume()
