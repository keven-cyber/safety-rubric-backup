"""Exercise the actual merge selection/repair shell with GPU merger intercepted."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest
from verl.utils import checkpoint_recovery as cr
from verl.utils.best_checkpoint_reference import reference
from scripts.select_final_checkpoints import select

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('final_numbered_complete_fixture',Path(__file__).with_name('test_checkpoint_recovery.py'))
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def actual_source(tmp_path):
    source=tmp_path/'source'
    for step in (900,990,1080,1110,1170):fixture.original(source,step)
    info=json.loads((source/'best_checkpoint_info.json').read_text())
    info.update(best_metric_step=1110,**reference(source,1110))
    (source/'best_checkpoint_info.json').write_text(json.dumps(info))
    (source/'best_checkpointed_iteration.txt').write_text('1110')
    return source


def actual_merge_block():
    s=(ROOT/'train_val.sh').read_text()
    return s[s.index('for ckpt_dir in "${CKPT_DIRS[@]}"; do'):s.index('# A best copy is optional:')]


@pytest.mark.parametrize('requested',[('900','990','1080','last','best'),('best',)])
def test_actual_exact_merge_loop_retains_numbered_best1110_and_every_selected_checkpoint(tmp_path,requested):
    source=actual_source(tmp_path);plan=select(source,requested,1170)
    paths=[x['checkpoint'] for x in plan['checkpoints']]
    import shlex
    prelude='''merge_rubricalign_checkpoint() { mkdir -p "$2"; echo '{}' >"$2/config.json"; }; MERGED_THIS_RUN=(); SAFETY_EVAL_MODELS=(); BEST_MERGED_MODEL=''; LAST_MERGED_MODEL=''; LAST_MERGED_STEP=-1;
'''
    code=prelude+'CKPT_DIRS=('+ ' '.join(shlex.quote(x) for x in paths)+')\n'+actual_merge_block()+'''printf '\\nCOUNT=%s BEST=%s SAFETY=%s\\n' "${#MERGED_THIS_RUN[@]}" "$BEST_MERGED_MODEL" "${#SAFETY_EVAL_MODELS[@]}"'''
    env=dict(os.environ,FINAL_EVAL_EXACT_TRIO='1',FINAL_SAFETY_EVAL_STEPS=','.join(requested),FINAL_EVAL_MIN_STEP='900',
             LAST_STANDARD_STEP='1170',SELECTED_BEST_CHECKPOINT_PATH=str(source/'global_step_1110'),
             MERGED_MODELS_DIR=str(tmp_path/'merged'),BASE_MODEL_NAME='Qwen2.5-3B-Instruct',RUBRIC_EVAL_ONLY='0')
    p=subprocess.run(['bash','-ec',code],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert f'COUNT={len(paths)}' in p.stdout and f'SAFETY={len(paths)}' in p.stdout
    assert 'BEST='+str(tmp_path/'merged/Qwen2.5-3B-Instruct-global_step_1110') in p.stdout
    assert '[SKIP] global_step_1110' not in p.stdout
    assert not (tmp_path/'source/changed').exists()


def test_actual_exact_merge_loop_cannot_report_success_after_omitting_selected_model(tmp_path):
    source=actual_source(tmp_path)
    code='''merge_rubricalign_checkpoint() { return 1; }; MERGED_THIS_RUN=(); SAFETY_EVAL_MODELS=(); CKPT_DIRS=("$MISSING/global_step_1110");\n'''+actual_merge_block()
    p=subprocess.run(['bash','-ec',code],env=dict(os.environ,MISSING=str(tmp_path/'absent'),FINAL_EVAL_EXACT_TRIO='1',
        FINAL_SAFETY_EVAL_STEPS='best',FINAL_EVAL_MIN_STEP='900',LAST_STANDARD_STEP='1170',SELECTED_BEST_CHECKPOINT_PATH=str(tmp_path/'absent/global_step_1110'),RUBRIC_EVAL_ONLY='0'),capture_output=True,text=True)
    assert p.returncode!=0 and 'refuse incomplete final evaluation' in p.stderr


def recovery_only_block():
    s=(ROOT/'train_val.sh').read_text()
    return s[s.index('# Explicit completion of a previously omitted'):s.index('source "${PROJECT_ROOT}/scripts/rubric_evidence_env.sh"')]


def test_actual_eval_only_repair_uses_verified_physical_copy_and_exact_original_best(tmp_path):
    source=actual_source(tmp_path);before=fixture.tree_bytes(source);dest=tmp_path/'copy'
    cr.prepare(source/'global_step_1170',dest,tmp_path)
    code=recovery_only_block()+'''printf '\\nRESULT=%s|%s|%s|%s\\n' "$RUBRIC_EVAL_ONLY" "$RUBRIC_EVAL_SOURCE_DIR" "$RUBRIC_EVAL_ACTOR_DIR" "$FINAL_SAFETY_EVAL_STEPS"'''
    env={k:v for k,v in os.environ.items() if not k.startswith(('PRIMUS_','RUBRIC_EVAL_'))}
    env.update(PROJECT_ROOT=str(ROOT),RUBRIC_EVAL_USE_RECOVERY_COPY='1',RUBRIC_EVAL_ONLY='1',CKPT_PATH=str(dest),
               RUBRIC_RECOVERY_SOURCE_CHECKPOINT=str(source/'global_step_1170'),RUBRIC_EVAL_STEP='1110')
    p=subprocess.run(['bash','-ec',code],env=env,capture_output=True,text=True)
    assert p.returncode==0,p.stdout+p.stderr
    assert f'RESULT=1|{dest}|{dest}/global_step_1110/actor|best' in p.stdout
    assert fixture.tree_bytes(source)==before and not any(p.is_symlink() for p in dest.rglob('*'))
    for changes in [dict(RUBRIC_EVAL_ONLY='0'),dict(RUBRIC_EVAL_STEP='1080')]:
        p=subprocess.run(['bash','-ec',code],env={**env,**changes},capture_output=True,text=True)
        assert p.returncode!=0
