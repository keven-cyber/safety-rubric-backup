#!/usr/bin/env bash
# Validation only: no policy model, optimizer, training launcher or data rewrite.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
OUT="${PRIMUS_OSS_WORKSPACE_DIR:?}/existing_answer_validation"
mkdir -p "$OUT/services" || exit 1
export RUBRIC_JUDGE_BACKEND=local_api RUBRIC_JUDGE_LOCAL_API_BASE=http://127.0.0.1:8000/v1
export RUBRIC_JUDGE_LOCAL_API_MODEL=qwen3_32B_instruct RUBRIC_JUDGE_API_CONCURRENCY=4
export RUBRIC_JUDGE_GRANULARITY=all_rules RUBRIC_JUDGE_GUIDED_OUTPUT=1
export RUBRIC_JUDGE_ANSWER_GROUNDING_ENABLE=1 RUBRIC_JUDGE_DROP_FAILED=0
export RUBRIC_TARGETED_REFUSAL_ENABLE=1 RUBRIC_TARGETED_REFUSAL_ADV_CAP=0.05
export RUBRIC_TARGETED_REFUSAL_START_STEP=360 RUBRIC_TARGETED_REFUSAL_INTERVAL=4
export RUBRIC_BENIGN_CONTRAST_ENABLE=0 RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE=0
export RUBRIC_RUN_DIR="$OUT" SAFETY_GUARD_API_BASE=http://127.0.0.1:8001/v1
export SAFETY_GUARD_MODEL=qwen3_guard_gen_8b
MODEL_ROOT="${PRIMUS_OSS_WORKSPACE_DIR}/validation_models"
export RUBRIC_JUDGE_TOKENIZER_DIR="$MODEL_ROOT/judge"
mkdir -p "$MODEL_ROOT" || exit 1
service_pid=''
stop_service() {
    if [ -n "$service_pid" ]; then
        kill -- "-$service_pid" 2>/dev/null || true
        wait "$service_pid" 2>/dev/null || true
        service_pid=''
    fi
}
trap stop_service EXIT
wait_service() {
    for ((i=0; i<240; i++)); do
        if curl -fsS "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1; then return 0; fi
        kill -0 "$service_pid" 2>/dev/null || return 1
        sleep 5
    done
    return 1
}
run_mode() {
    local mode="$1" model="$2" served="$3" port="$4"
    mkdir -p "$OUT/$mode"
    if [ "$mode" = judge ]; then
        if ! python3 scripts/validate_existing_answer_judge.py grammar --output "$OUT/protocol"; then
            python3 - "$OUT" <<'PY'
import sys
from pathlib import Path
from Evaluation.safety.json_evidence import read_json_complete, write_json_atomic
root = Path(sys.argv[1])
report = read_json_complete(root/'protocol/result.json')
report.update(passed=False, status='failed', real_judge_executed=False, stage='grammar_preflight')
write_json_atomic(root/'judge/result.json', report)
PY
            return 1
        fi
    fi
    if ! modelscope download --model "$model" --local_dir "$MODEL_ROOT/$mode" >"$OUT/services/${mode}_download.log" 2>&1; then
        printf '{"status":"failed","passed":false,"stage":"model_download","optimizer_updates":0}\n' >"$OUT/$mode/result.json"
        return 1
    fi
    if [ "$mode" = judge ]; then
        CUDA_VISIBLE_DEVICES=0 setsid python3 scripts/judge_replay_server.py \
            --model "$MODEL_ROOT/$mode" --served "$served" --port "$port" \
            --manifest "$OUT/services/judge_profile.json" >"$OUT/services/${mode}_vllm.log" 2>&1 &
    else
        CUDA_VISIBLE_DEVICES=0 setsid python3 -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ROOT/$mode" --served-model-name "$served" --port "$port" \
            --tensor-parallel-size 1 --max-model-len 16384 --gpu-memory-utilization 0.93 \
            --enable-prefix-caching >"$OUT/services/${mode}_vllm.log" 2>&1 &
    fi
    service_pid=$!
    export RUBRIC_REPLAY_SERVER_PID="$service_pid"
    if wait_service "$port"; then
        python3 scripts/validate_existing_answer_judge.py "$mode" --output "$OUT/$mode"
        local status=$?
    else
        printf '{"status":"failed","passed":false,"stage":"model_service","optimizer_updates":0}\n' >"$OUT/$mode/result.json"
        local status=1
    fi
    stop_service
    # Allow CUDA subprocesses in this validation-only process group to release VRAM.
    sleep 10
    return "$status"
}
run_mode judge Qwen/Qwen3-32B qwen3_32B_instruct 8000
judge_status=$?
# Guard evidence is still collected even when the Judge gate fails.
run_mode guard Qwen/Qwen3Guard-Gen-8B qwen3_guard_gen_8b 8001
guard_status=$?
python3 - "$OUT" "$judge_status" "$guard_status" <<'PY'
import json,sys
from pathlib import Path
from Evaluation.safety.json_evidence import read_json_complete, write_json_atomic
root=Path(sys.argv[1]); reports={}
for mode in ('judge','guard'):
    reports[mode]=read_json_complete(root/mode/'result.json')
passed=all(r.get('passed') is True for r in reports.values()) and sys.argv[2:] == ['0','0']
write_json_atomic(root/'result.json',dict(passed=passed,status='passed' if passed else 'failed',
    optimizer_updates=0,reports=reports))
sys.exit(0 if passed else 1)
PY
