#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
export RUBRIC_JUDGE_BACKEND=local_api
export RUBRIC_JUDGE_LOCAL_API_BASE=http://127.0.0.1:8000/v1
unset RUBRIC_JUDGE_LOCAL_API_BASES
export RUBRIC_JUDGE_LOCAL_API_MODEL=qwen3_32B_instruct
export RUBRIC_JUDGE_GRANULARITY=single_rule
export RUBRIC_JUDGE_API_CONCURRENCY=4
export RUBRIC_JUDGE_STRICT_PARSE=1
export RUBRIC_GUARD_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export RUBRIC_TARGETED_REFUSAL_ENABLE=1
export RUBRIC_TARGETED_REFUSAL_ADV_CAP=0.05
export RUBRIC_TARGETED_REFUSAL_START_STEP=360
export RUBRIC_TARGETED_REFUSAL_INTERVAL=4
export RUBRIC_BENIGN_CONTRAST_ENABLE=0
export RUBRIC_EVOLVE_EVIDENCE_V5_ENABLE=0
export RUBRIC_RUN_DIR="${PRIMUS_OSS_WORKSPACE_DIR:?Primus OSS workspace mount required}/targeted_refusal_smoke"
mkdir -p "$RUBRIC_RUN_DIR"
model_dir="${JUDGE_MODEL_PATH:-/root/models/judge}"
if [[ ! -f "$model_dir/config.json" ]]; then
    modelscope download --model Qwen/Qwen3-32B --local_dir "$model_dir"
fi
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" --served-model-name qwen3_32B_instruct \
    --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.93 \
    --port 8000 > "$RUBRIC_RUN_DIR/server.log" 2>&1 &
judge_pid=$!
trap 'kill "$judge_pid" 2>/dev/null || true' EXIT
ready=0
for ((attempt=0; attempt<180; attempt++)); do
    kill -0 "$judge_pid" || { tail -n 60 "$RUBRIC_RUN_DIR/server.log"; exit 1; }
    if curl --max-time 3 -fsS http://127.0.0.1:8000/health > /dev/null; then
        ready=1
        break
    fi
    sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Judge startup timed out'; exit 1; }
python scripts/smoke_targeted_refusal.py --output "$RUBRIC_RUN_DIR/result.json"
