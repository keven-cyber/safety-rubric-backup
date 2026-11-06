#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export RUBRIC_JUDGE_BACKEND=local_api
export RUBRIC_JUDGE_LOCAL_API_BASE=http://127.0.0.1:8000/v1
export RUBRIC_JUDGE_LOCAL_API_MODEL=qwen3_32B_instruct
export RUBRIC_JUDGE_GRANULARITY=single_rule
export RUBRIC_JUDGE_API_CONCURRENCY=4
export RUBRIC_JUDGE_STRICT_PARSE=1
export RUBRIC_GUARD_ENABLE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
output_dir="${PRIMUS_OSS_WORKSPACE_DIR:?Primus OSS workspace mount required}/calibrated_judge_smoke"
mkdir -p "$output_dir"
model_dir="${JUDGE_MODEL_PATH:-/root/models/judge}"
if [[ ! -f "$model_dir/config.json" ]]; then
    modelscope download --model Qwen/Qwen3-32B --local_dir "$model_dir"
fi
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
    --model "$model_dir" --served-model-name qwen3_32B_instruct \
    --tensor-parallel-size 1 --max-model-len 4096 --gpu-memory-utilization 0.93 \
    --port 8000 > "$output_dir/server.log" 2>&1 &
judge_pid=$!
trap 'kill "$judge_pid" 2>/dev/null || true' EXIT
ready=0
for ((attempt=0; attempt<180; attempt++)); do
    kill -0 "$judge_pid" || { tail -n 60 "$output_dir/server.log"; exit 1; }
    if curl --max-time 3 -fsS http://127.0.0.1:8000/health > /dev/null; then
        ready=1
        break
    fi
    sleep 5
done
[[ "$ready" == 1 ]] || { echo 'Judge startup timed out'; exit 1; }
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python scripts/smoke_calibrated_judge.py --output "$output_dir/result.json"
