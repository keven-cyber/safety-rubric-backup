#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
用法:
  bash scripts/merge_one.sh --hf_model_path <path> --local_dir <ckpt_dir> [--local_dir <ckpt_dir2> ...]

说明:
  --hf_model_path   基座模型路径（用于结构与 tokenizer）
  --local_dir       checkpoint 路径（通常是 .../global_step_xxx）

示例:
  bash scripts/merge_one.sh \
    --hf_model_path /openbayes/input/input0/models/qwen25_3B \
    --local_dir /openbayes/input/input0/checkpoints/qwen25_3B_unevolve_with_prediction/global_step_732
EOF
}

hf_model_path=""
local_dir=()

while [[ $# -gt 0 ]]; do
    case "$1" in
    --hf_model_path)
        hf_model_path="${2:-}"
        shift 2
        ;;
    --local_dir)
        local_dir+=("${2:-}")
        shift 2
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "未知参数: $1"
        usage
        exit 1
        ;;
    esac
done

if [[ -z "${hf_model_path}" ]]; then
    echo "错误: 必须提供 --hf_model_path"
    usage
    exit 1
fi

if [[ ${#local_dir[@]} -eq 0 ]]; then
    echo "错误: 至少提供一个 --local_dir"
    usage
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
base_model="$(basename "${hf_model_path}")"

for dir in "${local_dir[@]}"; do
    target_dir="$(dirname "${dir}")"
    mkdir -p "${target_dir}/models"
    target_dir="${target_dir}/models"
    step_name="$(basename "${dir}")"
    out_dir="${target_dir}/${base_model}-${step_name}"

    if [[ -d "${dir}/actor" ]]; then
        actor_dir="${dir}/actor"
    elif [[ "$(basename "${dir}")" == "actor" ]]; then
        actor_dir="${dir}"
    else
        echo "错误: 未找到 actor 目录。请传入 global_step_xxx 或 actor 目录。当前: ${dir}"
        exit 1
    fi

    python "${REPO_ROOT}/scripts/model_merger.py" \
        --backend fsdp \
        --hf_model_path "${hf_model_path}" \
        --local_dir "${actor_dir}" \
        --target_dir "${out_dir}"
    python "${REPO_ROOT}/scripts/move_tokenizer.py" \
        --source "${hf_model_path}" \
        --destination "${out_dir}"
    echo "合并完成: ${out_dir}"
done

