local_dir=(
    "/openbayes/input/input0/checkpoints/qwen25_3B_unevolve_with_prediction/global_step_732"
)
#

# "base model tokenizer"
hf_model_path="/openbayes/input/input0/models/qwen25_3B"
base_model=$(basename $hf_model_path)


for dir in ${local_dir[@]}; do
    target_dir=$(dirname $dir)
    mkdir -p "${target_dir}/models"
    target_dir="${target_dir}/models"
    step_name=$(basename $dir)
    python ./scripts/model_merger.py \
        --backend fsdp \
        --hf_model_path $hf_model_path \
        --local_dir $dir/actor \
        --target_dir ${target_dir}/${base_model}-${step_name}
    python ./scripts/move_tokenizer.py \
        --source $hf_model_path \
        --destination ${target_dir}/${base_model}-${step_name}
done


