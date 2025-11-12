model_name=$1
tasks=("humaneval_all" "mbpp_all" "gsm8k_cot" "minerva_math")
output_len=512
num_prompts=128

if [ "$model_name" == "qwen2.5_1.5b" ]; then
    model_path="/lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen2.5-1.5B-Base"
    served_model_name="qwen2.5_1.5b"
elif [ "$model_name" == "qwen3_8b" ]; then
    model_path="/lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B"
    served_model_name="qwen3_8b"
elif [ "$model_name" == "qwen3_eagle_8b" ]; then
    model_path="/lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B"
    served_model_name="qwen3_eagle_8b"
elif [ "$model_name" == "tidar_1.5b" ]; then
    model_path="/lustre/fsw/portfolios/nvr/users/jinliu/megatron_exp/tidar_1.5b_sglang"
    served_model_name="tidar_1.5b"
elif [ "$model_name" == "tidar_8b" ]; then
    model_path="/lustre/fsw/portfolios/nvr/users/jinliu/megatron_exp/tidar_8b_sglang"
    served_model_name="tidar_8b"
else
    echo "Invalid model name"
    exit 1
fi

for task in "${tasks[@]}"; do
    python -m sglang.bench_serving \
        --backend sglang-oai \
        --host 127.0.0.1 \
        --port 30000 \
        --request-rate 1 \
        --max-concurrency 1 \
        --seed 227 \
        --model $model_path \
        --served-model-name $served_model_name \
        --tokenizer /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
        --dataset-name sharegpt-single-turn \
        --dataset-path tidar_data/${task}.jsonl \
        --sharegpt-output-len $output_len \
        --num-prompts $num_prompts \
        --output-file tidar_data/benchmark_results/${model_name}_${task}.jsonl
done