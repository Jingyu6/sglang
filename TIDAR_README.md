
1. Base model

```
python -m sglang.bench_serving \
    --backend sglang-oai \
    --host 127.0.0.1 \
    --port 30000 \
    --request-rate 1 \
    --max-concurrency 1 \
    --model /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
    --served-model-name qwen3_8b \
    --tokenizer /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
    --dataset-name sharegpt \
    --dataset-path tidar_data/default_sharegpt.json \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/qwen3_out.json \
    --output-details
```

2. Eagle3

```
python -m sglang.bench_serving \
    --backend sglang-oai \
    --host 127.0.0.1 \
    --port 30000 \
    --request-rate 1 \
    --max-concurrency 1 \
    --model /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
    --served-model-name qwen3_eagle_8b \
    --tokenizer /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
    --dataset-name sharegpt \
    --dataset-path tidar_data/default_sharegpt.json \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/qwen3_eagle_out.json \
    --output-details
```

3. TiDAR
```
python -m sglang.bench_serving \
    --backend sglang-oai \
    --host 127.0.0.1 \
    --port 30000 \
    --request-rate 1 \
    --max-concurrency 1 \
    --model /lustre/fsw/portfolios/nvr/users/jinliu/megatron_exp/tidar_8b_sglang \
    --served-model-name tidar_8b \
    --tokenizer /lustre/fsw/portfolios/nvr/users/jinliu/public_models/Qwen3-8B \
    --dataset-name sharegpt \
    --dataset-path tidar_data/default_sharegpt.json \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/tidar.json \
    --output-details
```