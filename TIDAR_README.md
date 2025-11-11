### Generate data for benchmarking
```
bash tidar_data/gen_all_data.sh
```

### Launch server
```
python -m sglang.launch_server --config [qwen3 | qwen3_eagle | tidar]_8b_config.yaml
```

### Benchmark performance

use `sharegpt-single-turn` as the `dataset-name` for our custom dumped data. 

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
    --dataset-name sharegpt-single-turn \
    --dataset-path tidar_data/default_sharegpt.jsonl \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/qwen3_out.jsonl
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
    --dataset-name sharegpt-single-turn \
    --dataset-path tidar_data/default_sharegpt.jsonl \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/qwen3_eagle_out.jsonl
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
    --dataset-name sharegpt-single-turn \
    --dataset-path tidar_data/default_sharegpt.jsonl \
    --num-prompts 128 \
    --sharegpt-output-len 512 \
    --output-file tidar_data/benchmark_results/tidar.jsonl
```