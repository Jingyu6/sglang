### Generate data for benchmarking
```
bash tidar_data/gen_all_data.sh
```

### Supported models
```
tidar_1.5b
tidar_8b
qwen2.5_1.5b
qwen3_8b
qwen3_eagle_8b
```

### Launch server
```
bash tidar_launch_server <model_name>
```

### Launch client
We can get all benchmark numbers by running
```
bash tidar_launch_client <model_name>
```

### Run interative
```
python tidar_interative.py
```
When prompted, enter the model name and whether to use quantization
