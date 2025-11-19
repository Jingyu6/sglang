# TIDAR: Think in Diffusion, Talk in Autoregression


<p align="center">
        📄 <a href="https://arxiv.org/abs/2511.08923v1">Paper</a> 
        &nbsp&nbsp 
        📜 <a href="">Page</a>
</p>

### Introduction

### Setup

#### Environment Setup
Our implementation is based on SGLang with this specific [commit](https://github.com/sgl-project/sglang/pull/12548). To setup the environment, we recommend using Docker image with SGLang:
```bash
docker pull lmsysorg/sglang:latest
```
Install the latest dependencies in SGLang if needed:
```bash
cd tidar_sglang
pip install -e "python"
```

#### Supported models
<strong>We will release TiDAR 1.5B and 8B models as soon as we can</strong>, and our benchmarking on SGLang supports the following variants: 
```
tidar_1.5b
tidar_8b
qwen2.5_1.5b
qwen3_8b
qwen3_eagle_8b
```

#### Model checkpoint downloads
Change the `<work_dir>` to be the directory where model weights are saved. We recommend downloading the HF weights using `huggingface-cli`. 

You could download each models using the following commands:
```bash
# Qwen2.5 1.5b
huggingface-cli download Qwen/Qwen2.5-1.5B --local-dir <work_dir>public_models/qwen2.5_1.5b --local-dir-use-symlinks False
# Qwen3 8b
huggingface-cli download Qwen/Qwen3-8B --local-dir <work_dir>public_models/qwen3_8b --local-dir-use-symlinks False
# Qwen3 8b eagle
huggingface-cli download Tengyunw/qwen3_8b_eagle3 --local-dir <work_dir>public_models/qwen3_eagle_8b --local-dir-use-symlinks False
```

#### Benchmark data preparation
To run the latency benchmarking, first generate real query data (TiDAR's performance depends on the input). 
```
bash tidar_data/gen_all_data.sh
```

#### Generation configs
All model configs are specified in `tidar_data/configs`. And our implementation requires: 
```yaml
max-running-requests: 1
disable-overlap-schedule: true
disable-radix-cache: true
chunked-prefill-size: -1
```
CUDA graph can be turned on or off by setting: 
```yaml
# use one of the following
cuda-graph-max-bs: 1
disable-cuda-graph: true
```
For sampling parameters, please refer to `tidar_sglang/python/sglang/bench_serving.py` and `run_interactive.py`. 

### Benchmarking

#### Test model generation with interactive session
We can start by testing the generation output of different models using interative sessions by running: 
```bash
python run_interative.py
```
When prompted, enter the model name and whether to use quantization

#### Server API benchmarking
First launch the server using the following command:
```bash
bash launch_server.sh <model_name>
```

And then launch the client with the following commands over all tasks: 
We can get all benchmark numbers by running
```bash
bash launch_client.sh <model_name>
```

### Citation
If you find TiDAR to be useful, please consider to star the repo and cite the paper:
```bibtex
@misc{liu2025tidarthinkdiffusiontalk,
      title={TiDAR: Think in Diffusion, Talk in Autoregression}, 
      author={Jingyu Liu and Xin Dong and Zhifan Ye and Rishabh Mehta and Yonggan Fu and Vartika Singh and Jan Kautz and Ce Zhang and Pavlo Molchanov},
      year={2025},
      eprint={2511.08923},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.08923}, 
}
```

### License
Copyright © 2025, NVIDIA Corporation. All rights reserved.

This work is made available under the Apache 2.0 License. 