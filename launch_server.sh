#!/bin/bash
export PYTHONPATH="./tidar_sglang:$PYTHONPATH"

python -m sglang.launch_server --config ./tidar_data/configs/${1}_config.yaml