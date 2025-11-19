# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys

sys.path.insert(0, "./tidar_sglang")

import dataclasses
from sglang import function, gen, Runtime
from sglang.srt.server_args import prepare_server_args, ServerArgs


MAX_TOKENS = 512

@function
def text_gen(s, prompt):
    s += prompt + gen(
        'response', 
        max_tokens=MAX_TOKENS, 
        ignore_eos=True, 
        stop=[],
        stop_token_ids=[],
        n=1
    )

if __name__ == "__main__":
    model_name = input("Enter the model name: ")
    enable_fp8 = input("Enable FP8? (y/n): ")
    assert model_name in [
        "tidar_1.5b",
        "tidar_8b",
        "qwen2.5_1.5b",
        "qwen3_8b", 
        "qwen3_eagle_8b",
    ], "Invalid model name"
    server_args = prepare_server_args(["--config", f"./tidar_data/configs/{model_name}_config.yaml"])
    # Only pass dataclass fields (exclude computed attrs like model_config)
    _field_names = {f.name for f in dataclasses.fields(ServerArgs)}
    _kwargs = {k: getattr(server_args, k) for k in _field_names}
    if enable_fp8 == "y":
        _kwargs["quantization"] = "fp8"
    runtime = Runtime(**_kwargs)

    while True:
        print('='*100)
        prompt = input("Enter a prompt: ")
        if prompt == "input": 
            print("Reading input from input.txt")
            with open("tmp.txt", "r") as f:
                prompt = f.read()
        response = text_gen.run(prompt=prompt, backend=runtime)
        print(response['response'])
        print('='*100)
