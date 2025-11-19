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


from __future__ import annotations

from typing import Optional

from sglang.srt.models.qwen2 import Qwen2Config, Qwen2ForCausalLM


class TiDARSmallForCausalLM(Qwen2ForCausalLM):
    """
    TiDAR small model: identical architecture to Qwen2, different weights and default
    decoding behavior (driven by the TiDAR worker). We intentionally reuse the
    Qwen2 module stack so attention/backbone remain unchanged.
    """

    def __init__(
        self,
        config: Qwen2Config,
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)


# Make TiDAR small the entry class for this file
EntryClass = TiDARSmallForCausalLM
