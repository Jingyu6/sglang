from __future__ import annotations

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn

from sglang.srt.models.qwen3 import (
    Qwen3Config,
    Qwen3Model,
    Qwen3ForCausalLM,
    Pooler,
    PoolingType,
    ParallelLMHead,
    PPMissingLayer,
    add_prefix,
    get_layer_id,
    maybe_remap_kv_scale_name,
    default_weight_loader,
)


class TiDARForCausalLM(Qwen3ForCausalLM):
    """
    TiDAR model: identical architecture to Qwen3, different weights and default
    decoding behavior (driven by the TiDAR worker). We intentionally reuse the
    Qwen3 module stack so attention/backbone remain unchanged.
    """

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional["QuantizationConfig"] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)


# Make TiDAR the entry class for this file
EntryClass = TiDARForCausalLM


