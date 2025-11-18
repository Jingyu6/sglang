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
