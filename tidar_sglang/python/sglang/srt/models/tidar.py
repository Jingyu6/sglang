from __future__ import annotations

from typing import Optional

from sglang.srt.models.qwen3 import Qwen3Config, Qwen3ForCausalLM


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
