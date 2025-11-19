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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.spec_info import SpecInput, SpecInputType
from sglang.srt.speculative.eagle_info_v2 import assign_extend_cache_locs_func
from sglang.srt.layers.attention.utils import create_flashinfer_kv_indices_triton


@dataclass
class TiDARInput(SpecInput):
    # Verify-mode fields (prefill/verify attention)
    draft_token: Optional[torch.Tensor]
    positions: Optional[torch.Tensor]
    custom_mask: Optional[torch.Tensor]
    num_queries: int
    # Draft-carrier fields (handoff between steps)
    send_tokens: Optional[torch.Tensor] = None  # flattened length = bs * B
    B: Optional[int] = None
    # Extra fields to mirror EAGLE verify inputs for overlap/cuda-graph slicing
    draft_token_num: Optional[int] = None
    seq_lens_cpu: Optional[torch.Tensor] = None
    seq_lens_sum: Optional[int] = None
    capture_hidden_mode: CaptureHiddenMode = CaptureHiddenMode.NULL

    def __post_init__(self):
        # If carrying tokens across steps, mark as draft; else mark as verify
        if self.send_tokens is not None:
            super().__init__(SpecInputType.EAGLE_DRAFT)
        else:
            # Reuse verify type so FlashInfer prefill wrappers accept custom_mask path
            super().__init__(SpecInputType.EAGLE_VERIFY)
        # Keep draft_token_num aligned with num_queries to satisfy shared utilities
        if self.draft_token_num is None:
            self.draft_token_num = self.num_queries

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_queries, self.num_queries

    # Consumed by FlashInferIndicesUpdaterPrefill.call_begin_forward
    def generate_attn_arg_prefill(
        self,
        req_pool_indices: torch.Tensor,
        paged_kernel_lens: torch.Tensor,
        paged_kernel_lens_sum: int,
        req_to_token: torch.Tensor,
    ):
        device = req_pool_indices.device
        bs = len(req_pool_indices)

        # qo_indptr: grouped by request, each with num_queries
        qo_indptr = torch.arange(
            0, (bs + 1) * self.num_queries, step=self.num_queries, dtype=torch.int32, device=device
        )

        # cum_kv_seq_len is kv_indptr for paged KV cache
        cum_kv_seq_len = torch.zeros((bs + 1,), dtype=torch.int32, device=device)
        paged_kernel_lens = paged_kernel_lens + self.num_queries
        cum_kv_seq_len[1:] = torch.cumsum(paged_kernel_lens, dim=0)

        # # Validate custom_mask layout and dtype if provided: sum_i Q * (seq_len[i] + Q)
        # if self.custom_mask is not None:
        #     q_per_seq = (qo_indptr[1:] - qo_indptr[:-1]).to(torch.int64)
        #     kv_len_per_seq = (cum_kv_seq_len[1:] - cum_kv_seq_len[:-1]).to(torch.int64)
        #     expected = int((q_per_seq * kv_len_per_seq).sum().item())
        #     assert (
        #         self.custom_mask.numel() == expected
        #     ), f"TiDAR custom_mask size mismatch: got {self.custom_mask.numel()}, expected {expected}"
        #     # Accept both bool and uint8; normalize to uint8 for FlashInfer robustness
        #     if self.custom_mask.dtype is torch.bool:
        #         self.custom_mask = self.custom_mask.to(torch.uint8)
        #     assert (
        #         self.custom_mask.is_cuda
        #     ), "TiDAR custom_mask must be on CUDA device"
        #     self.custom_mask = self.custom_mask.contiguous()

        # Allocate kv_indices exactly as needed by cum_kv_seq_len to avoid any size drift
        total_kv_len = int(cum_kv_seq_len[-1].item())
        kv_indices = torch.empty(
            total_kv_len, dtype=torch.int32, device=device
        )
        create_flashinfer_kv_indices_triton[(bs,)](
            req_to_token,
            req_pool_indices,
            paged_kernel_lens,
            cum_kv_seq_len,
            None,
            kv_indices,
            req_to_token.shape[1],
        )
        return kv_indices, cum_kv_seq_len, qo_indptr, self.custom_mask
