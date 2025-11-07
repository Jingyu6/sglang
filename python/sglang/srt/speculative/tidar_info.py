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

    def __post_init__(self):
        # If carrying tokens across steps, mark as draft; else mark as verify
        if self.send_tokens is not None:
            super().__init__(SpecInputType.EAGLE_DRAFT)
        else:
            # Reuse verify type so FlashInfer prefill wrappers accept custom_mask path
            super().__init__(SpecInputType.EAGLE_VERIFY)

    def get_spec_adjust_token_coefficient(self) -> Tuple[int, int]:
        return self.num_queries, self.num_queries

    def prepare_forward_batch(self, req_to_token_pool, batch, target_worker):
        bs = len(batch.seq_lens)
        device = batch.seq_lens.device

        batch.input_ids = self.draft_token
        batch.out_cache_loc = assign_extend_cache_locs_func(
            batch.req_pool_indices,
            req_to_token_pool.req_to_token,
            batch.seq_lens,
            batch.seq_lens + self.num_queries,
            bs,
            self.num_queries,
            device,
        )

        # Route through verify (prefill wrappers) and override positions
        batch.forward_mode = ForwardMode.TARGET_VERIFY
        batch.capture_hidden_mode = CaptureHiddenMode.NULL

        batch.spec_info = self
        forward_batch = ForwardBatch.init_new(batch, target_worker.model_runner)
        
        # skip attention backend init because later it will be initialized
        return forward_batch, bool(
            target_worker.model_runner.graph_runner
            and target_worker.model_runner.graph_runner.can_run(forward_batch)
        )

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

        kv_indices = torch.empty(
            paged_kernel_lens_sum + bs * self.num_queries, dtype=torch.int32, device=device
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
