from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ModelWorkerBatch, ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.tidar_info import TiDARInput
from sglang.srt.speculative.tidar_utils import (
    build_tidar_positions_and_mask_prefill,
    build_tidar_positions_and_mask_decode,
)
from sglang.srt.speculative.eagle_info import (
    get_src_tgt_cache_loc,
    get_target_cache_loc,
    align_evict_mask_to_page_size,
)
from sglang.srt.utils.common import next_power_of_2


class TiDARWorker(BaseSpecWorker):
    def __init__(
        self,
        server_args,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.device = server_args.device
        self.gpu_id = gpu_id
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        # TiDAR parameters
        self.speculative_tidar_b = server_args.speculative_tidar_b
        self.mask_token_id = 151662

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        # TiDAR does not use a separate draft worker
        return None

    def clear_cache_pool(self):
        pass

    def forward_batch_generation(self, model_worker_batch: ModelWorkerBatch | ScheduleBatch):
        if isinstance(model_worker_batch, ScheduleBatch):
            model_worker_batch = model_worker_batch.get_model_worker_batch()

        if model_worker_batch.forward_mode.is_decode():
            return self._decode_step(model_worker_batch)
        else:
            # Run normal target prefill first (embedding/residual states, etc.)
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.FULL
            base = self.target_worker.forward_batch_generation(model_worker_batch)
            # Then emit B tokens using TiDAR prefill with custom mask
            return self._prefill_draft_only(model_worker_batch, base)

    def _prefill_draft_only(self, batch: ModelWorkerBatch, base: GenerationBatchResult):
        block_size = self.speculative_tidar_b
        draft_token, positions, custom_mask = build_tidar_positions_and_mask_prefill(
            seq_lens=batch.seq_lens, block_size=block_size, mask_token_id=self.mask_token_id, device=self.device
        )
        spec_input = TiDARInput(draft_token, positions, custom_mask, num_queries=block_size)
        self.target_worker.model_runner.attn_backend.num_draft_tokens = block_size
        verify_fb, can_graph = spec_input.prepare_for_v2_verify(
            self.target_worker.model_runner.req_to_token_pool,
            batch,
            self.target_worker,
        )
        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_fb,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_out.logits_output
        next_token_ids = torch.argmax(logits_output.next_token_logits, dim=-1)

        # Evict KV cache for the newly allocated B tokens (do not persist KV at prefill draft)
        self.target_worker.model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        # No acceptance at prefill draft stage; keep lengths unchanged
        accept_lens = torch.zeros_like(batch.seq_lens, dtype=torch.int32)

        # Send B draft tokens (no KV cache) to the first decode step
        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=next_token_ids,
            B=block_size,
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            allocate_lens=None,
        )

    def _decode_step(self, batch: ModelWorkerBatch):
        block_size = self.speculative_tidar_b
        B = block_size * (block_size + 1)

        # get the previous draft tokens
        assert batch.spec_info is not None, "Missing draft tokens from the previous step"
        prev_draft_tokens = batch.spec_info.send_tokens

        draft_token, positions, custom_mask = build_tidar_positions_and_mask_decode(
            seq_lens=batch.seq_lens, 
            block_size=block_size, 
            prev_draft_tokens=prev_draft_tokens, 
            mask_token_id=self.mask_token_id, 
            device=self.device
        )
        spec_input = TiDARInput(draft_token, positions, custom_mask, num_queries=B)
        self.target_worker.model_runner.attn_backend.num_draft_tokens = B
        verify_fb, can_graph = spec_input.prepare_for_v2_verify(
            self.target_worker.model_runner.req_to_token_pool,
            batch,
            self.target_worker,
        )
        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_fb,
            is_verify=True,
            skip_attn_backend_init=True,
        )
        logits_output = forward_out.logits_output
        print(logits_output.next_token_logits)
        print(logits_output.next_token_logits.shape)
        exit()
        # TiDAR verification
        tokens = torch.argmax(logits_output.next_token_logits, dim=-1)

        # Determine N (<= B) accepted per sequence; fallback to B if not provided
        N = getattr(batch.sampling_info, "max_new_tokens_per_step", block_size)
        if isinstance(N, int):
            accept_lens = torch.full_like(batch.seq_lens, N, dtype=torch.int32)
        else:
            accept_lens = N.to(dtype=torch.int32, device=self.device)

        # Build accept_index for first N per sequence
        bs = len(batch.seq_lens)
        row_offsets = torch.arange(bs, device=self.device) * B
        accept_index = torch.cat(
            [row_offsets[i] + torch.arange(0, accept_lens[i].item(), device=self.device) for i in range(bs)]
        )

        # Evict/compact KV for unaccepted tokens
        page_size = self.page_size
        if page_size == 1:
            evict_mask = torch.full((bs * B,), True, dtype=torch.bool, device=self.device)
            evict_mask[accept_index] = False
            self.target_worker.model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
            batch.out_cache_loc = batch.out_cache_loc[accept_index]
        else:
            if block_size == 1:
                evict_mask = torch.full((bs * B,), True, dtype=torch.bool, device=self.device)
                evict_mask[accept_index] = False
                align_evict_mask_to_page_size[(bs,)](
                    batch.seq_lens, evict_mask, page_size, B, next_power_of_2(B)
                )
                self.target_worker.model_runner.token_to_kv_pool_allocator.free(batch.out_cache_loc[evict_mask])
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
            else:
                src_loc, tgt_loc, to_free_num_slots = get_src_tgt_cache_loc(
                    batch.seq_lens, batch.out_cache_loc, accept_index, accept_lens, B, page_size
                )
                to_free_slots = torch.empty(
                    (to_free_num_slots.sum().item(),), dtype=torch.int64, device=self.device
                )
                get_target_cache_loc[(bs,)](
                    tgt_loc,
                    to_free_slots,
                    accept_lens,
                    to_free_num_slots,
                    batch.out_cache_loc,
                    B,
                    next_power_of_2(B),
                    next_power_of_2(bs),
                )
                self.target_worker.model_runner.token_to_kv_pool_allocator.free(to_free_slots)
                self.target_worker.model_runner.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                    tgt_loc, src_loc
                )
                batch.out_cache_loc = tgt_loc

        # Advance lengths
        batch.seq_lens.add_(accept_lens)

        # Send B tokens to the next decode step (placeholder selection: first B per seq)
        tokens_2d = tokens.view(bs, B)
        to_send = tokens_2d[:, :block_size].contiguous().view(-1)
        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=to_send,
            B=block_size,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=tokens,
            can_run_cuda_graph=can_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            allocate_lens=None,
        )


