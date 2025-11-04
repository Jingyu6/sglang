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
from sglang.srt.mem_cache.common import get_last_loc
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
        self.trust_ar_ratio = 0.0
        assert 0 <= self.trust_ar_ratio <= 1, "trust_ar_ratio must be between 0 and 1"

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

        self._alloc_kv_slots(batch, block_size)

        verify_fb, can_graph = spec_input.prepare_for_v2_verify(
            self.target_worker.model_runner.req_to_token_pool,
            batch,
            self.target_worker,
        )
        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_fb,
            is_verify=True,
            skip_attn_backend_init=False,
        )
        logits_output = forward_out.logits_output
        next_token_ids = torch.argmax(logits_output.next_token_logits, dim=-1)

        # No acceptance at prefill draft stage; keep lengths unchanged
        accept_lens = torch.zeros_like(batch.seq_lens, dtype=torch.int32)

        empty_idx = torch.empty(0, dtype=torch.int64, device=self.device)
        empty_lens = torch.zeros_like(batch.seq_lens, dtype=torch.int32, device=self.device)
        self._free_kv_slots(batch, empty_idx, empty_lens)

        # Send B draft tokens (no KV cache) to the first decode step
        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=next_token_ids.to(torch.int64).contiguous().clone(),
            B=block_size,
        )

        return GenerationBatchResult(
            logits_output=base.logits_output, # dummy
            next_token_ids=base.next_token_ids, # dummy
            can_run_cuda_graph=base.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            allocate_lens=None,
        )

    def _alloc_kv_slots(self, batch: ModelWorkerBatch, slot_num: int):
        bs = len(batch.seq_lens)
        allocator = self.target_worker.model_runner.token_to_kv_pool_allocator
        req_to_token_pool = self.target_worker.model_runner.req_to_token_pool
        if self.page_size == 1:
            num_tokens = bs * slot_num
            alloc_indices = allocator.alloc(num_tokens)
            if alloc_indices is not None:
                offset = 0
                for i in range(bs):
                    seq_len_i = int(batch.seq_lens[i].item())
                    req_idx_i = int(batch.req_pool_indices[i].item())
                    vals = alloc_indices[offset : offset + slot_num].to(torch.int32)
                    req_to_token_pool.write((req_idx_i, slice(seq_len_i, seq_len_i + slot_num)), vals)
                    offset += slot_num
        else:
            # Paged allocation using alloc_extend
            prefix_lens = batch.seq_lens
            seq_lens_next = batch.seq_lens + slot_num
            prefix_lens_cpu = batch.seq_lens_cpu if batch.seq_lens_cpu is not None else batch.seq_lens.cpu()
            seq_lens_next_cpu = prefix_lens_cpu + slot_num
            last_loc = get_last_loc(
                req_to_token_pool.req_to_token,
                batch.req_pool_indices,
                prefix_lens,
            )
            alloc_indices = allocator.alloc_extend(
                prefix_lens,
                prefix_lens_cpu,
                seq_lens_next,
                seq_lens_next_cpu,
                last_loc,
                bs * slot_num,
            )
            if alloc_indices is not None:
                offset = 0
                for i in range(bs):
                    seq_len_i = int(batch.seq_lens[i].item())
                    req_idx_i = int(batch.req_pool_indices[i].item())
                    vals = alloc_indices[offset : offset + slot_num].to(torch.int32)
                    req_to_token_pool.write((req_idx_i, slice(seq_len_i, seq_len_i + slot_num)), vals)
                    offset += slot_num

    def _free_kv_slots(self, batch: ModelWorkerBatch, accept_index: torch.Tensor, accept_lens: torch.Tensor):
        bs = len(batch.seq_lens)
        total_slots = int(batch.out_cache_loc.shape[0])
        if total_slots == 0 or bs == 0:
            return
        slots_per_req = total_slots // bs
        page_size = self.page_size

        # Short-circuit: nothing accepted (e.g., prefill draft cleanup). Free all and skip compaction.
        if accept_index.numel() == 0:
            to_free_all = batch.out_cache_loc[batch.out_cache_loc > 0]
            if to_free_all.numel() > 0:
                self.target_worker.model_runner.token_to_kv_pool_allocator.free(to_free_all)
            # Clear out_cache_loc; mapping cleanup happens below
            batch.out_cache_loc = batch.out_cache_loc[:0]
        else:
            if page_size == 1:
                evict_mask = torch.full((bs * slots_per_req,), True, dtype=torch.bool, device=self.device)
                evict_mask[accept_index] = False
                to_free = batch.out_cache_loc[evict_mask]
                to_free = to_free[to_free > 0]
                if to_free.numel() > 0:
                    self.target_worker.model_runner.token_to_kv_pool_allocator.free(to_free)
                batch.out_cache_loc = batch.out_cache_loc[accept_index]
            else:
                if slots_per_req == 1:
                    evict_mask = torch.full((bs * slots_per_req,), True, dtype=torch.bool, device=self.device)
                    evict_mask[accept_index] = False
                    align_evict_mask_to_page_size[(bs,)](
                        batch.seq_lens, evict_mask, page_size, slots_per_req, next_power_of_2(slots_per_req)
                    )
                    to_free = batch.out_cache_loc[evict_mask]
                    to_free = to_free[to_free > 0]
                    if to_free.numel() > 0:
                        self.target_worker.model_runner.token_to_kv_pool_allocator.free(to_free)
                    batch.out_cache_loc = batch.out_cache_loc[accept_index]
                else:
                    src_loc, tgt_loc, to_free_num_slots = get_src_tgt_cache_loc(
                        batch.seq_lens, batch.out_cache_loc, accept_index, accept_lens, slots_per_req, page_size
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
                        slots_per_req,
                        next_power_of_2(slots_per_req),
                        next_power_of_2(bs),
                    )
                    # Move first, then free leftover slots to avoid writing into freed memory
                    self.target_worker.model_runner.token_to_kv_pool_allocator.get_kvcache().move_kv_cache(
                        tgt_loc, src_loc
                    )
                    to_free = to_free_slots[to_free_slots > 0]
                    if to_free.numel() > 0:
                        self.target_worker.model_runner.token_to_kv_pool_allocator.free(to_free)
                    batch.out_cache_loc = tgt_loc

        # Update req_to_token mapping for accepted tokens and clear rejected ones
        pre_seq_lens = batch.seq_lens.clone()
        if bs > 0:
            req_to_token_pool = self.target_worker.model_runner.req_to_token_pool
            # Accepted locs are grouped per request in batch.out_cache_loc
            offset = 0
            for i in range(bs):
                acc_len_i = int(accept_lens[i].item())
                if acc_len_i > 0:
                    pos = torch.arange(
                        int(pre_seq_lens[i].item()),
                        int(pre_seq_lens[i].item()) + acc_len_i,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    vals = batch.out_cache_loc[offset : offset + acc_len_i].to(torch.int32)
                    req_to_token_pool.write((int(batch.req_pool_indices[i].item()), pos), vals)
                # Clear mappings for rejected draft slots
                rej_start = int(pre_seq_lens[i].item()) + acc_len_i
                rej_end = int(pre_seq_lens[i].item()) + slots_per_req
                if rej_end > rej_start:
                    rej_pos = torch.arange(
                        rej_start,
                        rej_end,
                        dtype=torch.int64,
                        device=self.device,
                    )
                    req_to_token_pool.write((int(batch.req_pool_indices[i].item()), rej_pos), torch.zeros_like(rej_pos, dtype=torch.int32))
                offset += acc_len_i

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

        self._alloc_kv_slots(batch, B)

        verify_fb, can_graph = spec_input.prepare_for_v2_verify(
            self.target_worker.model_runner.req_to_token_pool,
            batch,
            self.target_worker,
        )
        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_fb,
            is_verify=True,
            skip_attn_backend_init=False,
        )

        logits_output = forward_out.logits_output
        # first merge the logits
        bs = len(batch.seq_lens)
        logits = logits_output.next_token_logits.view(bs, block_size + 1, block_size, -1)
        # TODO: do the logits mixing here
        # logits[:, 1] = logits[:, 0].view(-1) * self.trust_ar_ratio + logits[:, 1:, 0].view(-1) * (1 - self.trust_ar_ratio)
        # sampling here
        verify_tokens = prev_draft_tokens.view(bs, block_size) # [bs, block_size]
        new_draft_tokens = torch.argmax(logits, dim=-1)        # [bs, block_size + 1, block_size]

        # TiDAR verification
        assert bs == 1, "TiDAR only supports batch size 1 for now"
        accept_cnt = 1
        select_draft_tokens = new_draft_tokens[0, 1] # we default to the first new draft set

        while accept_cnt < block_size:
            if new_draft_tokens[0, 0, accept_cnt - 1].item() != verify_tokens[0, accept_cnt].item():
                break
            select_draft_tokens = new_draft_tokens[0, accept_cnt + 1]
            accept_cnt += 1

        # Keep only the first accept_cnt queries' KV; evict the rest
        accept_lens = torch.full((bs,), accept_cnt, dtype=torch.int32, device=self.device)

        # Indices to keep per sequence: [0..accept_cnt-1]
        row_offsets = torch.arange(bs, device=self.device) * B
        accept_index = torch.cat([
            row_offsets[i] + torch.arange(0, accept_lens[i].item(), device=self.device)
            for i in range(bs)
        ])

        self._free_kv_slots(batch, accept_index, accept_lens)

        # Advance lengths
        batch.seq_lens.add_(accept_lens)

        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=select_draft_tokens.view(-1).to(torch.int64).contiguous().clone(),
            B=block_size,
        )

        accept_tokens = verify_tokens[0, :accept_cnt].view(-1)
        total_accepted = int(accept_lens.sum().item()) - bs # - bs to make the metric consistent
        assert total_accepted >= 0, "Total accepted tokens must be non-negative"

        # print(f"prev_draft_tokens: {prev_draft_tokens}")
        # print(f"new_draft_tokens: {new_draft_tokens}")
        # print(f"accept_tokens: {accept_tokens}")
        # print("================================================")

        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=accept_tokens,
            num_accepted_tokens=total_accepted,
            can_run_cuda_graph=can_graph,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            allocate_lens=None,
        )