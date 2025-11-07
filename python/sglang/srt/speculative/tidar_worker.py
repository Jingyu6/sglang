from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.mem_cache.common import alloc_token_slots
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_utils import assign_draft_cache_locs, next_power_of_2
from sglang.srt.speculative.tidar_info import TiDARInput
from sglang.srt.speculative.tidar_utils import (
    build_tidar_positions_and_mask_decode,
    build_tidar_positions_and_mask_prefill,
)


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
        assert self.page_size == 1, "TiDAR only supports page size 1 for now"

        # for API consistency
        self.num_new_pages_per_topk = torch.empty((), dtype=torch.int64, device=self.device)
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        # TiDAR does not use a separate draft worker
        return None

    def clear_cache_pool(self):
        pass

    def forward_batch_generation(self, batch: ScheduleBatch):
        if batch.forward_mode.is_decode():
            return self._decode_step(batch)
        else:
            # Run normal target prefill first (embedding/residual states, etc.)
            model_worker_batch = batch.get_model_worker_batch()
            model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
            base = self.target_worker.forward_batch_generation(model_worker_batch)
            # Then emit B tokens using TiDAR prefill with custom mask
            return self._prefill_draft_only(batch, base)

    def _prefill_draft_only(self, batch: ScheduleBatch, base: GenerationBatchResult):
        block_size = self.speculative_tidar_b
        draft_token, positions, custom_mask = build_tidar_positions_and_mask_prefill(
            seq_lens=batch.seq_lens_cpu, 
            block_size=block_size, 
            mask_token_id=self.mask_token_id, 
            device=self.device
        )
        spec_input = TiDARInput(draft_token, positions, custom_mask, num_queries=block_size)
        self.target_worker.model_runner.attn_backend.num_draft_tokens = block_size

        # alloc KV slots
        out_cache_loc = alloc_token_slots(
            batch.tree_cache, 
            block_size
        )

        assign_draft_cache_locs[(1, )] (
            batch.req_pool_indices, 
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens, 
            self.num_new_pages_per_topk,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            1, 
            block_size,
            self.page_size, 
            1, 
            next_power_of_2(block_size)
        )

        batch.out_cache_loc = out_cache_loc
        batch.return_hidden_states = False
        # get model worker batch
        model_worker_batch = batch.get_model_worker_batch()
        # get the forward batch
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        model_worker_batch.input_ids = draft_token
        model_worker_batch.spec_info = spec_input
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)

        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=forward_batch,
            is_verify=True,
            skip_attn_backend_init=False,
        )

        logits_output = forward_out.logits_output
        next_token_ids = torch.argmax(logits_output.next_token_logits, dim=-1)
        # print(f"next_token_ids: {next_token_ids}")
        # print("================================================")

        # clean all cache
        batch.tree_cache.token_to_kv_pool_allocator.free(batch.out_cache_loc)
        
        # Send B draft tokens (no KV cache) to the first decode step
        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=next_token_ids.to(torch.int32).contiguous().clone(),
            B=block_size,
        )

        return GenerationBatchResult(
            logits_output=base.logits_output, # dummy
            next_token_ids=base.next_token_ids, # dummy
            can_run_cuda_graph=base.can_run_cuda_graph,
            next_draft_input=next_draft_input,
            accept_lens=None,
            allocate_lens=None,
        )

    def _decode_step(self, batch: ScheduleBatch):
        max_new_tokens = batch.reqs[0].sampling_params.max_new_tokens
        max_extra_tokens = max_new_tokens - len(batch.reqs[0].output_ids)

        block_size = self.speculative_tidar_b
        B = block_size * (block_size + 1)

        # get the previous draft tokens
        assert batch.spec_info is not None, "Missing draft tokens from the previous step"
        prev_draft_tokens = batch.spec_info.send_tokens

        draft_token, positions, custom_mask = build_tidar_positions_and_mask_decode(
            seq_lens=batch.seq_lens_cpu, 
            block_size=block_size, 
            prev_draft_tokens=prev_draft_tokens, 
            mask_token_id=self.mask_token_id, 
            device=self.device
        )

        spec_input = TiDARInput(draft_token, positions, custom_mask, num_queries=B)
        self.target_worker.model_runner.attn_backend.num_draft_tokens = B

        # alloc KV slots
        out_cache_loc = alloc_token_slots(
            batch.tree_cache, 
            B
        )

        assign_draft_cache_locs[(1, )] (
            batch.req_pool_indices, 
            batch.req_to_token_pool.req_to_token,
            batch.seq_lens,
            self.extend_lens, 
            self.num_new_pages_per_topk,
            out_cache_loc,
            batch.req_to_token_pool.req_to_token.shape[1],
            1, 
            B,
            self.page_size, 
            1, 
            next_power_of_2(B)
        )

        batch.out_cache_loc = out_cache_loc
        batch.return_hidden_states = False
        # get model worker batch
        model_worker_batch = batch.get_model_worker_batch()
        # get the forward batch
        model_worker_batch.forward_mode = ForwardMode.TARGET_VERIFY
        model_worker_batch.capture_hidden_mode = CaptureHiddenMode.NULL
        model_worker_batch.input_ids = draft_token
        model_worker_batch.spec_info = spec_input
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.target_worker.model_runner)

        # print("Before forward")
        # print(batch.seq_lens)
        # print(batch.seq_lens_cpu)

        forward_out = self.target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=forward_batch,
            is_verify=True,
            skip_attn_backend_init=False,
        )

        # print("After forward")
        # print(batch.seq_lens)
        # print(batch.seq_lens_cpu)
        # print("================================================")

        logits_output = forward_out.logits_output
        # first merge the logits
        bs = len(batch.seq_lens_cpu)

        logits = logits_output.next_token_logits.view(bs, block_size + 1, block_size, -1).contiguous()
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
            if new_draft_tokens[0, 0, accept_cnt - 1] != verify_tokens[0, accept_cnt]:
                break
            select_draft_tokens = new_draft_tokens[0, accept_cnt + 1]
            accept_cnt += 1

        # clean cache
        if accept_cnt < max_extra_tokens:
            # intermediate decoding step
            batch.tree_cache.token_to_kv_pool_allocator.free(batch.out_cache_loc[accept_cnt:])
        else:
            # last decoding step, clean everything
            cur_req = batch.reqs[0]
            kv_indices = batch.tree_cache.req_to_token_pool.req_to_token[
                cur_req.req_pool_idx, 
                # this is very weird though
                : len(cur_req.origin_input_ids) + len(cur_req.output_ids) + B
            ]
            batch.tree_cache.req_to_token_pool.free(cur_req.req_pool_idx)
            batch.tree_cache.token_to_kv_pool_allocator.free(kv_indices)
            batch.tree_cache.protected_size_ -= len(cur_req.prefix_indices)
            accept_cnt = max_extra_tokens
        
        # Keep only the first accept_cnt queries' KV; evict the rest
        accept_lens = torch.full((bs,), accept_cnt, dtype=torch.int32, device=self.device)

        # Advance lengths
        batch.seq_lens.add_(accept_lens.to(batch.seq_lens.dtype))
        batch.seq_lens_cpu.add_(accept_lens.cpu().to(batch.seq_lens_cpu.dtype))
        # batch.seq_lens.copy_(batch.seq_lens_cpu.to(batch.seq_lens_cpu.device).to(batch.seq_lens.dtype))

        next_draft_input = TiDARInput(
            draft_token=None,
            positions=None,
            custom_mask=None,
            num_queries=0,
            send_tokens=select_draft_tokens.view(-1).to(torch.int32).contiguous().clone(),
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
            can_run_cuda_graph=False,
            next_draft_input=next_draft_input,
            accept_lens=accept_lens,
            allocate_lens=None,
        )