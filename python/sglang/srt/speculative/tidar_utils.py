from __future__ import annotations

import torch


def build_tidar_positions_and_mask_prefill(
    seq_lens: torch.Tensor, block_size: int, mask_token_id: int, device: str
):
    """
    Build positions and custom mask for TiDAR prefill.
    Positions: for each sequence, block_size queries at seq_len + [0..block_size-1].
    Custom mask: flattened boolean mask of length
      sum_b block_size * (seq_lens[b] + block_size)
      that enables each of the block_size draft queries to attend to:
        - all previous KV tokens (full prefix)
        - all block_size draft tokens (bidirectional within the draft block)
    Returns: (draft_token, positions, custom_mask)
    """
    bs = len(seq_lens)
    offsets = torch.arange(block_size, device=device, dtype=torch.long)
    positions = (
        torch.repeat_interleave(seq_lens.to(torch.long), repeats=block_size)
        + offsets.repeat(bs)
    ).contiguous()

    # Placeholder: user should supply planned ids; zeros for scaffold
    draft_tokens = torch.full((bs * block_size,), mask_token_id, dtype=torch.long, device=device)

    # TiDAR custom mask: for each sequence b, create a block_size x (seq_len[b] + block_size)
    # mask that is all True, then flatten and concatenate across sequences.
    # Since it's fully True per row, we can allocate directly by total length.
    total_mask_len = int(block_size * seq_lens.sum().item() + bs * block_size * block_size)
    custom_mask = torch.ones((total_mask_len,), dtype=torch.bool, device=device)
    return draft_tokens, positions, custom_mask


def build_tidar_positions_and_mask_decode(
    seq_lens: torch.Tensor, block_size: int, prev_draft_tokens: torch.Tensor, mask_token_id: int, device: str
):
    """
    Build positions and custom mask for TiDAR decode iteration.
    Positions: for each sequence, B queries at seq_len + [0..B-1].
    Custom mask: flattened boolean mask of length
      sum_b B * (seq_lens[b] + B)
      that enables each of the B draft queries to attend to:
        - all previous KV tokens (full prefix)
        - all B draft tokens (bidirectional within the draft block)
    Returns: (draft_token, positions, custom_mask)
    """
    bs = len(seq_lens)
    B = block_size * (block_size + 1)

    offsets = (
        torch.arange(block_size + 1, device=device, dtype=torch.long).unsqueeze(0)
        + torch.arange(block_size + 1, device=device, dtype=torch.long).unsqueeze(1)
    )[:, :block_size].contiguous().view(-1)
    positions = (
        torch.repeat_interleave(seq_lens.to(torch.long), repeats=B)
        + offsets.repeat(bs)
    ).contiguous()

    draft_token = prev_draft_tokens.view(bs, block_size)
    draft_token = torch.concat(
        [draft_token, torch.full((bs, block_size * block_size), mask_token_id, device=device)], 
        dim=-1
    ).view(-1)

    # TODO: lets start with a single for loop and later on optimize it
    masks = []
    for i in range(bs):
        prefix_mask = torch.full((B, seq_lens[i].item()), 1, dtype=torch.bool, device=device)
        draft_mask = torch.zeros((B, B), dtype=torch.bool, device=device)
        # first triangular mask
        draft_mask[:block_size, :block_size] = torch.tril(
            torch.ones((block_size, block_size), dtype=torch.bool, device=device)
        )
        # diagonal block with size B
        for j in range(1, block_size + 1):
            draft_mask[j * block_size:(j + 1) * block_size, j * block_size:(j + 1) * block_size] = torch.ones((block_size, block_size), dtype=torch.bool, device=device)
            draft_mask[j * block_size:(j + 1) * block_size, :j] = torch.ones((block_size, j), dtype=torch.bool, device=device)
        
        masks.append(
            torch.concat(
                [prefix_mask, draft_mask], dim=-1
            ).view(-1)
        )
    custom_mask = torch.concat(masks, dim=-1)

    return draft_token, positions, custom_mask
