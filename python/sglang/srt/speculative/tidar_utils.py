from __future__ import annotations

import torch


def build_tidar_positions_and_mask_prefill(
    seq_lens: torch.Tensor, B: int, mask_token_id: int, device: str
):
    """
    Build positions and custom mask for TiDAR prefill.
    Positions: for each sequence, B queries at seq_len + [0..B-1].
    Custom mask: flattened boolean mask of length
      sum_b B * (seq_lens[b] + B)
      that enables each of the B draft queries to attend to:
        - all previous KV tokens (full prefix)
        - all B draft tokens (bidirectional within the draft block)
    Returns: (draft_token, positions, custom_mask)
    """
    bs = len(seq_lens)
    offsets = torch.arange(B, device=device)
    positions = torch.repeat_interleave(seq_lens, repeats=B) + offsets.repeat(bs)

    # Placeholder: user should supply planned ids; zeros for scaffold
    draft_tokens = torch.full((bs * B,), mask_token_id, dtype=torch.int64, device=device)

    # TiDAR custom mask: for each sequence b, create a B x (seq_len[b] + B)
    # mask that is all True, then flatten and concatenate across sequences.
    # Since it's fully True per row, we can allocate directly by total length.
    total_mask_len = int(B * seq_lens.sum().item() + bs * B * B)
    custom_mask = torch.ones((total_mask_len,), dtype=torch.bool, device=device)
    return draft_tokens, positions, custom_mask


def build_tidar_positions_and_mask_decode(
    seq_lens: torch.Tensor, B: int, prev_draft_tokens: torch.Tensor, mask_token_id: int, device: str
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
    m = B * (B + 1)

    offsets = (
        torch.arange(B + 1, device=device).unsqueeze(0) + 
        torch.arange(B + 1, device=device).unsqueeze(1)
    )[:, :B].contiguous().view(-1)
    positions = torch.repeat_interleave(seq_lens, repeats=m) + offsets.repeat(bs)

    draft_token = prev_draft_tokens.view(bs, B)
    draft_token = torch.concat(
        [draft_token, torch.full((bs, B * B), mask_token_id, device=device)], 
        dim=-1
    ).view(-1)

    # the custom mask for each sample i is like this
    # q_len: B * (B + 1)
    # kv_len: B * (B + 1) + seq_lens[i]
    # first each q will attend to all previous KV before seq_lens[i]
    # it will also attend to the first B

    # TODO: lets start with a single for loop and later on optimize it
    masks = []
    for i in range(bs):
        prefix_mask = torch.full((m, seq_lens[i].item()), True, dtype=torch.bool, device=device)
        draft_mask = torch.zeros((m, m), dtype=torch.bool, device=device)
        # first triangular mask
        draft_mask[:B, :B] = torch.tril(torch.ones((B, B), dtype=torch.bool, device=device))
        # diagonal block with size B
        for j in range(1, B + 1):
            draft_mask[j * B:(j + 1) * B, j * B:(j + 1) * B] = torch.ones((B, B), dtype=torch.bool, device=device)
            draft_mask[j * B:(j + 1) * B, :j] = torch.ones((B, j), dtype=torch.bool, device=device)
        
        masks.append(
            torch.concat(
                [prefix_mask, draft_mask], dim=-1
            ).view(-1)
        )
    custom_mask = torch.concat(masks, dim=-1)

    return draft_token, positions, custom_mask
