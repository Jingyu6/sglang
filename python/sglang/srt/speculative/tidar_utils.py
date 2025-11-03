from __future__ import annotations

import torch


def build_tidar_positions_and_mask_prefill(
    seq_lens: torch.Tensor, B: int, device: str
):
    """
    Build positions and a placeholder custom mask for TiDAR prefill.
    Positions: for each sequence, B queries at seq_len + [0..B-1].
    Custom mask: placeholder (all ones); replace with TiDAR connectivity.
    Returns: (draft_token, positions, custom_mask)
    """
    bs = len(seq_lens)
    offsets = torch.arange(B, device=device)
    positions = torch.repeat_interleave(seq_lens, repeats=B) + offsets.repeat(bs)

    # Placeholder: user should supply planned ids; zeros for scaffold
    draft_token = torch.zeros((bs * B,), dtype=torch.int64, device=device)

    # Placeholder custom mask buffer
    custom_mask = torch.ones((bs * B,), dtype=torch.uint8, device=device)
    return draft_token, positions, custom_mask


def build_tidar_positions_and_mask_decode(
    seq_lens: torch.Tensor, B: int, device: str
):
    """
    Build positions and custom mask for TiDAR decode iteration.
    Produces m=B*(B+1) queries per sequence. Positions are seq_len + [0..m-1].
    Returns: (draft_token, positions, custom_mask)
    """
    bs = len(seq_lens)
    m = B * (B + 1)
    offsets = torch.arange(m, device=device)
    positions = torch.repeat_interleave(seq_lens, repeats=m) + offsets.repeat(bs)

    draft_token = torch.zeros((bs * m,), dtype=torch.int64, device=device)
    custom_mask = torch.ones((bs * m,), dtype=torch.uint8, device=device)
    return draft_token, positions, custom_mask


