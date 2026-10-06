"""Shared helpers for the SGLang attention benchmarks."""

import torch


def repeat_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand grouped-query KV heads to match the number of query heads.

    ``x`` is a ``[tokens, num_kv_heads, head_dim]`` tensor. When ``num_kv_heads``
    already equals ``num_q_heads`` it is returned unchanged; otherwise each KV
    head is repeated ``num_q_heads // num_kv_heads`` times along the head axis.
    """
    if x.shape[1] == num_q_heads:
        return x
    return torch.repeat_interleave(x, num_q_heads // x.shape[1], dim=1)
