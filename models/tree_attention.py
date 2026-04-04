"""
Tree attention masks for Token Recycling (arXiv:2408.08696).

Input layout: input_ids = concat(prefix, draft) where len(prefix)=T, draft = merged[1:]
(merged BFS order includes root at merged[0], equal to prefix[-1]).

Causal LM with tree mask: position q may attend to k iff k <= q AND
(k < T or merged_key is an ancestor of merged_query in the draft tree).
"""

from __future__ import annotations

import torch


def _ancestor_set(parents_merged: list[int], merged_idx: int) -> set[int]:
    s = {merged_idx}
    p = parents_merged[merged_idx]
    while p >= 0:
        s.add(p)
        p = parents_merged[p]
    return s


def build_tree_attention_mask_4d(
    prefix_len: int,
    parents_merged: torch.LongTensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Additive mask: 0 = attend, min = block. Shape (1, 1, seq_len, seq_len).

    parents_merged: length L (merged nodes, root at 0), parent indices or -1 for root.
    Draft segment length = L - 1; seq_len = prefix_len + L - 1.
    """
    parents_list = parents_merged.tolist()
    L = len(parents_list)
    seq_len = prefix_len + L - 1
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_dtype, device=device, dtype=dtype)

    # Base: lower-triangular causal (standard LM)
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask[0, 0].masked_fill_(cols <= rows, 0.0)

    if L <= 1:
        return mask

    # Draft key index k in [prefix_len, seq_len) maps to merged index (k - prefix_len) + 1
    def kv_to_merged(k: int) -> int:
        return k - prefix_len + 1

    for q in range(prefix_len, seq_len):
        merged_q = q - prefix_len + 1
        anc = _ancestor_set(parents_list, merged_q)
        for k in range(prefix_len, q + 1):
            mk = kv_to_merged(k)
            if mk not in anc:
                mask[0, 0, q, k] = min_dtype
        # Full prefix visibility for draft queries (already causal for k < prefix_len)
        mask[0, 0, q, :prefix_len] = 0.0

    return mask


def build_ancestor_table(parents_merged: list[int] | torch.LongTensor) -> list[set[int]]:
    if isinstance(parents_merged, torch.Tensor):
        parents_merged = parents_merged.tolist()
    L = len(parents_merged)
    return [_ancestor_set(parents_merged, i) for i in range(L)]
