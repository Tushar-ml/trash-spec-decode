"""
Tree attention masks for Token Recycling (arXiv:2408.08696).

Input layout: input_ids = concat(prefix, draft) where len(prefix)=T, draft = merged[1:]
(merged BFS order includes root at merged[0], equal to prefix[-1]).

Causal LM with tree mask: position q may attend to k iff k <= q AND
(k < T or merged_key is an ancestor of merged_query in the draft tree).

Masks are built with vectorized torch ops (no per-cell Python) so GPU steps stay fast.
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


def build_ancestor_bool_matrix(parents_merged: torch.LongTensor) -> torch.Tensor:
    """
    anc[i, j] is True iff merged node j is on the path from merged node i to the root
    (including i = j). parents_merged[j] is parent of j, or -1 for the root.
    """
    device = parents_merged.device
    pl = parents_merged.tolist()
    L = len(pl)
    anc = torch.zeros((L, L), dtype=torch.bool, device=device)
    for i in range(L):
        c = i
        while c >= 0:
            anc[i, c] = True
            c = pl[c]
    return anc


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
    parents_merged = parents_merged.to(device)
    L = int(parents_merged.shape[0])
    draft_len = L - 1
    seq_len = prefix_len + draft_len
    min_dtype = torch.finfo(dtype).min
    mask = torch.full((1, 1, seq_len, seq_len), min_dtype, device=device, dtype=dtype)

    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(seq_len, device=device).unsqueeze(0)
    mask[0, 0].masked_fill_(cols <= rows, 0.0)

    if L <= 1:
        return mask

    anc = build_ancestor_bool_matrix(parents_merged)
    anc_sub = anc[1:L, 1:L]
    I = torch.arange(draft_len, device=device).unsqueeze(1)
    J = torch.arange(draft_len, device=device).unsqueeze(0)
    tril = J <= I
    bad = tril & (~anc_sub)
    sub = mask[0, 0, prefix_len : prefix_len + draft_len, prefix_len : prefix_len + draft_len]
    sub[bad] = min_dtype

    return mask


def build_tree_attention_mask_4d_with_past(
    prefix_len: int,
    parents_merged: torch.LongTensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Tree mask for an incremental forward: queries = draft tokens only (len = L-1),
    keys = prefix (len prefix_len) + draft (len L-1).

    Shape (1, 1, draft_len, prefix_len + draft_len). Use with
    model(draft_ids, past_key_values=prefill_cache, attention_mask=this).
    """
    parents_merged = parents_merged.to(device)
    L = int(parents_merged.shape[0])
    draft_len = L - 1
    if draft_len <= 0:
        raise ValueError("parents_merged must have length >= 2")

    min_dtype = torch.finfo(dtype).min
    anc = build_ancestor_bool_matrix(parents_merged)
    anc_sub = anc[1:L, 1:L]
    I = torch.arange(draft_len, device=device).unsqueeze(1)
    J = torch.arange(draft_len, device=device).unsqueeze(0)
    tril = J <= I
    good = tril & anc_sub
    z = torch.zeros((), dtype=dtype, device=device)
    draft_part = torch.where(good, z, torch.tensor(min_dtype, dtype=dtype, device=device))

    left = torch.zeros((draft_len, prefix_len), dtype=dtype, device=device)
    full = torch.cat([left, draft_part], dim=1)
    return full.view(1, 1, draft_len, prefix_len + draft_len)


def build_ancestor_table(parents_merged: list[int] | torch.LongTensor) -> list[set[int]]:
    if isinstance(parents_merged, torch.Tensor):
        parents_merged = parents_merged.tolist()
    L = len(parents_merged)
    return [_ancestor_set(parents_merged, i) for i in range(L)]
