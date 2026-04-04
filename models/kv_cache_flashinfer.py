"""
KV cache for FlashInfer-backed decoding.

FlashInfer `single_*_with_kv_cache` APIs expect contiguous K/V in NHD layout
`[kv_len, num_kv_heads, head_dim]` (after batch squeeze). Hugging Face
`DynamicCache` / `DynamicLayer` already store
`[batch, num_kv_heads, seq_len, head_dim]`, which is the same memory layout
after `permute(1, 0, 2)` — no separate paged allocator is required for the
integration in `flashinfer_attention.py`.

A future optimization can swap `DynamicLayer` for true paged blocks + indices
(SGLang-style) without changing Token Recycling matrix/tree logic.
"""

from __future__ import annotations

from transformers.cache_utils import DynamicCache

__all__ = ["FlashInferCompatCache"]


class FlashInferCompatCache(DynamicCache):
    """Same behavior as `DynamicCache`; documents FlashInfer-compatible contiguous KV."""

    pass
