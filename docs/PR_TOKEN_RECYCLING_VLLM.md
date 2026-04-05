# PR: Add Token Recycling (`token_recycling`) speculative decoding method

## Summary

This PR adds a new train-free speculative decoding method, **`token_recycling`**, inspired by [Token Recycling (arXiv:2408.08696)](https://arxiv.org/abs/2408.08696). It maintains a per-request **adjacency matrix** of top-*k* successor token ids, **updates rows from the target model’s logits** after each forward, and proposes **linear draft chains** that are verified with vLLM’s existing **non-tree** speculative path (same flow as **ngram**).

This does **not** implement the paper’s **BFS draft tree + custom tree attention** in one target forward; that would require deeper integration with vLLM’s tree/EAGLE attention stack. This change focuses on the **matrix + online top-*k* update** core and **chain speculation** compatible with current `SpecDecodeMetadata` and rejection sampling. See [TOKEN_RECYCLING_VLLM_TREE_TRACK.md](TOKEN_RECYCLING_VLLM_TREE_TRACK.md) for a future tree-attention integration outline.

## Optimizations (follow-up)

- **`token_recycling_update_mode`**: `all_logits` (default), `bonus_only` (one top-k per request per step), `decode_only` (skip prompt-phase rows).
- **`token_recycling_max_rows_per_step`**: cap rows after deduplication.
- **`token_recycling_use_sparse_matrix` / `token_recycling_sparse_max_rows`**: LRU sparse rows per request to reduce RAM.
- **Bootstrap file**: `token_recycling_matrix_path` can use a **read-only shared numpy** plus per-request **delta** overlay instead of cloning full dense matrices.
- **CPU scatter**: batched D2H + NumPy grouping in `stage_gpu_rows_then_update` / `update_rows_batched`.

## Configuration

Example `speculative_config`:

- `method`: `"token_recycling"`
- `num_speculative_tokens`: draft length
- `token_recycling_k`: matrix width (default 8)
- `token_recycling_matrix_path`: optional warm-start `.pt` tensor `(vocab_size, k)`
- `token_recycling_update_mode`, `token_recycling_max_rows_per_step`, `token_recycling_use_sparse_matrix`, `token_recycling_sparse_max_rows`: optional tuning

## Testing

- [ ] Short greedy decode with `method: token_recycling`
- [ ] `bonus_only` vs `all_logits` throughput comparison
- [ ] Sparse mode with many concurrent requests

## Related

- Paper: [Turning Trash into Treasure (arXiv:2408.08696)](https://arxiv.org/abs/2408.08696)
