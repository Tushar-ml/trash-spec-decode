# Token Recycling: tree draft + tree attention (future work)

Full parity with arXiv:2408.08696 requires **BFS draft trees** and **one target forward** with a **tree attention mask** over draft nodes, then greedy (or sampled) walk to accept a prefix.

## vLLM integration points

1. **Tree metadata** — `vllm.v1.attention.backends.tree_attn` (`TreeAttentionMetadata`, `TreeAttentionMetadataBuilder`) used by EAGLE tree speculation.
2. **Proposer** — Replace linear `propose()` with BFS over the adjacency matrix (see `trash-spec-decode/models/tree.py` / `DraftTree`) and emit tree-shaped draft token ids + parent links compatible with the tree backend.
3. **Verify** — Reuse EAGLE’s batched tree verify path (`vllm/v1/spec_decode/eagle.py`, rejection sampling with tree logits indices) or SGLang-style `verify_tree_greedy` concepts documented in `docs/EAGLE3_SGLANG_VLLM_VS_TOKEN_RECYCLING.md`.
4. **CUDA graphs** — Tree shapes may need bucketing/padding like existing spec decode.

## Current linear path

The `token_recycling` method in vLLM uses **chain** drafts and standard `SpecDecodeMetadata` (no `TreeAttentionMetadata`). Matrix updates and `token_recycling_update_mode` optimize that path only.
