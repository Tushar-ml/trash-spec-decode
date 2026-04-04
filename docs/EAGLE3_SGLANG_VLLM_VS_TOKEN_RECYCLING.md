# EAGLE3 / tree speculative decoding: SGLang, vLLM, and this repo

This document records how **SGLang** and **vLLM** implement **EAGLE/EAGLE3** tree speculative decoding (upstream: `sgl-project/sglang`, `vllm-project/vllm`), and how that compares to **Token Recycling** in this repository. Sources are the current `main` branches as of the writing date.

## Role split: trained draft vs train-free matrix

| Stack | Draft / proposal | Verification |
|--------|-------------------|--------------|
| **SGLang EAGLE3** | Separate **EAGLE3** checkpoint (`--speculative-draft-model-path`); CUDA `build_tree_kernel_efficient` | Target LLM + `verify_tree_greedy` / sampling kernels |
| **vLLM EAGLE3** | `Eagle3LlamaForCausalLM` (or similar) in `propose()`; optional **tree** path via `propose_tree()` + `TreeAttentionMetadata` | Target model under `CommonAttentionMetadata` + spec decode worker |
| **This repo (Token Recycling)** | **Adjacency matrix** + BFS draft tree ([`models/tree.py`](../models/tree.py)) | Same **Llama** ([`models/modeling_llama.py`](../models/modeling_llama.py)) + tree mask ([`models/tree_attention.py`](../models/tree_attention.py)) |

EAGLE3 is **not** interchangeable with Token Recycling: it depends on **learned** draft weights aligned to the target model.

---

## SGLang (Python + `sgl-kernel`)

### `eagle_utils.py`

- **`TreeMaskMode`** (`IntEnum`): `FULL_MASK`, `QLEN_ONLY`, `QLEN_ONLY_BITPACKING`. Controls how large/dense the tree mask buffer is (full rectangular mask vs query-length-only vs bit-packed storage for memory).
- **`build_tree_kernel_efficient(...)`** (delegates to **`sgl_build_tree_kernel_efficient`** on CUDA/HIP): takes `verified_id`, `parent_list`, `top_scores_index`, `draft_tokens`, `seq_lens`, `topk`, `spec_steps`, `num_verify_tokens`, `tree_mask_mode`, optional preallocated buffers. Returns:
  - **`tree_mask`** (layout depends on mode; comment in code: for `bs=1`, mask relates `num_draft_token` to `seq_lens_sum + num_draft_token`),
  - **`positions`** (per-draft-token absolute position for RoPE),
  - **`retrive_index`**, **`retrive_next_token`**, **`retrive_next_sibling`** (tree navigation for verify; note upstream spelling `retrive`),
  - flattened **`draft_tokens`**.
- **`verify_tree_greedy_func(...)`**: dispatches to **`verify_tree_greedy`** in `sgl_kernel` (CUDA/HIP) or NPU implementation; compares **candidates** to **target_predict** using the retrieve tensors.

### `eagle_info.py` (Eagle verify path)

- **`EagleVerifyInput`** dataclass: holds **`draft_token`**, **`custom_mask`**, **`positions`**, retrieve tensors, **`spec_steps`**, **`topk`**, **`draft_token_num`**, **`seq_lens_sum`**, etc.
- **`prepare_for_verify`**: allocates **KV slots** (paged or token pool) for draft tokens extending the batch.
- **`generate_attn_arg_prefill`**: builds **FlashInfer-style** `kv_indices` via **`create_flashinfer_kv_indices_triton`**, `cum_kv_seq_len`, `qo_indptr`, and pads **`custom_mask`** when needed for **CUDA graph** capture (`mask_numel` combines prefix and draft×draft terms).
- **`verify`**: reshapes logits to **`target_predict`**, calls **`verify_tree_greedy_func`** for greedy (or tree speculative sampling when kernels available), then walks accepted indices to update requests, **KV commit length**, and **evict** unaccepted draft slots.

**Takeaway for this repo:** SGLang fuses **tree construction**, **mask layout modes**, **paged KV indices**, and **greedy verify** in C++/Triton/CUDA. Our [`build_tree_attention_mask_4d_with_past`](models/tree_attention.py) is the conceptual analog of **`custom_mask`**, but dense and built in PyTorch.

---

## vLLM (`vllm/v1/spec_decode/eagle.py`)

### Proposer (`SpecDecodeBaseProposer` / EAGLE path)

- **`method in ("eagle3", "dflash")`**: asserts draft model types such as **`Eagle3LlamaForCausalLM`**, **`Eagle3DeepseekV2ForCausalLM`**, **`DFlashQwen3ForCausalLM`**; combines hidden states via **`combine_hidden_states`** for EAGLE3.
- **`propose(...)`**: first pass runs the **draft model** under `set_forward_context` with **piecewise CUDA graphs** when enabled; samples draft tokens (greedy or **`propose_tree`**).
- **`propose_tree(...)`** (when attention metadata includes **`TreeAttentionMetadata`**): uses **`TreeAttentionMetadataBuilder.build_for_drafting`**, level-wise expansion with **`cu_drafts_per_level`**, **`child_drafts_per_level`**, **`topk`** at each level, concatenates **`tree_input_ids` / `tree_positions` / `tree_hidden_states`**, and advances **tree depth** with updated **`CommonAttentionMetadata`** (query length, seq lens, etc.).

### Integration

- Imports **`TreeAttentionMetadata`** / **`TreeAttentionMetadataBuilder`** from **`vllm.v1.attention.backends.tree_attn`** — tree attention is a **first-class backend**, not a generic 4D padding mask in Hugging Face.

**Takeaway for this repo:** vLLM’s EAGLE3 tree path is **draft-model-driven** and tied to **v1 attention metadata** and **KV block tables**. Token Recycling keeps a **single** target model and builds the tree from **`M[token]`**, which is closer in *shape* to SGLang’s **parent_list + top-k** inputs than to vLLM’s **Eagle3Llama** forward.

---

## This repository (Token Recycling)

| Concept | Location |
|--------|----------|
| Draft tree (BFS, breadth limits) | [`models/tree.py`](../models/tree.py) |
| Tree mask for **incremental** decode (`past` + draft queries) | [`build_tree_attention_mask_4d_with_past`](models/tree_attention.py) |
| Context KV + draft forward + matrix update + greedy walk | [`models/token_recycling.py`](../models/token_recycling.py) |

Parallel concepts:

- **SGLang `retrive_*` / parent indices** ↔ our **`parents`** in **`MergedTree`** and **`build_children_from_parents`**.
- **SGLang `verify_tree_greedy`** ↔ our Python loop comparing **`argmax(logits_row)`** to child **`token_id`** along the tree.
- **SGLang `positions`** ↔ Hugging Face **`cache_position`** / RoPE driven by incremental cache (we do not hand-build per-node RoPE offsets like some tree backends).

---

## Optional speed-ups (scoping): Triton / CUDA graphs vs this codebase

These are **not** implemented here; they are scoped options if you need **EAGLE-like** throughput without adopting EAGLE weights.

1. **Triton or fused tree attention**  
   Replace dense `(1,1,Q,KV)` FP16 masks in SDPA with a **kernel that reads a compact tree** (parent id + depth or CSR) and applies the same allow/deny rules as [`tree_attention.py`](models/tree_attention.py). Effort: high; must match RoPE **positions** policy used today.

2. **Pad + bucket + CUDA graph**  
   Same idea as SGLang padding **`custom_mask`** for graph capture: fix **`draft_len`** to a small set of buckets (e.g. 8, 16, 32), **pad** draft tokens and mask, run **`torch.cuda.CUDAGraph`** on the **draft** forward only. Helps when many steps share the same bucket.

3. **FlashInfer / paged KV indices**  
   SGLang’s **`create_flashinfer_kv_indices_triton`** builds **indices into paged KV**. Our path uses **Hugging Face `DynamicCache`** (contiguous per layer). Porting would mean either integrating **vLLM/SGLang as a backend** or reimplementing a minimal paged path — large project.

4. **Integration**  
   Fastest path to “production” throughput is often **running the target in vLLM/SGLang** and adding a **custom proposer** that implements Token Recycling’s matrix + tree **outside** those stacks, then feeding proposals into their verify API — **if** such a hook exists; otherwise a fork is required.

---

## References (upstream paths)

- SGLang: `python/sglang/srt/speculative/eagle_utils.py`, `eagle_info.py`; CUDA: `sgl-kernel/csrc/speculative/eagle_utils.cu`.
- vLLM: `vllm/v1/spec_decode/eagle.py`, `vllm/v1/attention/backends/tree_attn.py` (tree metadata).
