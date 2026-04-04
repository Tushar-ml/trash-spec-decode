from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache

from .modeling_llama import LlamaForCausalLM
from .tree import DraftTree, TreeConfig, build_children_from_parents, tree_to_merged
from .tree_attention import build_tree_attention_mask_4d_with_past


class TokenRecycling:
    def __init__(
        self,
        model_id: str,
        k: int = 8,
        trash_file: Optional[str] = None,
        tree_config: Optional[TreeConfig] = None,
        compile_model: bool = True,
        compile_mode: str = "reduce-overhead",
        cuda_tf32: bool = True,
    ) -> None:
        logger.info(f"Loading {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and cuda_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
            # SDPA is much faster than eager; tree mask still uses the sdpa path with attn_mask set
            attn_implementation="sdpa" if self.device == "cuda" else "eager",
        ).to(self.device)
        self.model.eval()

        if compile_model and self.device == "cuda":
            try:
                self.model = torch.compile(
                    self.model,
                    mode=compile_mode,
                    dynamic=True,
                    fullgraph=False,
                )
                logger.info(
                    f"torch.compile enabled (mode={compile_mode}, dynamic=True)"
                )
            except Exception as exc:
                logger.warning(f"torch.compile skipped: {exc}")

        self.config = AutoConfig.from_pretrained(model_id)
        self.vocab_size = self.config.vocab_size
        self.k = k
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.tree_config = tree_config or TreeConfig()

        if trash_file and os.path.exists(trash_file):
            self.adjacency_matrix = torch.load(trash_file, map_location="cpu")
        else:
            self.adjacency_matrix = torch.zeros((self.vocab_size, k), dtype=torch.long)

        if self.adjacency_matrix.shape != (self.vocab_size, k):
            raise ValueError(
                f"Expected matrix {(self.vocab_size, k)}, got {tuple(self.adjacency_matrix.shape)}"
            )

        logger.info(f"Adjacency matrix shape: {tuple(self.adjacency_matrix.shape)}")

    def _matrix_on_device(self) -> torch.Tensor:
        return self.adjacency_matrix.to(self.device)

    @staticmethod
    def _as_cache(past_key_values: object) -> Cache:
        if isinstance(past_key_values, Cache):
            return past_key_values
        return DynamicCache.from_legacy_cache(past_key_values)

    def _forward_ctx(self, input_ids: torch.LongTensor, past_kv: Optional[Cache]) -> object:
        """KV cache after processing all tokens in input_ids (length T). One token at a time after prefill."""
        batch, t = input_ids.shape
        assert batch == 1
        with torch.inference_mode():
            if past_kv is None:
                return self.model(input_ids, use_cache=True)
            sl = past_kv.get_seq_length()
            if sl == t - 1:
                return self.model(
                    input_ids[:, -1:],
                    past_key_values=past_kv,
                    use_cache=True,
                )
            return self.model(input_ids, use_cache=True)

    def _past_for_next_step(
        self, ctx_past: Cache, accepted: torch.LongTensor
    ) -> Cache:
        """
        After a speculative step, build KV for the next step's invariant:
        next forward only needs past covering new sequence[:-1].

        ctx_past: cache after full current prefix (length T).
        accepted: (1, n) newly appended tokens, n >= 1.
        Returns cache with length T + n - 1.
        """
        n = accepted.shape[1]
        if n == 1:
            return ctx_past
        with torch.inference_mode():
            out = self.model(
                accepted[:, :-1],
                past_key_values=ctx_past,
                use_cache=True,
            )
        return self._as_cache(out.past_key_values)

    def generate_matrix_using_dataset(
        self,
        dataset_path: str,
        max_length: int = 512,
        nsamples: int = -1,
        save_matrix_path: Optional[str] = None,
    ) -> None:
        with open(dataset_path, "r") as f:
            conversations = json.load(f)

        nsamples = nsamples if nsamples > 0 else len(conversations)
        for conv in conversations[:nsamples]:
            prompt = self.tokenizer.apply_chat_template(
                conv, add_generation_prompt=True, tokenize=False
            )
            self.prefill_matrix(prompt, max_length)

        if save_matrix_path:
            torch.save(self.adjacency_matrix, save_matrix_path)

    def prefill_matrix(self, prompt: str, max_length: int = 128) -> None:
        tokens = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        m = self._matrix_on_device()
        past_key_values = None

        with torch.inference_mode():
            for _ in tqdm(range(max_length), leave=False):
                if past_key_values is None:
                    outputs = self.model(tokens, use_cache=True)
                else:
                    outputs = self.model(
                        tokens[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = self._as_cache(outputs.past_key_values)
                current_token = tokens[:, -1].squeeze(-1)
                logits = outputs.logits[:, -1, :]
                top_k_indices = torch.topk(logits, k=self.k, dim=-1).indices
                m[current_token[0]] = top_k_indices[0].long()
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token_id], dim=-1)

        self.adjacency_matrix = m.cpu()

    def _update_matrix_from_logits(
        self, logits_row: torch.Tensor, token_ids: torch.LongTensor
    ) -> None:
        """M[token_id] <- top-k(logits) for each row (parallel scatter)."""
        m = self.adjacency_matrix
        row = logits_row.float()
        _, topk_idx = torch.topk(row, k=self.k, dim=-1)
        m[token_ids.long()] = topk_idx.cpu().long()

    def _speculative_step(
        self,
        input_ids: torch.LongTensor,
        past_kv: Optional[Cache],
    ) -> Tuple[torch.LongTensor, int, Cache]:
        """
        One Token Recycling step. Returns (new_token_chunk, num_tokens_appended, past_for_next_step).

        past_kv: KV after processing input_ids[:-1], or None (full prefill on first step).
        """
        batch, T = input_ids.shape
        assert batch == 1
        device = self.device

        root_id = int(input_ids[0, -1].item())
        draft_tree = DraftTree(
            self.tokenizer,
            self._matrix_on_device(),
            self.tree_config,
        )
        root_node = draft_tree.build_tree(root_id)
        assert root_node is not None
        merged = tree_to_merged(root_node)
        L = merged.token_ids.shape[0]

        ctx_out = self._forward_ctx(input_ids, past_kv)
        ctx_past = self._as_cache(ctx_out.past_key_values)

        if L <= 1:
            last = ctx_out.logits[:, -1, :]
            next_id = last.argmax(dim=-1, keepdim=True)
            tid = torch.tensor([root_id], device="cpu", dtype=torch.long)
            self._update_matrix_from_logits(last[0].cpu(), tid)
            next_past = self._past_for_next_step(ctx_past, next_id)
            return next_id, 1, next_past

        draft = merged.token_ids[1:].to(device).unsqueeze(0)
        parents_dev = merged.parents.to(device)
        attn = build_tree_attention_mask_4d_with_past(
            T, parents_dev, dtype=self.model.dtype, device=device
        )

        with torch.inference_mode():
            logits_ctx = ctx_out.logits[:, -1, :]
            draft_out = self.model(
                draft,
                past_key_values=ctx_past,
                attention_mask=attn,
                use_cache=False,
            )
            logits_draft = draft_out.logits

        parents_cpu = merged.parents
        tok_cpu = merged.token_ids.cpu()
        logits_ctx_cpu = logits_ctx[0].float().cpu()
        logits_draft_cpu = logits_draft[0].float().cpu()

        logits_rows = torch.cat(
            [logits_ctx_cpu.unsqueeze(0), logits_draft_cpu], dim=0
        )
        _, topk_idx = torch.topk(logits_rows, k=self.k, dim=-1)
        self.adjacency_matrix[tok_cpu] = topk_idx.long()

        children = build_children_from_parents(parents_cpu)
        cur = 0
        accepted: List[int] = []
        while True:
            row = logits_ctx_cpu if cur == 0 else logits_draft_cpu[cur - 1]
            pred = int(row.argmax().item())
            nxt = None
            for ch in children[cur]:
                if int(tok_cpu[ch].item()) == pred:
                    nxt = ch
                    break
            if nxt is None:
                break
            accepted.append(pred)
            cur = nxt

        if not accepted:
            nxt_id = int(logits_ctx_cpu.argmax().item())
            single = torch.tensor([[nxt_id]], device=device, dtype=torch.long)
            next_past = self._past_for_next_step(ctx_past, single)
            return single, 1, next_past

        new_toks = torch.tensor([accepted], device=device, dtype=torch.long)
        next_past = self._past_for_next_step(ctx_past, new_toks)
        return new_toks, len(accepted), next_past

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        out = input_ids.to(self.device)
        past_kv: Optional[Cache] = None
        generated = 0
        while generated < max_new_tokens:
            new_toks, n, past_kv = self._speculative_step(out, past_kv)
            out = torch.cat([out, new_toks], dim=-1)
            generated += n
            if eos_token_id is not None and (new_toks == eos_token_id).any():
                break
        return out

    def generate_greedy_ar(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        """Greedy autoregressive baseline: one new token per step with KV cache."""
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        out = input_ids.to(self.device)
        past_key_values: Optional[Cache] = None
        nxt: Optional[torch.LongTensor] = None
        for _ in range(max_new_tokens):
            with torch.inference_mode():
                pk_in = (
                    self._as_cache(past_key_values)
                    if past_key_values is not None
                    else None
                )
                if pk_in is None:
                    outputs = self.model(out, use_cache=True)
                else:
                    outputs = self.model(
                        nxt,
                        past_key_values=pk_in,
                        use_cache=True,
                    )
                past_key_values = self._as_cache(outputs.past_key_values)
                logits = outputs.logits[:, -1, :]
                nxt = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=-1)
            if eos_token_id is not None and nxt.item() == eos_token_id:
                break
        return out

    def save_matrix(self, path: str) -> None:
        torch.save(self.adjacency_matrix, path)


if __name__ == "__main__":
    import os

    _here = os.path.dirname(os.path.abspath(__file__))
    _data = os.path.join(os.path.dirname(_here), "data", "sharegpt_common.json")
    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    tr = TokenRecycling(model_id, k=8)
    if os.path.isfile(_data):
        tr.generate_matrix_using_dataset(_data, nsamples=1, max_length=10)
