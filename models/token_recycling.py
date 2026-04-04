from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer

from .modeling_llama import LlamaForCausalLM
from .tree import DraftTree, TreeConfig, build_children_from_parents, tree_to_merged
from .tree_attention import build_tree_attention_mask_4d


class TokenRecycling:
    def __init__(
        self,
        model_id: str,
        k: int = 8,
        trash_file: Optional[str] = None,
        tree_config: Optional[TreeConfig] = None,
    ) -> None:
        logger.info(f"Loading {model_id}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = LlamaForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

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

        with torch.no_grad():
            for _ in tqdm(range(max_length), leave=False):
                outputs = self.model(tokens, use_cache=False)
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
        self, input_ids: torch.LongTensor
    ) -> Tuple[torch.LongTensor, int]:
        """
        One Token Recycling step. Returns (new_token_chunk, num_tokens_appended).
        """
        batch, T = input_ids.shape
        assert batch == 1
        dtype = self.model.dtype
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

        if L <= 1:
            with torch.no_grad():
                out = self.model(input_ids, use_cache=False)
            logits = out.logits
            last = logits[:, -1, :]
            next_id = last.argmax(dim=-1, keepdim=True)
            tid = torch.tensor([root_id], device="cpu", dtype=torch.long)
            self._update_matrix_from_logits(last[0].cpu(), tid)
            return next_id, 1

        draft = merged.token_ids[1:].to(device).unsqueeze(0)
        full_ids = torch.cat([input_ids, draft], dim=-1)
        attn = build_tree_attention_mask_4d(
            T, merged.parents.to(device), dtype=dtype, device=device
        )

        with torch.no_grad():
            out = self.model(full_ids, attention_mask=attn, use_cache=False)
        logits = out.logits
        parents_cpu = merged.parents
        tok_cpu = merged.token_ids.cpu()
        logits_cpu = logits[0].float().cpu()
        pos = torch.arange(T - 1, T - 1 + L)
        _, topk_idx = torch.topk(logits_cpu[pos], k=self.k, dim=-1)
        self.adjacency_matrix[tok_cpu] = topk_idx.cpu().long()

        children = build_children_from_parents(parents_cpu)
        cur = 0
        accepted: List[int] = []
        while True:
            pos = T - 1 + cur
            pred = int(logits_cpu[pos].argmax().item())
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
            nxt_id = int(logits_cpu[T - 1].argmax().item())
            return torch.tensor([[nxt_id]], device=device, dtype=torch.long), 1

        return (
            torch.tensor([accepted], device=device, dtype=torch.long),
            len(accepted),
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_new_tokens: int = 256,
        eos_token_id: Optional[int] = None,
    ) -> torch.LongTensor:
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        out = input_ids.to(self.device)
        generated = 0
        while generated < max_new_tokens:
            new_toks, n = self._speculative_step(out)
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
        """Baseline autoregressive greedy (no draft tree)."""
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id
        out = input_ids.to(self.device)
        for _ in range(max_new_tokens):
            with torch.no_grad():
                logits = self.model(out, use_cache=False).logits[:, -1, :]
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
