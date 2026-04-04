#!/usr/bin/env python3
"""
Benchmark Token Recycling (draft) vs greedy autoregressive (AR) decoding.

Run from repo root:
  python examples/benchmark_decode.py --mode both --model meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Literal, Optional, Tuple

# Repo root (parent of examples/)
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from transformers import AutoTokenizer

from models.token_recycling import TokenRecycling
from models.tree import TreeConfig


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _time_run(fn, warmup: int, runs: int) -> Tuple[float, torch.LongTensor]:
    for _ in range(warmup):
        fn()
    _sync()
    out = None
    t0 = time.perf_counter()
    for _ in range(runs):
        _sync()
        out = fn()
        _sync()
    t1 = time.perf_counter()
    avg_s = (t1 - t0) / runs
    assert out is not None
    return avg_s, out


def _default_prompt() -> str:
    return "Explain what speculative decoding is in two sentences."


def run_benchmark(
    mode: Literal["draft", "ar", "both"],
    model_id: str,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    matrix_path: Optional[str],
    k: int,
    warmup: int,
    runs: int,
    tree_max_depth: int,
    tree_max_nodes: int,
    tree_max_breadth: int,
) -> None:
    tree_config = TreeConfig(
        max_depth=tree_max_depth,
        max_nodes=tree_max_nodes,
        max_breadth=tree_max_breadth,
    )
    tr = TokenRecycling(
        model_id,
        k=k,
        trash_file=matrix_path,
        tree_config=tree_config,
    )
    eos = tr.tokenizer.eos_token_id
    prompt_len = input_ids.shape[1]
    ids = input_ids.to(tr.device)

    def run_draft():
        return tr.generate(ids.clone(), max_new_tokens=max_new_tokens, eos_token_id=eos)

    def run_ar():
        return tr.generate_greedy_ar(
            ids.clone(), max_new_tokens=max_new_tokens, eos_token_id=eos
        )

    results: dict[str, Tuple[float, int]] = {}

    if mode in ("draft", "both"):
        avg_s, out = _time_run(run_draft, warmup, runs)
        gen = out.shape[1] - prompt_len
        results["draft"] = (avg_s, gen)
        print(f"[draft]  time={avg_s:.4f}s  new_tokens={gen}  tok/s={gen / avg_s:.2f}")

    if mode in ("ar", "both"):
        avg_s, out = _time_run(run_ar, warmup, runs)
        gen = out.shape[1] - prompt_len
        results["ar"] = (avg_s, gen)
        print(f"[ar]     time={avg_s:.4f}s  new_tokens={gen}  tok/s={gen / avg_s:.2f}")

    if mode == "both" and "draft" in results and "ar" in results:
        td, _ = results["draft"]
        ta, _ = results["ar"]
        speedup = ta / td if td > 0 else float("nan")
        print(f"speedup (AR_time / draft_time) = {speedup:.3f}x  (>1 means draft wall-clock is faster)")


def _encode_prompt(
    model_id: str, text: str, chat: bool
) -> torch.LongTensor:
    tok = AutoTokenizer.from_pretrained(model_id)
    if chat:
        messages = [{"role": "user", "content": text}]
        return tok.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )
    return tok(text, return_tensors="pt", add_special_tokens=False).input_ids


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark draft vs AR decoding")
    p.add_argument(
        "--mode",
        choices=["draft", "ar", "both"],
        default="both",
        help="Which decoder(s) to run",
    )
    p.add_argument(
        "--model",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Hugging Face model id",
    )
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--warmup", type=int, default=1, help="Warmup runs before timing")
    p.add_argument("--runs", type=int, default=3, help="Timed runs (average wall time)")
    p.add_argument("--matrix", default=None, help="Optional hot-start adjacency matrix .pt")
    p.add_argument("--k", type=int, default=8, help="Adjacency matrix width (top-k)")
    p.add_argument("--prompt", default=None, help="User prompt (plain text unless --chat)")
    p.add_argument(
        "--prompt-file",
        default=None,
        help="UTF-8 file with prompt text (overrides --prompt)",
    )
    p.add_argument("--chat", action="store_true", help="Wrap prompt as a single user turn")
    p.add_argument("--tree-max-depth", type=int, default=6)
    p.add_argument("--tree-max-nodes", type=int, default=80)
    p.add_argument("--tree-max-breadth", type=int, default=80)
    args = p.parse_args()

    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            text = f.read().strip()
    elif args.prompt:
        text = args.prompt
    else:
        text = _default_prompt()

    input_ids = _encode_prompt(args.model, text, args.chat)

    run_benchmark(
        mode=args.mode,
        model_id=args.model,
        input_ids=input_ids,
        max_new_tokens=args.max_new_tokens,
        matrix_path=args.matrix,
        k=args.k,
        warmup=args.warmup,
        runs=args.runs,
        tree_max_depth=args.tree_max_depth,
        tree_max_nodes=args.tree_max_nodes,
        tree_max_breadth=args.tree_max_breadth,
    )


if __name__ == "__main__":
    main()
