# Examples: draft vs autoregressive decoding

Scripts assume you run them from the **repository root** so `models/` is importable.

## Dependencies

Install from the repo root:

```bash
pip install -r requirements.txt
```

Use a GPU for meaningful throughput numbers.

## Benchmark: Token Recycling (draft) vs greedy AR

[`benchmark_decode.py`](benchmark_decode.py) times both decoders on the same prompt and reports tokens/s plus a wall-clock ratio.

```bash
# Both modes (default): draft + baseline AR
python examples/benchmark_decode.py --mode both --model meta-llama/Llama-3.2-1B-Instruct

# Only draft (Token Recycling)
python examples/benchmark_decode.py --mode draft --max-new-tokens 256

# Only greedy autoregressive (no draft tree)
python examples/benchmark_decode.py --mode ar --max-new-tokens 256

# Chat template (single user message)
python examples/benchmark_decode.py --chat --prompt "What is attention in transformers?"

# Hot-started adjacency matrix (paper: warm library from prior runs)
python examples/benchmark_decode.py --matrix /path/to/matrix.pt --mode both
```

Useful flags:

| Flag | Meaning |
|------|--------|
| `--warmup` | Runs before timing (default 1) |
| `--runs` | Averaged timed runs (default 3) |
| `--matrix` | Load `torch.save`’d adjacency matrix |
| `--k` | Top-k stored per vocabulary row (default 8) |
| `--tree-max-depth` / `--tree-max-nodes` / `--tree-max-breadth` | Draft tree limits |

To **build** a matrix file first, use `TokenRecycling.generate_matrix_using_dataset` or `prefill_matrix` from [`models/token_recycling.py`](../models/token_recycling.py), then pass `--matrix` here.
