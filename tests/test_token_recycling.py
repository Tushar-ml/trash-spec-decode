import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models.tree_attention import (
    build_tree_attention_mask_4d,
    build_tree_attention_mask_4d_with_past,
)
from models.token_recycling import TokenRecycling
from models.tree import DraftTree, TreeConfig, tree_to_merged


def test_tree_mask_ancestors_only():
    # Merged: 0 root, 1 child of 0, 2 child of 0 (sibling of 1)
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    T = 4
    dtype = torch.float32
    device = torch.device("cpu")
    mask = build_tree_attention_mask_4d(T, parents, dtype, device)
    min_dtype = torch.finfo(dtype).min
    m = mask[0, 0]
    q = T  # first draft query — merged node 1
    # Sibling draft column at T+1 should be blocked at query T (only self path)
    assert m[q, T + 1].item() == min_dtype
    assert m[q, T].item() == 0.0
    assert m[q, T - 1].item() == 0.0


def test_tree_mask_single_chain():
    parents = torch.tensor([-1, 0, 1], dtype=torch.long)
    T = 3
    dtype = torch.float32
    device = torch.device("cpu")
    mask = build_tree_attention_mask_4d(T, parents, dtype, device)
    min_dtype = torch.finfo(dtype).min
    m = mask[0, 0]
    q2 = T + 1  # merged index 2
    assert m[q2, T - 1].item() == 0.0
    assert m[q2, T].item() == 0.0
    assert m[q2, T + 1].item() == 0.0


def test_update_rule_mock():
    vocab = 100
    k = 4
    tr = object.__new__(TokenRecycling)
    tr.k = k
    tr.vocab_size = vocab
    tr.adjacency_matrix = torch.zeros((vocab, k), dtype=torch.long)
    logits = torch.randn(vocab)
    TokenRecycling._update_matrix_from_logits(tr, logits, torch.tensor([7], dtype=torch.long))
    _, expected = torch.topk(logits, k=k)
    assert torch.equal(tr.adjacency_matrix[7], expected.long())


def test_mask_seq_len_matches_paper_layout():
    parents = torch.tensor([-1, 0], dtype=torch.long)
    T = 3
    mask = build_tree_attention_mask_4d(T, parents, torch.float32, torch.device("cpu"))
    assert mask.shape[-1] == T + parents.shape[0] - 1


def test_past_mask_matches_full_draft_rows():
    parents = torch.tensor([-1, 0, 0], dtype=torch.long)
    T, dtype, dev = 5, torch.float32, torch.device("cpu")
    full_m = build_tree_attention_mask_4d(T, parents, dtype, dev)[0, 0]
    past_m = build_tree_attention_mask_4d_with_past(T, parents, dtype, dev)[0, 0]
    draft_len = parents.shape[0] - 1
    for i in range(draft_len):
        q_full = T + i
        assert torch.allclose(
            full_m[q_full, : T + i + 1],
            past_m[i, : T + i + 1],
        )


def test_draft_tree_chain_matches_merge():
    class _Tok:
        pass

    m = torch.zeros((50, 8), dtype=torch.long)
    for i in range(1, 20):
        m[i, : min(3, 8)] = torch.tensor([i + 1, i + 2, i + 3][: min(3, 8)])
    dt = DraftTree(_Tok(), m, TreeConfig(max_depth=5, max_nodes=10, max_breadth=80))
    root = dt.build_tree(1)
    merged = tree_to_merged(root)
    assert merged.token_ids[0].item() == 1
    assert merged.parents[0].item() == -1
