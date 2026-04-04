from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional

import torch

@dataclass
class TreeConfig:
    # Paper uses depth 6 / ~80 nodes; smaller defaults cut draft-verify cost per step.
    # Raise toward paper values when the matrix is hot and acceptance is high.
    max_depth: int = 5
    max_breadth: int = 40
    max_nodes: int = 40

    def __post_init__(self) -> None:
        if self.max_breadth > self.max_nodes:
            self.max_breadth = self.max_nodes


@dataclass
class TreeNode:
    token_id: int
    children: List["TreeNode"]
    layer: int
    position: int


@dataclass
class MergedTree:
    """BFS merged sequence (root first) and parent index per merged node."""

    token_ids: torch.LongTensor
    parents: torch.LongTensor


def _valid_candidates(row: torch.Tensor, num_take: int) -> List[int]:
    """Skip 0 and negative entries (paper: 0 = no candidate)."""
    out: List[int] = []
    for x in row.tolist():
        if len(out) >= num_take:
            break
        if x <= 0:
            continue
        out.append(int(x))
    return out


class DraftTree:
    def __init__(
        self,
        tokenizer,
        adjacency_matrix: torch.Tensor,
        config: Optional[TreeConfig] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.adjacency_matrix = adjacency_matrix
        self.config = config or TreeConfig()
        self.node_count = 0

    def get_candidate_tokens(self, token_id: int, num_candidates: int, current_layer: int) -> List[int]:
        row = self.adjacency_matrix[token_id]
        k_take = num_candidates
        if current_layer > self.config.max_depth // 2:
            k_take = max(1, num_candidates // 2)
        return _valid_candidates(row, k_take)

    def build_tree(self, root_token_id: int) -> Optional[TreeNode]:
        self.node_count = 0
        root = TreeNode(token_id=root_token_id, children=[], layer=0, position=0)
        self.node_count = 1

        current_level: List[TreeNode] = [root]
        while current_level and self.node_count < self.config.max_nodes:
            next_level: List[TreeNode] = []
            for idx_in_layer, node in enumerate(current_level):
                if len(next_level) >= self.config.max_breadth:
                    break
                if node.layer >= self.config.max_depth:
                    continue
                max_children = 2 if idx_in_layer == 0 else 1
                candidates = self.get_candidate_tokens(node.token_id, max_children, node.layer)
                if not candidates:
                    continue
                for cand in candidates:
                    if self.node_count >= self.config.max_nodes:
                        break
                    if len(next_level) >= self.config.max_breadth:
                        break
                    child = TreeNode(
                        token_id=cand,
                        children=[],
                        layer=node.layer + 1,
                        position=idx_in_layer,
                    )
                    node.children.append(child)
                    next_level.append(child)
                    self.node_count += 1
            current_level = next_level

        return root

    def merge_sequences(self, root: TreeNode) -> List[int]:
        merged: List[int] = []
        q: deque[TreeNode] = deque([root])
        while q:
            level_size = len(q)
            for _ in range(level_size):
                node = q.popleft()
                merged.append(node.token_id)
                q.extend(node.children)
        return merged


def tree_to_merged(root: TreeNode) -> MergedTree:
    """Layer-order (BFS) token list and parent indices in merged space."""
    nodes: List[TreeNode] = []
    q: deque[TreeNode] = deque([root])
    while q:
        level_size = len(q)
        for _ in range(level_size):
            node = q.popleft()
            nodes.append(node)
            q.extend(node.children)

    n = len(nodes)
    token_ids = torch.tensor([nodes[i].token_id for i in range(n)], dtype=torch.long)
    parents = torch.full((n,), -1, dtype=torch.long)
    index_of_id: dict[int, int] = {}
    for i, node in enumerate(nodes):
        index_of_id[id(node)] = i
    for i, node in enumerate(nodes):
        for ch in node.children:
            parents[index_of_id[id(ch)]] = i

    return MergedTree(token_ids=token_ids, parents=parents)


def build_children_from_parents(parents: torch.LongTensor) -> List[List[int]]:
    L = parents.shape[0]
    children: List[List[int]] = [[] for _ in range(L)]
    for i in range(1, L):
        p = int(parents[i].item())
        if p >= 0:
            children[p].append(i)
    return children
