import torch
from dataclasses import dataclass
from typing import List, Optional
from collections import deque

@dataclass
class TreeConfig:
    """Configuration for tree structure"""
    max_depth: int = 6       # Default depth of 6 layers as mentioned in the paper
    max_breadth: int = 80    # Optimal breadth of 80 nodes as per experiments
    max_nodes: int = 80      # Total maximum nodes in the tree
    
    def __post_init__(self):
        # Validate configuration
        if self.max_breadth > self.max_nodes:
            self.max_breadth = self.max_nodes
        if self.max_depth * 2 > self.max_nodes:  # Minimum 2 nodes per layer
            self.max_depth = self.max_nodes // 2

@dataclass
class TreeNode:
    """Tree node structure for draft tree"""
    token_id: int
    children: List['TreeNode']
    layer: int              # Track layer for depth-aware operations
    position: int           # Position in layer for breadth control
    
    def can_add_child(self, tree_config: TreeConfig, current_breadth: int) -> bool:
        """
        Determine if node can add more children based on tree constraints.
        """
        # Check if adding a child would exceed max breadth
        if current_breadth >= tree_config.max_breadth:
            return False
            
        # First node in layer can have two children, others one
        max_children = 2 if self.position == 0 else 1
        return len(self.children) < max_children

class DraftTree:
    def __init__(
        self,
        tokenizer,
        adjacency_matrix: torch.Tensor,
        config: Optional[TreeConfig] = None
    ):
        """
        Initialize Draft Tree with optimized structure.
        
        Args:
            tokenizer: HuggingFace tokenizer
            adjacency_matrix: Token adjacency matrix M[token_id, candidate_idx]
            config: Tree configuration parameters
        """
        self.tokenizer = tokenizer
        self.adjacency_matrix = adjacency_matrix
        self.config = config or TreeConfig()
        self.node_count = 0
        
    def get_candidate_tokens(
        self,
        token_id: int,
        num_candidates: int,
        current_layer: int
    ) -> List[int]:
        """
        Get candidate tokens with layer-aware selection.
        
        Args:
            token_id: Input token ID
            num_candidates: Number of candidates to retrieve
            current_layer: Current layer in tree
            
        Returns:
            List of candidate token IDs
        """
        # Get candidates from adjacency matrix (already sorted by probability)
        candidates = self.adjacency_matrix[token_id][:num_candidates]
        
        # For deeper layers, take fewer candidates to maintain efficiency
        if current_layer > self.config.max_depth // 2:
            candidates = candidates[:num_candidates//2]
            
        return candidates.tolist()

    def build_tree(self, root_token_id: int) -> Optional[TreeNode]:
        """
        Build optimized draft tree with controlled growth.
        
        Args:
            root_token_id: Starting token ID
            
        Returns:
            Root node of the constructed tree
        """
        self.node_count = 0
        
        # Initialize root node
        root = TreeNode(
            token_id=root_token_id,
            children=[],
            layer=0,
            position=0
        )
        self.node_count += 1
        
        # Use queue for BFS traversal with layer tracking
        queue = deque([(root, 0)])  # (node, breadth_at_layer)
        current_layer = 0
        layer_nodes = [root]
        
        while queue and current_layer < self.config.max_depth:
            node, current_breadth = queue.popleft()
            
            # Start new layer
            if node.layer > current_layer:
                current_layer = node.layer
                layer_nodes = []
                
            # Skip if we've reached maximum nodes
            if self.node_count >= self.config.max_nodes:
                break
                
            # Get candidates based on layer depth
            max_candidates = 2 if node.position == 0 else 1
            candidates = self.get_candidate_tokens(
                node.token_id,
                max_candidates,
                current_layer
            )
            
            # Create child nodes with position tracking
            for candidate_id in candidates:
                # Check if we can add more nodes
                if self.node_count >= self.config.max_nodes:
                    break
                    
                # Check breadth constraints
                new_breadth = len(layer_nodes) + 1
                if new_breadth > self.config.max_breadth:
                    break
                    
                child = TreeNode(
                    token_id=candidate_id,
                    children=[],
                    layer=current_layer + 1,
                    position=len(layer_nodes)
                )
                
                node.children.append(child)
                layer_nodes.append(child)
                queue.append((child, new_breadth))
                self.node_count += 1
        
        return root
        
    def merge_sequences(self, root: TreeNode) -> List[int]:
            """
            Traverse draft tree by layers to construct merged sequence.
            """
            merged_sequence = []
            queue = deque([root])
            
            while queue:
                level_size = len(queue)
                level_tokens = []
                
                for _ in range(level_size):
                    node = queue.popleft()
                    level_tokens.append(node.token_id)
                    queue.extend(node.children)
                
                merged_sequence.extend(level_tokens)
            
            return merged_sequence


def get_layers(root: TreeNode) -> List[List[TreeNode]]:
    """Helper function to get nodes by layer"""
    layers = []
    current_layer = [root]
    
    while current_layer:
        layers.append(current_layer)
        next_layer = []
        for node in current_layer:
            next_layer.extend(node.children)
        current_layer = next_layer
        
    return layers

def get_all_nodes(root: TreeNode) -> List[TreeNode]:
    """Helper function to get all nodes"""
    nodes = []
    queue = deque([root])
    
    while queue:
        node = queue.popleft()
        nodes.append(node)
        queue.extend(node.children)
        
    return nodes