"""
Tree Encoder

Implements tree-aware positional encoding for hierarchical HTML structures.
Supports both BFS and DFS traversal with learnable position embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
from dataclasses import dataclass
import math

logger = logging.getLogger(__name__)


@dataclass
class TreeNode:
    """Represents a node in the HTML tree structure."""
    node_id: str
    tag: str
    css_classes: List[str]
    content: Dict[str, Any]
    depth: int
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []


@dataclass
class PositionInfo:
    """Contains positional information for a tree node."""
    node_id: str
    bfs_position: int
    dfs_position: int
    depth: int
    parent_position: Optional[int] = None
    sibling_index: int = 0
    path_from_root: List[int] = None
    
    def __post_init__(self):
        if self.path_from_root is None:
            self.path_from_root = []


class TreeEncoder:
    """
    Encodes hierarchical HTML structures with tree-aware positional information.
    
    Supports multiple traversal strategies and position embedding types
    for optimal representation of DOM tree relationships.
    """
    
    def __init__(self, 
                 max_depth: int = 20,
                 max_nodes: int = 1000,
                 embedding_dim: int = 768,
                 encoding_strategy: str = 'bfs'):
        
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.embedding_dim = embedding_dim
        self.encoding_strategy = encoding_strategy
        
        # Supported encoding strategies
        self.supported_strategies = ['bfs', 'dfs', 'hybrid', 'path_based']
        
        if encoding_strategy not in self.supported_strategies:
            raise ValueError(f"Encoding strategy must be one of {self.supported_strategies}")
        
        # Position embedding types
        self.position_embedding_types = ['learnable', 'sinusoidal', 'relative']
        self.current_embedding_type = 'learnable'
        
        # Tree statistics
        self.tree_stats = {}
        
        logger.info(f"TreeEncoder initialized with {encoding_strategy} strategy")
    
    def build_tree_from_structure(self, html_structure: Dict[str, Any], 
                                node_id_prefix: str = "node") -> Dict[str, TreeNode]:
        """
        Build tree representation from HTML structure.
        
        Args:
            html_structure: Nested HTML structure dictionary
            node_id_prefix: Prefix for generated node IDs
            
        Returns:
            Dictionary mapping node IDs to TreeNode objects
        """
        nodes = {}
        node_counter = 0
        
        def create_node(key: str, value: Any, parent_id: Optional[str], depth: int) -> TreeNode:
            nonlocal node_counter
            node_id = f"{node_id_prefix}_{node_counter}"
            node_counter += 1
            
            # Parse element key to extract tag and classes
            if '@' in key:
                parts = key.split('@')
                tag = parts[0]
                css_classes = parts[1:]
            else:
                tag = key
                css_classes = []
            
            # Extract content
            content = {}
            if isinstance(value, dict):
                for content_key in ['text', 'src', 'svg']:
                    if content_key in value:
                        content[content_key] = value[content_key]
            
            # Create node
            node = TreeNode(
                node_id=node_id,
                tag=tag,
                css_classes=css_classes,
                content=content,
                depth=depth,
                parent_id=parent_id
            )
            
            return node
        
        def traverse_structure(structure: Dict[str, Any], parent_id: Optional[str], depth: int):
            for key, value in structure.items():
                # Skip content keys
                if key in ['text', 'src', 'svg']:
                    continue
                
                # Create node for this element
                node = create_node(key, value, parent_id, depth)
                nodes[node.node_id] = node
                
                # Add to parent's children
                if parent_id and parent_id in nodes:
                    nodes[parent_id].children_ids.append(node.node_id)
                
                # Recursively process children
                if isinstance(value, dict):
                    traverse_structure(value, node.node_id, depth + 1)
        
        # Start traversal
        traverse_structure(html_structure, None, 0)
        
        return nodes
    
    def get_bfs_positions(self, nodes: Dict[str, TreeNode]) -> Dict[str, PositionInfo]:
        """
        Get BFS (breadth-first search) positions for all nodes.
        
        Args:
            nodes: Dictionary of TreeNode objects
            
        Returns:
            Dictionary mapping node IDs to PositionInfo with BFS positions
        """
        positions = {}
        
        # Find root nodes (nodes with no parent)
        root_nodes = [node for node in nodes.values() if node.parent_id is None]
        
        if not root_nodes:
            return positions
        
        # BFS traversal
        queue = deque()
        position_counter = 0
        
        # Initialize with root nodes
        for root in root_nodes:
            queue.append((root.node_id, 0))  # (node_id, sibling_index)
        
        while queue:
            node_id, sibling_index = queue.popleft()
            node = nodes[node_id]
            
            # Create position info
            parent_position = None
            if node.parent_id and node.parent_id in positions:
                parent_position = positions[node.parent_id].bfs_position
            
            positions[node_id] = PositionInfo(
                node_id=node_id,
                bfs_position=position_counter,
                dfs_position=-1,  # Will be filled by DFS
                depth=node.depth,
                parent_position=parent_position,
                sibling_index=sibling_index
            )
            
            position_counter += 1
            
            # Add children to queue
            for i, child_id in enumerate(node.children_ids):
                queue.append((child_id, i))
        
        return positions
    
    def get_dfs_positions(self, nodes: Dict[str, TreeNode]) -> Dict[str, PositionInfo]:
        """
        Get DFS (depth-first search) positions for all nodes.
        
        Args:
            nodes: Dictionary of TreeNode objects
            
        Returns:
            Dictionary mapping node IDs to PositionInfo with DFS positions
        """
        positions = {}
        
        # Find root nodes
        root_nodes = [node for node in nodes.values() if node.parent_id is None]
        
        if not root_nodes:
            return positions
        
        position_counter = 0
        
        def dfs_traverse(node_id: str, sibling_index: int, path: List[int]):
            nonlocal position_counter
            
            node = nodes[node_id]
            
            # Create position info
            parent_position = None
            if node.parent_id and node.parent_id in positions:
                parent_position = positions[node.parent_id].dfs_position
            
            positions[node_id] = PositionInfo(
                node_id=node_id,
                bfs_position=-1,  # Will be filled by BFS
                dfs_position=position_counter,
                depth=node.depth,
                parent_position=parent_position,
                sibling_index=sibling_index,
                path_from_root=path.copy()
            )
            
            position_counter += 1
            
            # Recursively traverse children
            for i, child_id in enumerate(node.children_ids):
                child_path = path + [i]
                dfs_traverse(child_id, i, child_path)
        
        # Start DFS from each root
        for i, root in enumerate(root_nodes):
            dfs_traverse(root.node_id, i, [i])
        
        return positions
    
    def get_hybrid_positions(self, nodes: Dict[str, TreeNode]) -> Dict[str, PositionInfo]:
        """
        Get hybrid positions combining BFS and DFS information.
        
        Args:
            nodes: Dictionary of TreeNode objects
            
        Returns:
            Dictionary mapping node IDs to PositionInfo with both BFS and DFS positions
        """
        bfs_positions = self.get_bfs_positions(nodes)
        dfs_positions = self.get_dfs_positions(nodes)
        
        # Merge position information
        hybrid_positions = {}
        for node_id in nodes.keys():
            if node_id in bfs_positions and node_id in dfs_positions:
                bfs_info = bfs_positions[node_id]
                dfs_info = dfs_positions[node_id]
                
                hybrid_positions[node_id] = PositionInfo(
                    node_id=node_id,
                    bfs_position=bfs_info.bfs_position,
                    dfs_position=dfs_info.dfs_position,
                    depth=bfs_info.depth,
                    parent_position=bfs_info.parent_position,
                    sibling_index=bfs_info.sibling_index,
                    path_from_root=dfs_info.path_from_root
                )
        
        return hybrid_positions
    
    def encode_structure(self, html_structure: Dict[str, Any]) -> Tuple[Dict[str, TreeNode], Dict[str, PositionInfo]]:
        """
        Encode HTML structure with tree-aware positions.
        
        Args:
            html_structure: Nested HTML structure dictionary
            
        Returns:
            Tuple of (nodes, position_info)
        """
        # Build tree representation
        nodes = self.build_tree_from_structure(html_structure)
        
        # Get positional information based on strategy
        if self.encoding_strategy == 'bfs':
            positions = self.get_bfs_positions(nodes)
        elif self.encoding_strategy == 'dfs':
            positions = self.get_dfs_positions(nodes)
        elif self.encoding_strategy == 'hybrid':
            positions = self.get_hybrid_positions(nodes)
        elif self.encoding_strategy == 'path_based':
            positions = self.get_path_based_positions(nodes)
        else:
            raise ValueError(f"Unknown encoding strategy: {self.encoding_strategy}")
        
        # Update statistics
        self._update_tree_stats(nodes, positions)
        
        return nodes, positions
    
    def get_path_based_positions(self, nodes: Dict[str, TreeNode]) -> Dict[str, PositionInfo]:
        """
        Get path-based positions encoding route from root.
        
        Args:
            nodes: Dictionary of TreeNode objects
            
        Returns:
            Dictionary mapping node IDs to PositionInfo with path-based encoding
        """
        positions = {}
        
        # Find root nodes
        root_nodes = [node for node in nodes.values() if node.parent_id is None]
        
        if not root_nodes:
            return positions
        
        def encode_path(node_id: str, sibling_index: int, path: List[int], position: int):
            node = nodes[node_id]
            
            # Encode path as position
            path_encoding = 0
            for i, step in enumerate(path):
                path_encoding += step * (self.max_nodes ** i)
            
            parent_position = None
            if node.parent_id and node.parent_id in positions:
                parent_position = positions[node.parent_id].bfs_position
            
            positions[node_id] = PositionInfo(
                node_id=node_id,
                bfs_position=path_encoding,
                dfs_position=position,
                depth=node.depth,
                parent_position=parent_position,
                sibling_index=sibling_index,
                path_from_root=path.copy()
            )
            
            # Process children
            for i, child_id in enumerate(node.children_ids):
                child_path = path + [i]
                encode_path(child_id, i, child_path, position + i + 1)
        
        # Start encoding from roots
        for i, root in enumerate(root_nodes):
            encode_path(root.node_id, i, [i], i)
        
        return positions
    
    def generate_position_embeddings(self, positions: Dict[str, PositionInfo]) -> torch.Tensor:
        """
        Generate position embeddings for tree nodes.
        
        Args:
            positions: Dictionary of PositionInfo objects
            
        Returns:
            Position embeddings tensor of shape (num_nodes, embedding_dim)
        """
        num_nodes = len(positions)
        
        if self.current_embedding_type == 'learnable':
            return self._generate_learnable_embeddings(positions)
        elif self.current_embedding_type == 'sinusoidal':
            return self._generate_sinusoidal_embeddings(positions)
        elif self.current_embedding_type == 'relative':
            return self._generate_relative_embeddings(positions)
        else:
            raise ValueError(f"Unknown embedding type: {self.current_embedding_type}")
    
    def _generate_learnable_embeddings(self, positions: Dict[str, PositionInfo]) -> torch.Tensor:
        """Generate learnable position embeddings."""
        num_nodes = len(positions)
        
        # Initialize random embeddings
        embeddings = torch.randn(num_nodes, self.embedding_dim) * 0.02
        
        return embeddings
    
    def _generate_sinusoidal_embeddings(self, positions: Dict[str, PositionInfo]) -> torch.Tensor:
        """Generate sinusoidal position embeddings."""
        num_nodes = len(positions)
        embeddings = torch.zeros(num_nodes, self.embedding_dim)
        
        # Get position values based on strategy
        position_values = []
        for pos_info in positions.values():
            if self.encoding_strategy == 'bfs':
                position_values.append(pos_info.bfs_position)
            elif self.encoding_strategy == 'dfs':
                position_values.append(pos_info.dfs_position)
            else:
                position_values.append(pos_info.bfs_position)
        
        position_tensor = torch.tensor(position_values, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * 
                           (-math.log(10000.0) / self.embedding_dim))
        
        # Apply sine to even indices
        embeddings[:, 0::2] = torch.sin(position_tensor * div_term)
        
        # Apply cosine to odd indices
        if self.embedding_dim % 2 == 1:
            embeddings[:, 1::2] = torch.cos(position_tensor * div_term)
        else:
            embeddings[:, 1::2] = torch.cos(position_tensor * div_term)
        
        return embeddings
    
    def _generate_relative_embeddings(self, positions: Dict[str, PositionInfo]) -> torch.Tensor:
        """Generate relative position embeddings."""
        num_nodes = len(positions)
        embeddings = torch.zeros(num_nodes, self.embedding_dim)
        
        # Encode relative relationships
        for i, pos_info in enumerate(positions.values()):
            # Encode depth
            depth_component = pos_info.depth / self.max_depth
            
            # Encode sibling index
            sibling_component = pos_info.sibling_index / 10.0  # Normalize
            
            # Encode parent relationship
            parent_component = 1.0 if pos_info.parent_position is not None else 0.0
            
            # Combine components
            embeddings[i, 0] = depth_component
            embeddings[i, 1] = sibling_component
            embeddings[i, 2] = parent_component
            
            # Fill remaining dimensions with sinusoidal encoding
            if self.embedding_dim > 3:
                pos_val = pos_info.bfs_position
                for j in range(3, self.embedding_dim, 2):
                    if j < self.embedding_dim:
                        embeddings[i, j] = math.sin(pos_val / (10000 ** (j / self.embedding_dim)))
                    if j + 1 < self.embedding_dim:
                        embeddings[i, j + 1] = math.cos(pos_val / (10000 ** (j / self.embedding_dim)))
        
        return embeddings
    
    def _update_tree_stats(self, nodes: Dict[str, TreeNode], positions: Dict[str, PositionInfo]):
        """Update tree statistics."""
        depths = [node.depth for node in nodes.values()]
        children_counts = [len(node.children_ids) for node in nodes.values()]
        
        self.tree_stats = {
            'total_nodes': len(nodes),
            'max_depth': max(depths) if depths else 0,
            'average_depth': sum(depths) / len(depths) if depths else 0,
            'max_children': max(children_counts) if children_counts else 0,
            'average_children': sum(children_counts) / len(children_counts) if children_counts else 0,
            'leaf_nodes': sum(1 for count in children_counts if count == 0),
            'internal_nodes': sum(1 for count in children_counts if count > 0)
        }
    
    def get_tree_statistics(self) -> Dict[str, Any]:
        """Get tree statistics."""
        return self.tree_stats.copy()
    
    def set_encoding_strategy(self, strategy: str):
        """Set the tree encoding strategy."""
        if strategy not in self.supported_strategies:
            raise ValueError(f"Strategy must be one of {self.supported_strategies}")
        self.encoding_strategy = strategy
        logger.info(f"Set encoding strategy to: {strategy}")
    
    def set_embedding_type(self, embedding_type: str):
        """Set the position embedding type."""
        if embedding_type not in self.position_embedding_types:
            raise ValueError(f"Embedding type must be one of {self.position_embedding_types}")
        self.current_embedding_type = embedding_type
        logger.info(f"Set embedding type to: {embedding_type}")
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Get encoder configuration information."""
        return {
            'max_depth': self.max_depth,
            'max_nodes': self.max_nodes,
            'embedding_dim': self.embedding_dim,
            'encoding_strategy': self.encoding_strategy,
            'embedding_type': self.current_embedding_type,
            'supported_strategies': self.supported_strategies,
            'supported_embedding_types': self.position_embedding_types,
            'tree_stats': self.tree_stats
        }


if __name__ == "__main__":
    # Example usage
    encoder = TreeEncoder(encoding_strategy='hybrid')
    
    # Test structure
    test_structure = {
        "div@container": {
            "header@main-header": {
                "nav@navigation": {
                    "ul@nav-list": {
                        "li@nav-item": {"text": "Home"},
                        "li@nav-item-2": {"text": "About"}
                    }
                }
            },
            "main@content": {
                "section@hero": {
                    "h1@title": {"text": "Welcome"},
                    "p@description": {"text": "Description"}
                }
            }
        }
    }
    
    print("=== Tree Encoding Example ===")
    nodes, positions = encoder.encode_structure(test_structure)
    
    print(f"Total nodes: {len(nodes)}")
    print(f"Tree stats: {encoder.get_tree_statistics()}")
    
    # Generate embeddings
    embeddings = encoder.generate_position_embeddings(positions)
    print(f"Position embeddings shape: {embeddings.shape}")
    
    # Show some position info
    print(f"\n=== Position Information ===")
    for node_id, pos_info in list(positions.items())[:3]:
        print(f"Node {node_id}:")
        print(f"  BFS position: {pos_info.bfs_position}")
        print(f"  DFS position: {pos_info.dfs_position}")
        print(f"  Depth: {pos_info.depth}")
        print(f"  Path: {pos_info.path_from_root}")
        print("---") 