"""
Embedding Handler

Manages multi-modal embeddings for HTML structures including compositional
embeddings for CSS attributes and hash embeddings for large vocabularies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import hashlib
import math

logger = logging.getLogger(__name__)


class HashEmbedding(nn.Module):
    """
    Memory-efficient hash-based embeddings for large vocabularies.
    
    Reduces memory requirements by hashing tokens to a smaller embedding table.
    Multiple tokens may map to the same embedding, but this rarely hurts performance.
    """
    
    def __init__(self, num_embeddings: int, embedding_dim: int, num_hashes: int = 2):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.num_hashes = num_hashes
        
        # Create multiple hash tables for better distribution
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings, embedding_dim // num_hashes)
            for _ in range(num_hashes)
        ])
        
        # Initialize embeddings
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
    
    def hash_token(self, token: str, hash_index: int) -> int:
        """Hash a token to an embedding index."""
        # Use different hash functions for each hash table
        hash_input = f"{token}_{hash_index}".encode('utf-8')
        hash_value = int(hashlib.md5(hash_input).hexdigest(), 16)
        return hash_value % self.num_embeddings
    
    def forward(self, tokens: List[str]) -> torch.Tensor:
        """
        Get embeddings for tokens.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Tensor of shape (len(tokens), embedding_dim)
        """
        batch_size = len(tokens)
        embeddings_list = []
        
        for i, embedding_layer in enumerate(self.embeddings):
            # Hash all tokens for this hash function
            indices = torch.tensor([
                self.hash_token(token, i) for token in tokens
            ], dtype=torch.long)
            
            # Get embeddings
            hash_embeddings = embedding_layer(indices)
            embeddings_list.append(hash_embeddings)
        
        # Concatenate embeddings from all hash functions
        final_embeddings = torch.cat(embeddings_list, dim=1)
        
        return final_embeddings


class CompositionalEmbedding(nn.Module):
    """
    Compositional embeddings for CSS attribute components.
    
    Combines embeddings for different attribute parts (device, pseudo, attribute, value, unit)
    using learned composition functions.
    """
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 component_dim: int = 128,
                 use_attention: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.component_dim = component_dim
        self.use_attention = use_attention
        
        # Component embeddings
        self.device_embedding = nn.Embedding(10, component_dim)  # t, m, d, l, xl
        self.pseudo_embedding = nn.Embedding(10, component_dim)  # h, f, a, v, d, e
        self.attribute_embedding = nn.Embedding(100, component_dim)  # pt, fs, c, etc.
        self.unit_embedding = nn.Embedding(20, component_dim)  # r, x, p, e, etc.
        
        # Value embedding (for numeric values)
        self.value_projection = nn.Linear(1, component_dim)
        
        # Composition layers
        if use_attention:
            self.composition_attention = nn.MultiheadAttention(
                embed_dim=component_dim,
                num_heads=4,
                batch_first=True
            )
        else:
            self.composition_layer = nn.Linear(component_dim * 5, embedding_dim)
        
        # Final projection
        self.output_projection = nn.Linear(component_dim, embedding_dim)
        
        # Initialize embeddings
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding layers."""
        for embedding in [self.device_embedding, self.pseudo_embedding, 
                         self.attribute_embedding, self.unit_embedding]:
            nn.init.normal_(embedding.weight, std=0.02)
        
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)
    
    def encode_component(self, component_dict: Dict[str, Any]) -> torch.Tensor:
        """
        Encode a single CSS component dictionary.
        
        Args:
            component_dict: Dictionary with keys like device_context, attribute, etc.
            
        Returns:
            Tensor of shape (1, embedding_dim)
        """
        components = []
        
        # Device context
        device = component_dict.get('device_context')
        if device:
            device_id = hash(device) % 10
            device_emb = self.device_embedding(torch.tensor([device_id]))
        else:
            device_emb = torch.zeros(1, self.component_dim)
        components.append(device_emb)
        
        # Pseudo context
        pseudo = component_dict.get('pseudo_context')
        if pseudo:
            pseudo_id = hash(pseudo) % 10
            pseudo_emb = self.pseudo_embedding(torch.tensor([pseudo_id]))
        else:
            pseudo_emb = torch.zeros(1, self.component_dim)
        components.append(pseudo_emb)
        
        # Attribute
        attribute = component_dict.get('attribute')
        if attribute:
            attr_id = hash(attribute) % 100
            attr_emb = self.attribute_embedding(torch.tensor([attr_id]))
        else:
            attr_emb = torch.zeros(1, self.component_dim)
        components.append(attr_emb)
        
        # Value
        value = component_dict.get('value')
        if value and value.replace('.', '').replace('-', '').isdigit():
            try:
                value_float = float(value)
                value_emb = self.value_projection(torch.tensor([[value_float]]))
            except ValueError:
                value_emb = torch.zeros(1, self.component_dim)
        else:
            value_emb = torch.zeros(1, self.component_dim)
        components.append(value_emb)
        
        # Unit
        unit = component_dict.get('unit')
        if unit:
            unit_id = hash(unit) % 20
            unit_emb = self.unit_embedding(torch.tensor([unit_id]))
        else:
            unit_emb = torch.zeros(1, self.component_dim)
        components.append(unit_emb)
        
        # Compose components
        if self.use_attention:
            # Stack components for attention
            component_stack = torch.stack(components, dim=1)  # (1, 5, component_dim)
            
            # Apply self-attention
            attended, _ = self.composition_attention(
                component_stack, component_stack, component_stack
            )
            
            # Pool attended representations
            composed = torch.mean(attended, dim=1)  # (1, component_dim)
        else:
            # Simple concatenation and projection
            concatenated = torch.cat(components, dim=1)  # (1, component_dim * 5)
            composed = self.composition_layer(concatenated)  # (1, embedding_dim)
            return composed
        
        # Final projection
        output = self.output_projection(composed)  # (1, embedding_dim)
        
        return output
    
    def forward(self, component_dicts: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode a batch of CSS components.
        
        Args:
            component_dicts: List of component dictionaries
            
        Returns:
            Tensor of shape (len(component_dicts), embedding_dim)
        """
        embeddings = []
        
        for component_dict in component_dicts:
            emb = self.encode_component(component_dict)
            embeddings.append(emb)
        
        return torch.cat(embeddings, dim=0)


class EmbeddingHandler:
    """
    Main handler for all embedding types in HTML structure encoding.
    
    Manages tag embeddings, compositional CSS embeddings, content embeddings,
    and position embeddings with memory-efficient strategies.
    """
    
    def __init__(self,
                 embedding_dim: int = 768,
                 max_vocab_size: int = 50000,
                 use_hash_embeddings: bool = True,
                 hash_embedding_size: int = 10000,
                 compositional_dim: int = 128):
        
        self.embedding_dim = embedding_dim
        self.max_vocab_size = max_vocab_size
        self.use_hash_embeddings = use_hash_embeddings
        self.hash_embedding_size = hash_embedding_size
        
        # Tag vocabulary and embeddings
        self.tag_vocab = {}
        self.tag_embedding = None
        
        # Content embeddings
        self.content_embedding_dim = embedding_dim // 4  # Smaller for content
        
        # Compositional CSS embeddings
        self.css_embedding = CompositionalEmbedding(
            embedding_dim=embedding_dim,
            component_dim=compositional_dim
        )
        
        # Hash embeddings for large vocabularies
        if use_hash_embeddings:
            self.hash_embedding = HashEmbedding(
                num_embeddings=hash_embedding_size,
                embedding_dim=embedding_dim
            )
        
        # Special token embeddings
        self.special_tokens = {
            '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3,
            '[MASK]': 4, '[TAG]': 5, '[CSS]': 6, '[CONTENT]': 7
        }
        
        # Statistics
        self.embedding_stats = {}
        
        logger.info(f"EmbeddingHandler initialized with {embedding_dim}D embeddings")
    
    def build_tag_vocabulary(self, html_structures: List[Dict[str, Any]]) -> None:
        """
        Build vocabulary for HTML tags from training data.
        
        Args:
            html_structures: List of HTML structure dictionaries
        """
        tag_counts = {}
        
        def extract_tags(structure: Dict[str, Any]):
            for key, value in structure.items():
                if key in ['text', 'src', 'svg']:
                    continue
                
                # Extract tag
                tag = key.split('@')[0] if '@' in key else key
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
                
                # Recurse
                if isinstance(value, dict):
                    extract_tags(value)
        
        # Extract tags from all structures
        for structure in html_structures:
            extract_tags(structure)
        
        # Build vocabulary
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Add special tokens first
        self.tag_vocab = self.special_tokens.copy()
        
        # Add most frequent tags
        for tag, count in sorted_tags:
            if len(self.tag_vocab) >= self.max_vocab_size:
                break
            if tag not in self.tag_vocab:
                self.tag_vocab[tag] = len(self.tag_vocab)
        
        # Create embedding layer
        self.tag_embedding = nn.Embedding(len(self.tag_vocab), self.embedding_dim)
        nn.init.normal_(self.tag_embedding.weight, std=0.02)
        
        logger.info(f"Built tag vocabulary with {len(self.tag_vocab)} tags")
    
    def encode_tag(self, tag: str) -> torch.Tensor:
        """
        Encode a single HTML tag.
        
        Args:
            tag: HTML tag string
            
        Returns:
            Tag embedding tensor of shape (1, embedding_dim)
        """
        if self.tag_embedding is None:
            raise RuntimeError("Tag vocabulary not built. Call build_tag_vocabulary() first.")
        
        tag_id = self.tag_vocab.get(tag, self.tag_vocab['[UNK]'])
        tag_tensor = torch.tensor([tag_id], dtype=torch.long)
        
        return self.tag_embedding(tag_tensor)
    
    def encode_css_components(self, css_components: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Encode CSS components using compositional embeddings.
        
        Args:
            css_components: List of CSS component dictionaries
            
        Returns:
            CSS embeddings tensor of shape (len(css_components), embedding_dim)
        """
        if not css_components:
            return torch.zeros(0, self.embedding_dim)
        
        return self.css_embedding(css_components)
    
    def encode_content(self, content_dict: Dict[str, str]) -> torch.Tensor:
        """
        Encode content (text, src, svg) using simple embeddings.
        
        Args:
            content_dict: Dictionary with content information
            
        Returns:
            Content embedding tensor of shape (1, embedding_dim)
        """
        # Simple content encoding - can be improved with pre-trained embeddings
        content_features = []
        
        # Text content
        if 'text' in content_dict:
            text = content_dict['text']
            # Simple hash-based encoding for text
            text_hash = hash(text) % 1000
            content_features.extend([1.0, float(text_hash) / 1000.0, len(text) / 100.0])
        else:
            content_features.extend([0.0, 0.0, 0.0])
        
        # Source content
        if 'src' in content_dict:
            src = content_dict['src']
            src_hash = hash(src) % 1000
            content_features.extend([1.0, float(src_hash) / 1000.0])
        else:
            content_features.extend([0.0, 0.0])
        
        # SVG content
        if 'svg' in content_dict:
            svg = content_dict['svg']
            svg_hash = hash(svg) % 1000
            content_features.extend([1.0, float(svg_hash) / 1000.0])
        else:
            content_features.extend([0.0, 0.0])
        
        # Pad to embedding dimension
        while len(content_features) < self.embedding_dim:
            content_features.append(0.0)
        
        # Truncate if too long
        content_features = content_features[:self.embedding_dim]
        
        return torch.tensor([content_features], dtype=torch.float)
    
    def encode_tokens_with_hash(self, tokens: List[str]) -> torch.Tensor:
        """
        Encode tokens using hash embeddings for memory efficiency.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Token embeddings tensor of shape (len(tokens), embedding_dim)
        """
        if not self.use_hash_embeddings:
            raise RuntimeError("Hash embeddings not enabled")
        
        return self.hash_embedding(tokens)
    
    def combine_embeddings(self, 
                          tag_embedding: torch.Tensor,
                          css_embeddings: torch.Tensor,
                          content_embedding: torch.Tensor,
                          position_embedding: torch.Tensor) -> torch.Tensor:
        """
        Combine different types of embeddings for a node.
        
        Args:
            tag_embedding: Tag embedding (1, embedding_dim)
            css_embeddings: CSS embeddings (num_classes, embedding_dim)
            content_embedding: Content embedding (1, embedding_dim)
            position_embedding: Position embedding (1, embedding_dim)
            
        Returns:
            Combined embedding (1, embedding_dim)
        """
        # Start with tag embedding
        combined = tag_embedding.clone()
        
        # Add CSS embeddings (average if multiple classes)
        if css_embeddings.size(0) > 0:
            css_avg = torch.mean(css_embeddings, dim=0, keepdim=True)
            combined = combined + css_avg
        
        # Add content embedding
        combined = combined + content_embedding
        
        # Add position embedding
        combined = combined + position_embedding
        
        # Normalize
        combined = F.layer_norm(combined, [self.embedding_dim])
        
        return combined
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        stats = {
            'embedding_dim': self.embedding_dim,
            'tag_vocab_size': len(self.tag_vocab) if self.tag_vocab else 0,
            'use_hash_embeddings': self.use_hash_embeddings,
            'hash_embedding_size': self.hash_embedding_size if self.use_hash_embeddings else 0,
        }
        
        # Add parameter counts
        total_params = 0
        if self.tag_embedding:
            total_params += sum(p.numel() for p in self.tag_embedding.parameters())
        
        total_params += sum(p.numel() for p in self.css_embedding.parameters())
        
        if self.use_hash_embeddings:
            total_params += sum(p.numel() for p in self.hash_embedding.parameters())
        
        stats['total_parameters'] = total_params
        stats['memory_mb'] = total_params * 4 / (1024 * 1024)  # Assume float32
        
        return stats


if __name__ == "__main__":
    # Example usage
    handler = EmbeddingHandler(embedding_dim=256)
    
    # Build tag vocabulary
    sample_structures = [
        {"div@container": {"h1@title": {"text": "Hello"}}},
        {"section@hero": {"p@description": {"text": "World"}}}
    ]
    
    handler.build_tag_vocabulary(sample_structures)
    
    # Test tag encoding
    tag_emb = handler.encode_tag("div")
    print(f"Tag embedding shape: {tag_emb.shape}")
    
    # Test CSS component encoding
    css_components = [
        {
            'device_context': 't',
            'attribute': 'pt',
            'value': '7',
            'unit': 'r'
        }
    ]
    
    css_emb = handler.encode_css_components(css_components)
    print(f"CSS embedding shape: {css_emb.shape}")
    
    # Test content encoding
    content = {"text": "Hello World"}
    content_emb = handler.encode_content(content)
    print(f"Content embedding shape: {content_emb.shape}")
    
    # Test hash embeddings
    if handler.use_hash_embeddings:
        tokens = ["pt_7r", "fs_18x", "c_blue"]
        hash_emb = handler.encode_tokens_with_hash(tokens)
        print(f"Hash embeddings shape: {hash_emb.shape}")
    
    print(f"Statistics: {handler.get_embedding_statistics()}") 