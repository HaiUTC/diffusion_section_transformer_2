"""
HTML Structure Tokenizer

Main interface for tokenizing and encoding HTML structures.
Combines CSS parsing, subword tokenization, tree encoding, and embeddings.
"""

import torch
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

from .css_parser import CSSClassParser
from .subword_tokenizer import SubwordTokenizer
from .tree_encoder import TreeEncoder
from .embedding_handler import EmbeddingHandler
from .vocabulary_builder import VocabularyBuilder

logger = logging.getLogger(__name__)


@dataclass
class TokenizedStructure:
    """Contains the complete tokenized representation of an HTML structure."""
    
    # Original structure
    html_structure: Dict[str, Any]
    
    # Parsed components
    css_components: List[Dict[str, Any]]
    tree_nodes: Dict[str, Any]
    position_info: Dict[str, Any]
    
    # Tokenized sequences
    token_ids: List[int]
    token_strings: List[str]
    
    # Embeddings
    node_embeddings: torch.Tensor
    position_embeddings: torch.Tensor
    combined_embeddings: torch.Tensor
    
    # Metadata
    sequence_length: int
    num_nodes: int
    max_depth: int
    vocabulary_coverage: float


class HTMLStructureTokenizer:
    """
    Complete HTML structure tokenization pipeline.
    
    Integrates CSS parsing, subword tokenization, tree encoding, and embeddings
    to produce rich representations suitable for transformer models.
    """
    
    def __init__(self,
                 vocab_size: int = 50000,
                 embedding_dim: int = 768,
                 max_sequence_length: int = 512,
                 tree_encoding_strategy: str = 'bfs',
                 use_hash_embeddings: bool = True):
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_sequence_length = max_sequence_length
        self.tree_encoding_strategy = tree_encoding_strategy
        self.use_hash_embeddings = use_hash_embeddings
        
        # Initialize components
        self.css_parser = CSSClassParser()
        self.subword_tokenizer = SubwordTokenizer(vocab_size=vocab_size)
        self.tree_encoder = TreeEncoder(
            embedding_dim=embedding_dim,
            encoding_strategy=tree_encoding_strategy
        )
        self.embedding_handler = EmbeddingHandler(
            embedding_dim=embedding_dim,
            use_hash_embeddings=use_hash_embeddings
        )
        self.vocabulary_builder = VocabularyBuilder()
        
        # Training state
        self.is_trained = False
        self.training_stats = {}
        
        logger.info(f"HTMLStructureTokenizer initialized with {vocab_size} vocab size")
    
    def train(self, html_structures: List[Dict[str, Any]]) -> None:
        """
        Train the tokenizer on a dataset of HTML structures.
        
        Args:
            html_structures: List of HTML structure dictionaries
        """
        logger.info(f"Training tokenizer on {len(html_structures)} structures")
        
        # Extract all CSS class strings for subword tokenizer training
        css_strings = []
        all_decomposed = []
        
        for structure in html_structures:
            # Decompose structure
            decomposed = self.css_parser.decompose_structure(structure)
            all_decomposed.extend(decomposed)
            
            # Extract CSS class strings
            for element in decomposed:
                for css_comp in element['css_components']:
                    if css_comp.get('raw_class'):
                        css_strings.append(css_comp['raw_class'])
        
        logger.info(f"Extracted {len(css_strings)} CSS class strings")
        
        # Train subword tokenizer
        unique_css_strings = list(set(css_strings))
        self.subword_tokenizer.train(unique_css_strings)
        
        # Build tag vocabulary for embedding handler
        self.embedding_handler.build_tag_vocabulary(html_structures)
        
        # Build comprehensive vocabulary
        self.vocabulary_builder.build_from_structures(html_structures, self.css_parser)
        
        # Update training state
        self.is_trained = True
        
        # Compile training statistics
        self.training_stats = {
            'num_training_structures': len(html_structures),
            'num_css_strings': len(css_strings),
            'unique_css_strings': len(unique_css_strings),
            'num_decomposed_elements': len(all_decomposed),
            'subword_tokenizer_stats': self.subword_tokenizer.get_training_stats(),
            'css_parsing_stats': self.css_parser.get_parsing_statistics(html_structures),
            'vocabulary_stats': self.vocabulary_builder.get_statistics(),
            'embedding_stats': self.embedding_handler.get_embedding_statistics()
        }
        
        logger.info("Training completed successfully")
    
    def tokenize(self, html_structure: Dict[str, Any]) -> TokenizedStructure:
        """
        Tokenize a single HTML structure.
        
        Args:
            html_structure: HTML structure dictionary
            
        Returns:
            TokenizedStructure with complete tokenization
        """
        if not self.is_trained:
            raise RuntimeError("Tokenizer must be trained before use. Call train() first.")
        
        # Step 1: Parse CSS components
        css_components = self.css_parser.decompose_structure(html_structure)
        
        # Step 2: Build tree representation and get positions
        tree_nodes, position_info = self.tree_encoder.encode_structure(html_structure)
        
        # Step 3: Generate token sequences
        token_ids, token_strings = self._generate_token_sequence(css_components, tree_nodes)
        
        # Step 4: Generate embeddings
        embeddings = self._generate_embeddings(css_components, tree_nodes, position_info)
        
        # Step 5: Calculate metadata
        vocab_coverage = self._calculate_vocabulary_coverage(token_strings)
        max_depth = max([pos.depth for pos in position_info.values()]) if position_info else 0
        
        return TokenizedStructure(
            html_structure=html_structure,
            css_components=css_components,
            tree_nodes={k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in tree_nodes.items()},
            position_info={k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in position_info.items()},
            token_ids=token_ids,
            token_strings=token_strings,
            node_embeddings=embeddings['node_embeddings'],
            position_embeddings=embeddings['position_embeddings'],
            combined_embeddings=embeddings['combined_embeddings'],
            sequence_length=len(token_ids),
            num_nodes=len(tree_nodes),
            max_depth=max_depth,
            vocabulary_coverage=vocab_coverage
        )
    
    def _generate_token_sequence(self, css_components: List[Dict[str, Any]], 
                               tree_nodes: Dict[str, Any]) -> Tuple[List[int], List[str]]:
        """Generate token sequence from parsed components."""
        token_strings = ['[CLS]']  # Start with classification token
        
        # Process each element in tree order
        for element in css_components:
            # Add tag token
            tag = element.get('tag', 'unknown')
            token_strings.append(f"[TAG]{tag}")
            
            # Add CSS class tokens
            for css_comp in element.get('css_components', []):
                raw_class = css_comp.get('raw_class', '')
                if raw_class:
                    # Use subword tokenization for CSS classes
                    css_token_ids = self.subword_tokenizer.encode(raw_class, add_special_tokens=False)
                    css_tokens = [self.subword_tokenizer.decode([tid]) for tid in css_token_ids]
                    token_strings.extend([f"[CSS]{token}" for token in css_tokens if token])
            
            # Add content tokens if present
            content = element.get('content', {})
            for content_type, content_value in content.items():
                if content_value:
                    token_strings.append(f"[CONTENT]{content_type}:{content_value[:50]}")  # Truncate long content
            
            # Add separator
            token_strings.append('[SEP]')
        
        # Truncate if too long
        if len(token_strings) > self.max_sequence_length:
            token_strings = token_strings[:self.max_sequence_length-1] + ['[EOS]']
        else:
            token_strings.append('[EOS]')
        
        # Convert to token IDs using hash embeddings or vocabulary
        if self.use_hash_embeddings:
            # Use string hashes as IDs for simplicity
            token_ids = [hash(token) % self.vocab_size for token in token_strings]
        else:
            # Use vocabulary mapping
            vocab = self.vocabulary_builder.get_token_vocabulary()
            token_ids = [vocab.get(token, vocab.get('[UNK]', 1)) for token in token_strings]
        
        return token_ids, token_strings
    
    def _generate_embeddings(self, css_components: List[Dict[str, Any]], 
                           tree_nodes: Dict[str, Any], 
                           position_info: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Generate embeddings for all components."""
        
        # Generate position embeddings
        position_embeddings = self.tree_encoder.generate_position_embeddings(position_info)
        
        # Generate node embeddings
        node_embeddings_list = []
        
        for element in css_components:
            # Tag embedding
            tag = element.get('tag', 'unknown')
            tag_emb = self.embedding_handler.encode_tag(tag)
            
            # CSS embeddings
            css_comps = element.get('css_components', [])
            css_emb = self.embedding_handler.encode_css_components(css_comps)
            
            # Content embedding
            content = element.get('content', {})
            content_emb = self.embedding_handler.encode_content(content)
            
            # Position embedding (get corresponding position)
            pos_emb = torch.zeros(1, self.embedding_dim)  # Default
            element_path = element.get('path', '')
            for pos_info in position_info.values():
                if hasattr(pos_info, 'node_id') and element_path.endswith(pos_info.node_id):
                    node_idx = list(position_info.keys()).index(pos_info.node_id)
                    if node_idx < position_embeddings.size(0):
                        pos_emb = position_embeddings[node_idx:node_idx+1]
                    break
            
            # Combine embeddings
            combined_emb = self.embedding_handler.combine_embeddings(
                tag_emb, css_emb, content_emb, pos_emb
            )
            
            node_embeddings_list.append(combined_emb)
        
        # Stack node embeddings
        if node_embeddings_list:
            node_embeddings = torch.cat(node_embeddings_list, dim=0)
        else:
            node_embeddings = torch.zeros(0, self.embedding_dim)
        
        # Combined embeddings (node + position)
        if node_embeddings.size(0) > 0 and position_embeddings.size(0) > 0:
            min_size = min(node_embeddings.size(0), position_embeddings.size(0))
            combined_embeddings = node_embeddings[:min_size] + position_embeddings[:min_size]
        else:
            combined_embeddings = node_embeddings
        
        return {
            'node_embeddings': node_embeddings,
            'position_embeddings': position_embeddings,
            'combined_embeddings': combined_embeddings
        }
    
    def _calculate_vocabulary_coverage(self, token_strings: List[str]) -> float:
        """Calculate vocabulary coverage for the tokenized sequence."""
        if not token_strings:
            return 0.0
        
        # Check how many tokens are unknown
        vocab = self.vocabulary_builder.get_token_vocabulary()
        unknown_count = sum(1 for token in token_strings if token not in vocab)
        
        return 1.0 - (unknown_count / len(token_strings))
    
    def tokenize_batch(self, html_structures: List[Dict[str, Any]]) -> List[TokenizedStructure]:
        """
        Tokenize a batch of HTML structures.
        
        Args:
            html_structures: List of HTML structure dictionaries
            
        Returns:
            List of TokenizedStructure objects
        """
        return [self.tokenize(structure) for structure in html_structures]
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to string representation.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string
        """
        if self.use_hash_embeddings:
            # Simple decoding for hash-based tokens
            return f"<hashed_tokens_{len(token_ids)}>"
        else:
            vocab = self.vocabulary_builder.get_id_to_token_mapping()
            tokens = [vocab.get(tid, '[UNK]') for tid in token_ids]
            return ' '.join(tokens)
    
    def get_tokenizer_info(self) -> Dict[str, Any]:
        """Get comprehensive tokenizer information."""
        return {
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'max_sequence_length': self.max_sequence_length,
            'tree_encoding_strategy': self.tree_encoding_strategy,
            'use_hash_embeddings': self.use_hash_embeddings,
            'is_trained': self.is_trained,
            'training_stats': self.training_stats,
            'component_info': {
                'css_parser': type(self.css_parser).__name__,
                'subword_tokenizer': type(self.subword_tokenizer).__name__,
                'tree_encoder': type(self.tree_encoder).__name__,
                'embedding_handler': type(self.embedding_handler).__name__
            }
        }
    
    def save_tokenizer(self, filepath: str) -> None:
        """
        Save trained tokenizer to file.
        
        Args:
            filepath: Path to save the tokenizer
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained tokenizer")
        
        # Save subword tokenizer vocabulary
        subword_vocab_path = filepath.replace('.json', '_subword.json')
        self.subword_tokenizer.save_vocabulary(subword_vocab_path)
        
        # Save main vocabulary
        vocab_path = filepath.replace('.json', '_vocab.json')
        self.vocabulary_builder.save_vocabulary(vocab_path)
        
        logger.info(f"Tokenizer saved to {filepath}")
    
    def load_tokenizer(self, filepath: str) -> None:
        """
        Load trained tokenizer from file.
        
        Args:
            filepath: Path to load the tokenizer from
        """
        # Load subword tokenizer vocabulary
        subword_vocab_path = filepath.replace('.json', '_subword.json')
        self.subword_tokenizer.load_vocabulary(subword_vocab_path)
        
        # Load main vocabulary
        vocab_path = filepath.replace('.json', '_vocab.json')
        self.vocabulary_builder.load_vocabulary(vocab_path)
        
        self.is_trained = True
        logger.info(f"Tokenizer loaded from {filepath}")
    
    def analyze_structure(self, html_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an HTML structure without full tokenization.
        
        Args:
            html_structure: HTML structure dictionary
            
        Returns:
            Analysis results
        """
        # Parse components
        css_components = self.css_parser.decompose_structure(html_structure)
        tree_nodes, position_info = self.tree_encoder.encode_structure(html_structure)
        
        # Basic statistics
        css_classes = []
        tags = []
        content_types = set()
        
        for element in css_components:
            tags.append(element.get('tag', 'unknown'))
            content_types.update(element.get('content', {}).keys())
            
            for css_comp in element.get('css_components', []):
                if css_comp.get('raw_class'):
                    css_classes.append(css_comp['raw_class'])
        
        return {
            'num_elements': len(css_components),
            'num_nodes': len(tree_nodes),
            'max_depth': max([pos.depth for pos in position_info.values()]) if position_info else 0,
            'unique_tags': len(set(tags)),
            'unique_css_classes': len(set(css_classes)),
            'content_types': list(content_types),
            'tree_stats': self.tree_encoder.get_tree_statistics(),
            'css_parsing_stats': self.css_parser.get_parsing_statistics([html_structure])
        }


if __name__ == "__main__":
    # Example usage
    tokenizer = HTMLStructureTokenizer(vocab_size=1000, embedding_dim=256)
    
    # Sample structures for training
    training_structures = [
        {
            "div@container@mx_auto": {
                "h1@text_center@fs_24x": {"text": "Welcome"},
                "p@text_gray@fs_16x": {"text": "Description"}
            }
        },
        {
            "section@hero@bg_blue": {
                "div@flex@jc_center": {
                    "img@w_100p": {"src": "image.jpg"}
                }
            }
        }
    ]
    
    print("=== Training Tokenizer ===")
    tokenizer.train(training_structures)
    
    print(f"Training completed. Stats: {tokenizer.get_tokenizer_info()}")
    
    # Test tokenization
    test_structure = training_structures[0]
    
    print(f"\n=== Tokenization Example ===")
    result = tokenizer.tokenize(test_structure)
    
    print(f"Sequence length: {result.sequence_length}")
    print(f"Number of nodes: {result.num_nodes}")
    print(f"Max depth: {result.max_depth}")
    print(f"Vocabulary coverage: {result.vocabulary_coverage:.3f}")
    print(f"Token strings: {result.token_strings[:10]}...")  # First 10 tokens
    print(f"Embeddings shape: {result.combined_embeddings.shape}")
    
    # Test analysis
    print(f"\n=== Structure Analysis ===")
    analysis = tokenizer.analyze_structure(test_structure)
    for key, value in analysis.items():
        if isinstance(value, dict):
            print(f"{key}: {len(value)} items")
        else:
            print(f"{key}: {value}") 