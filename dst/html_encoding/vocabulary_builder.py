"""
Vocabulary Builder

Builds and manages comprehensive vocabularies for HTML structure tokenization.
Handles tags, CSS attributes, content types, and special tokens.
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)


class VocabularyBuilder:
    """
    Builds comprehensive vocabularies for HTML structure tokenization.
    
    Manages token-to-ID mappings for all types of tokens including
    HTML tags, CSS attributes, content tokens, and special tokens.
    """
    
    def __init__(self, max_vocab_size: int = 50000):
        self.max_vocab_size = max_vocab_size
        
        # Special tokens (reserved IDs)
        self.special_tokens = {
            '[PAD]': 0,    # Padding token
            '[UNK]': 1,    # Unknown token
            '[CLS]': 2,    # Classification token
            '[SEP]': 3,    # Separator token
            '[MASK]': 4,   # Mask token for MLM
            '[BOS]': 5,    # Beginning of sequence
            '[EOS]': 6,    # End of sequence
            '[TAG]': 7,    # Tag prefix
            '[CSS]': 8,    # CSS prefix
            '[CONTENT]': 9 # Content prefix
        }
        
        # Vocabularies
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Statistics
        self.vocabulary_stats = {}
        
        # Token counts for frequency analysis
        self.token_counts = Counter()
        
        # Initialize with special tokens
        self._initialize_special_tokens()
        
        logger.info(f"VocabularyBuilder initialized with max size {max_vocab_size}")
    
    def _initialize_special_tokens(self):
        """Initialize vocabulary with special tokens."""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
    
    def build_from_structures(self, html_structures: List[Dict[str, Any]], 
                            css_parser) -> None:
        """
        Build vocabulary from HTML structures using CSS parser.
        
        Args:
            html_structures: List of HTML structure dictionaries
            css_parser: CSS parser instance for decomposing structures
        """
        logger.info(f"Building vocabulary from {len(html_structures)} structures")
        
        all_tokens = set()
        
        # Process each structure
        for i, structure in enumerate(html_structures):
            # Decompose structure
            decomposed = css_parser.decompose_structure(structure)
            
            # Extract tokens from decomposed elements
            structure_tokens = self._extract_tokens_from_decomposed(decomposed)
            all_tokens.update(structure_tokens)
            
            # Update token counts
            for token in structure_tokens:
                self.token_counts[token] += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(html_structures)} structures")
        
        # Build final vocabulary
        self._build_final_vocabulary(all_tokens)
        
        # Calculate statistics
        self._calculate_vocabulary_statistics(html_structures, css_parser)
        
        logger.info(f"Vocabulary built with {len(self.token_to_id)} tokens")
    
    def _extract_tokens_from_decomposed(self, decomposed_elements: List[Dict[str, Any]]) -> Set[str]:
        """Extract all possible tokens from decomposed elements."""
        tokens = set()
        
        for element in decomposed_elements:
            # HTML tag tokens
            tag = element.get('tag', '')
            if tag:
                tokens.add(f"[TAG]{tag}")
            
            # CSS component tokens
            for css_comp in element.get('css_components', []):
                # Raw class token
                raw_class = css_comp.get('raw_class', '')
                if raw_class:
                    tokens.add(f"[CSS]{raw_class}")
                
                # Component-level tokens
                for comp_type in ['device_context', 'pseudo_context', 'attribute', 'value', 'unit']:
                    comp_value = css_comp.get(comp_type)
                    if comp_value:
                        tokens.add(f"[CSS]{comp_type}:{comp_value}")
            
            # Content tokens
            content = element.get('content', {})
            for content_type, content_value in content.items():
                if content_value:
                    # Content type token
                    tokens.add(f"[CONTENT]{content_type}")
                    
                    # Content value token (truncated for manageability)
                    if len(str(content_value)) <= 50:
                        tokens.add(f"[CONTENT]{content_type}:{content_value}")
            
            # Structural tokens
            depth = element.get('depth', 0)
            tokens.add(f"[DEPTH]{depth}")
            
            if element.get('has_children'):
                tokens.add("[HAS_CHILDREN]")
            else:
                tokens.add("[LEAF_NODE]")
        
        return tokens
    
    def _build_final_vocabulary(self, all_tokens: Set[str]) -> None:
        """Build final vocabulary from all extracted tokens."""
        # Start with current vocabulary (special tokens)
        current_vocab_size = len(self.token_to_id)
        
        # Sort tokens by frequency (most frequent first)
        sorted_tokens = sorted(
            [(token, self.token_counts[token]) for token in all_tokens],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Add tokens until we reach max vocabulary size
        for token, count in sorted_tokens:
            if len(self.token_to_id) >= self.max_vocab_size:
                break
            
            if token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
        
        logger.info(f"Added {len(self.token_to_id) - current_vocab_size} tokens to vocabulary")
    
    def _calculate_vocabulary_statistics(self, html_structures: List[Dict[str, Any]], 
                                       css_parser) -> None:
        """Calculate comprehensive vocabulary statistics."""
        tag_counts = Counter()
        css_class_counts = Counter()
        content_type_counts = Counter()
        depth_counts = Counter()
        
        total_elements = 0
        total_css_components = 0
        
        for structure in html_structures:
            decomposed = css_parser.decompose_structure(structure)
            total_elements += len(decomposed)
            
            for element in decomposed:
                # Tag statistics
                tag = element.get('tag', '')
                if tag:
                    tag_counts[tag] += 1
                
                # CSS statistics
                css_components = element.get('css_components', [])
                total_css_components += len(css_components)
                
                for css_comp in css_components:
                    raw_class = css_comp.get('raw_class', '')
                    if raw_class:
                        css_class_counts[raw_class] += 1
                
                # Content statistics
                content = element.get('content', {})
                for content_type in content.keys():
                    content_type_counts[content_type] += 1
                
                # Depth statistics
                depth = element.get('depth', 0)
                depth_counts[depth] += 1
        
        # Token type analysis
        token_type_counts = defaultdict(int)
        for token in self.token_to_id.keys():
            if token.startswith('[TAG]'):
                token_type_counts['tag'] += 1
            elif token.startswith('[CSS]'):
                token_type_counts['css'] += 1
            elif token.startswith('[CONTENT]'):
                token_type_counts['content'] += 1
            elif token.startswith('[DEPTH]'):
                token_type_counts['depth'] += 1
            elif token in self.special_tokens:
                token_type_counts['special'] += 1
            else:
                token_type_counts['other'] += 1
        
        self.vocabulary_stats = {
            'total_tokens': len(self.token_to_id),
            'unique_tags': len(tag_counts),
            'unique_css_classes': len(css_class_counts),
            'unique_content_types': len(content_type_counts),
            'max_depth': max(depth_counts.keys()) if depth_counts else 0,
            'avg_elements_per_structure': total_elements / len(html_structures),
            'avg_css_components_per_element': total_css_components / max(total_elements, 1),
            'token_type_distribution': dict(token_type_counts),
            'most_frequent_tags': tag_counts.most_common(10),
            'most_frequent_css_classes': css_class_counts.most_common(10),
            'most_frequent_content_types': content_type_counts.most_common(5),
            'vocabulary_coverage': len(self.token_to_id) / self.max_vocab_size
        }
    
    def get_token_id(self, token: str) -> int:
        """
        Get token ID for a token string.
        
        Args:
            token: Token string
            
        Returns:
            Token ID (returns [UNK] ID if token not found)
        """
        return self.token_to_id.get(token, self.token_to_id['[UNK]'])
    
    def get_token_string(self, token_id: int) -> str:
        """
        Get token string for a token ID.
        
        Args:
            token_id: Token ID
            
        Returns:
            Token string (returns '[UNK]' if ID not found)
        """
        return self.id_to_token.get(token_id, '[UNK]')
    
    def encode_tokens(self, tokens: List[str]) -> List[int]:
        """
        Encode a list of token strings to IDs.
        
        Args:
            tokens: List of token strings
            
        Returns:
            List of token IDs
        """
        return [self.get_token_id(token) for token in tokens]
    
    def decode_tokens(self, token_ids: List[int]) -> List[str]:
        """
        Decode a list of token IDs to strings.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            List of token strings
        """
        return [self.get_token_string(token_id) for token_id in token_ids]
    
    def get_token_vocabulary(self) -> Dict[str, int]:
        """Get the complete token-to-ID vocabulary."""
        return self.token_to_id.copy()
    
    def get_id_to_token_mapping(self) -> Dict[int, str]:
        """Get the complete ID-to-token mapping."""
        return self.id_to_token.copy()
    
    def get_vocabulary_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.token_to_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        return self.vocabulary_stats.copy()
    
    def get_token_frequency(self, token: str) -> int:
        """Get frequency count for a token."""
        return self.token_counts.get(token, 0)
    
    def get_most_frequent_tokens(self, n: int = 50) -> List[Tuple[str, int]]:
        """Get the n most frequent tokens."""
        return self.token_counts.most_common(n)
    
    def analyze_token_coverage(self, tokens: List[str]) -> Dict[str, Any]:
        """
        Analyze vocabulary coverage for a list of tokens.
        
        Args:
            tokens: List of token strings
            
        Returns:
            Coverage analysis
        """
        if not tokens:
            return {'coverage': 0.0, 'unknown_tokens': [], 'known_tokens': []}
        
        known_tokens = []
        unknown_tokens = []
        
        for token in tokens:
            if token in self.token_to_id:
                known_tokens.append(token)
            else:
                unknown_tokens.append(token)
        
        coverage = len(known_tokens) / len(tokens)
        
        return {
            'coverage': coverage,
            'total_tokens': len(tokens),
            'known_tokens': len(known_tokens),
            'unknown_tokens': len(unknown_tokens),
            'unknown_token_list': unknown_tokens[:20],  # First 20 unknown tokens
            'coverage_percentage': coverage * 100
        }
    
    def filter_vocabulary_by_frequency(self, min_frequency: int) -> None:
        """
        Filter vocabulary to only include tokens with minimum frequency.
        
        Args:
            min_frequency: Minimum frequency threshold
        """
        logger.info(f"Filtering vocabulary with min frequency {min_frequency}")
        
        # Keep special tokens
        filtered_vocab = self.special_tokens.copy()
        
        # Add frequent tokens
        for token, count in self.token_counts.items():
            if count >= min_frequency and len(filtered_vocab) < self.max_vocab_size:
                if token not in filtered_vocab:
                    filtered_vocab[token] = len(filtered_vocab)
        
        # Update vocabularies
        self.token_to_id = filtered_vocab
        self.id_to_token = {v: k for k, v in filtered_vocab.items()}
        
        logger.info(f"Filtered vocabulary to {len(self.token_to_id)} tokens")
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'token_to_id': self.token_to_id,
            'special_tokens': self.special_tokens,
            'vocabulary_stats': self.vocabulary_stats,
            'token_counts': dict(self.token_counts.most_common(1000)),  # Save top 1000
            'max_vocab_size': self.max_vocab_size
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary from file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['token_to_id']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.special_tokens = vocab_data.get('special_tokens', self.special_tokens)
        self.vocabulary_stats = vocab_data.get('vocabulary_stats', {})
        self.max_vocab_size = vocab_data.get('max_vocab_size', self.max_vocab_size)
        
        # Restore token counts
        saved_counts = vocab_data.get('token_counts', {})
        self.token_counts = Counter(saved_counts)
        
        logger.info(f"Vocabulary loaded from {filepath}")
    
    def extend_vocabulary(self, new_tokens: List[str]) -> int:
        """
        Extend vocabulary with new tokens.
        
        Args:
            new_tokens: List of new token strings
            
        Returns:
            Number of tokens actually added
        """
        added_count = 0
        
        for token in new_tokens:
            if token not in self.token_to_id and len(self.token_to_id) < self.max_vocab_size:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token
                added_count += 1
        
        logger.info(f"Extended vocabulary with {added_count} new tokens")
        return added_count
    
    def get_token_type_distribution(self) -> Dict[str, int]:
        """Get distribution of token types in vocabulary."""
        return self.vocabulary_stats.get('token_type_distribution', {})


if __name__ == "__main__":
    # Example usage
    from .css_parser import CSSClassParser
    
    vocab_builder = VocabularyBuilder(max_vocab_size=1000)
    css_parser = CSSClassParser()
    
    # Sample structures
    sample_structures = [
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
    
    print("=== Building Vocabulary ===")
    vocab_builder.build_from_structures(sample_structures, css_parser)
    
    print(f"Vocabulary size: {vocab_builder.get_vocabulary_size()}")
    print(f"Statistics: {vocab_builder.get_statistics()}")
    
    # Test token encoding/decoding
    test_tokens = ["[TAG]div", "[CSS]container", "[CONTENT]text:Welcome"]
    encoded = vocab_builder.encode_tokens(test_tokens)
    decoded = vocab_builder.decode_tokens(encoded)
    
    print(f"\n=== Token Encoding Test ===")
    print(f"Original: {test_tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    
    # Coverage analysis
    coverage = vocab_builder.analyze_token_coverage(test_tokens)
    print(f"\n=== Coverage Analysis ===")
    for key, value in coverage.items():
        print(f"{key}: {value}")
    
    # Most frequent tokens
    print(f"\n=== Most Frequent Tokens ===")
    for token, count in vocab_builder.get_most_frequent_tokens(10):
        print(f"{token}: {count}") 