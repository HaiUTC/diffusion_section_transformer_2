"""
Subword Tokenizer

Implements Byte Pair Encoding (BPE) for managing CSS class vocabulary.
Converts complex CSS strings into manageable subword tokens.
"""

import re
import logging
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import Counter, defaultdict
import json

logger = logging.getLogger(__name__)


class SubwordTokenizer:
    """
    Byte Pair Encoding tokenizer for CSS class strings.
    
    Manages vocabulary explosion by learning subword units from training data.
    Handles complex CSS patterns while maintaining semantic compositionality.
    """
    
    def __init__(self, vocab_size: int = 50000, min_frequency: int = 2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        
        # Special tokens
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1, 
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4,
            '[BOS]': 5,  # Beginning of sequence
            '[EOS]': 6   # End of sequence
        }
        
        # BPE vocabulary and merges
        self.vocab = {}
        self.bpe_merges = []
        self.token_to_id = {}
        self.id_to_token = {}
        
        # Statistics
        self.training_stats = {}
        
        # CSS-specific device contexts and pseudo contexts for tokenization
        self.device_contexts = {'t', 'm', 'md', 'lg', 'xl', 'sm'}  # Common responsive breakpoints
        self.pseudo_contexts = {'h', 'f', 'a', 'v', 'd'}  # hover, focus, active, visited, disabled
        
        # Pre-tokenization pattern for CSS classes
        # Updated to properly split device:pseudo:attribute_value+unit format
        self._compile_css_patterns()
        
        self._initialize_vocab()
    
    def _compile_css_patterns(self):
        """Compile CSS-specific tokenization patterns."""
        # Build pattern to match CSS class format: [device:]?[pseudo:]?attribute[_value][unit]?
        device_pattern = f"(?:{'|'.join(self.device_contexts)})"
        pseudo_pattern = f"(?:{'|'.join(self.pseudo_contexts)})"
        
        # Complete CSS class pattern - improved to better separate value and unit
        # Separate numeric value from alphabetic unit properly
        self.css_class_pattern = re.compile(
            rf'^(?:({device_pattern}):)?(?:({pseudo_pattern}):)?([a-zA-Z]+)(?:(_)([0-9]+(?:\.[0-9]+)?)([a-zA-Z]*)?)?$'
        )
        
        # Fallback pattern for simple tokenization
        self.pre_tokenize_pattern = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*|[0-9]+\.?[0-9]*|[:\-@_])')
    
    def _initialize_vocab(self):
        """Initialize vocabulary with special tokens."""
        self.token_to_id = self.special_tokens.copy()
        self.id_to_token = {v: k for k, v in self.special_tokens.items()}
        
        # Add character-level tokens for robustness
        for i in range(256):
            char = chr(i)
            if char not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[char] = token_id
                self.id_to_token[token_id] = char
    
    def pre_tokenize(self, text: str) -> List[str]:
        """
        Pre-tokenize CSS class into semantic components.
        
        Splits format: [device:]?[pseudo:]?attribute[_value1][unit1][_value2][unit2]...
        Examples:
        - fs_12x => ["fs", "_12", "x"]
        - c_#000 => ["c", "_#000"] 
        - m:fw_700 => ["m", ":", "fw", "_700"]
        - h:bc_red => ["h", ":", "bc", "_red"]
        - t:h:w_20p => ["t", ":", "h", ":", "w", "_20", "p"]
        - p_12x_20x => ["p", "_12", "x", "_20", "x"]
        - b_1x_solid_#000 => ["b", "_1", "x", "_solid", "_#000"]
        
        Args:
            text: Input CSS class string
            
        Returns:
            List of semantic tokens
        """
        if not text:
            return []
        
        tokens = []
        remaining = text
        
        # Extract device contexts (device:)
        while True:
            device_match = None
            for device in self.device_contexts:
                if remaining.startswith(f"{device}:"):
                    device_match = device
                    break
            
            if device_match:
                tokens.extend([device_match, ':'])
                remaining = remaining[len(device_match) + 1:]
            else:
                break
        
        # Extract pseudo contexts (pseudo:)
        while True:
            pseudo_match = None
            for pseudo in self.pseudo_contexts:
                if remaining.startswith(f"{pseudo}:"):
                    pseudo_match = pseudo
                    break
            
            if pseudo_match:
                tokens.extend([pseudo_match, ':'])
                remaining = remaining[len(pseudo_match) + 1:]
            else:
                break
        
        # Split remaining part on underscores
        if '_' in remaining:
            parts = remaining.split('_')
            # First part is the attribute
            attribute = parts[0]
            tokens.append(attribute)
            
            # Process value parts (everything after first _)
            for i, value_part in enumerate(parts[1:], 1):
                if not value_part:
                    continue
                    
                # Add underscore prefix to value
                tokens.append(f"_{value_part}")
                
                # Try to separate numeric value from unit
                # Pattern: extract trailing alphabetic characters as unit
                unit_match = re.search(r'([0-9\.]+)([a-zA-Z]+)$', value_part)
                if unit_match:
                    numeric_value, unit = unit_match.groups()
                    # Replace the combined token with separated tokens
                    tokens[-1] = f"_{numeric_value}"
                    tokens.append(unit)
        else:
            # No underscore, treat whole remaining as attribute
            tokens.append(remaining)
        
        return [token for token in tokens if token]
    
    def get_word_frequency(self, texts: List[str]) -> Dict[str, int]:
        """
        Get frequency count of pre-tokenized words.
        
        Args:
            texts: List of CSS class strings
            
        Returns:
            Dictionary mapping words to frequencies
        """
        word_freq = Counter()
        
        for text in texts:
            pre_tokens = self.pre_tokenize(text)
            for token in pre_tokens:
                word_freq[token] += 1
        
        return dict(word_freq)
    
    def get_pair_frequency(self, word_freq: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """
        Get frequency of character pairs in vocabulary.
        
        Args:
            word_freq: Word frequency dictionary
            
        Returns:
            Dictionary mapping character pairs to frequencies
        """
        pair_freq = defaultdict(int)
        
        for word, freq in word_freq.items():
            # Convert word to character sequence
            chars = list(word)
            
            # Count adjacent pairs
            for i in range(len(chars) - 1):
                pair = (chars[i], chars[i + 1])
                pair_freq[pair] += freq
        
        return dict(pair_freq)
    
    def merge_pair(self, word_freq: Dict[str, int], pair: Tuple[str, str]) -> Dict[str, int]:
        """
        Merge a character pair across all words.
        
        Args:
            word_freq: Current word frequency dictionary
            pair: Character pair to merge
            
        Returns:
            Updated word frequency dictionary
        """
        new_word_freq = {}
        bigram = ''.join(pair)
        
        for word, freq in word_freq.items():
            # Replace pair with merged token
            new_word = word.replace(''.join(pair), bigram)
            new_word_freq[new_word] = freq
        
        return new_word_freq
    
    def train(self, texts: List[str]) -> None:
        """
        Train BPE tokenizer on CSS class strings.
        
        Args:
            texts: List of CSS class strings for training
        """
        logger.info(f"Training BPE tokenizer on {len(texts)} examples")
        
        # Pre-tokenize all texts
        all_tokens = []
        for text in texts:
            all_tokens.extend(self.pre_tokenize(text))
        
        # Get initial word frequencies
        word_freq = Counter(all_tokens)
        
        # Filter by minimum frequency
        word_freq = {word: freq for word, freq in word_freq.items() 
                    if freq >= self.min_frequency}
        
        logger.info(f"Initial vocabulary size: {len(word_freq)}")
        
        # Learn BPE merges
        num_merges = self.vocab_size - len(self.token_to_id)
        
        for i in range(num_merges):
            # Get pair frequencies
            pair_freq = self.get_pair_frequency(word_freq)
            
            if not pair_freq:
                break
            
            # Find most frequent pair
            best_pair = max(pair_freq.items(), key=lambda x: x[1])
            pair, freq = best_pair
            
            if freq < self.min_frequency:
                break
            
            # Merge the pair
            word_freq = self.merge_pair(word_freq, pair)
            
            # Add to BPE merges
            self.bpe_merges.append(pair)
            
            # Add merged token to vocabulary
            merged_token = ''.join(pair)
            if merged_token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[merged_token] = token_id
                self.id_to_token[token_id] = merged_token
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Completed {i + 1} merges, current vocab size: {len(self.token_to_id)}")
        
        # Build final vocabulary
        self.vocab = word_freq
        
        # Store training statistics
        self.training_stats = {
            'num_training_examples': len(texts),
            'num_merges_learned': len(self.bpe_merges),
            'final_vocab_size': len(self.token_to_id),
            'unique_tokens_in_training': len(set(all_tokens)),
            'total_tokens_in_training': len(all_tokens)
        }
        
        logger.info(f"Training complete. Final vocabulary size: {len(self.token_to_id)}")
    
    def apply_bpe(self, word: str) -> List[str]:
        """
        Apply learned BPE merges to a word.
        
        Args:
            word: Input word to tokenize
            
        Returns:
            List of BPE tokens
        """
        if not word:
            return []
        
        # Start with character sequence
        tokens = list(word)
        
        # Apply BPE merges in order
        for pair in self.bpe_merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    # Merge the pair
                    merged = ''.join(pair)
                    tokens = tokens[:i] + [merged] + tokens[i + 2:]
                else:
                    i += 1
        
        return tokens
    
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input CSS class string
            add_special_tokens: Whether to add [BOS] and [EOS] tokens
            
        Returns:
            List of token IDs
        """
        if not text:
            return []
        
        # Pre-tokenize
        pre_tokens = self.pre_tokenize(text)
        
        # Apply BPE to each pre-token
        bpe_tokens = []
        for pre_token in pre_tokens:
            bpe_tokens.extend(self.apply_bpe(pre_token))
            
        # Convert to IDs
        token_ids = []
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['[BOS]'])
        
        for token in bpe_tokens:
            if token in self.token_to_id:
                token_ids.append(self.token_to_id[token])
            else:
                # Fall back to character-level for unknown tokens
                for char in token:
                    if char in self.token_to_id:
                        token_ids.append(self.token_to_id[char])
                    else:
                        token_ids.append(self.token_to_id['[UNK]'])
        
        if add_special_tokens:
            token_ids.append(self.token_to_id['[EOS]'])
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                
                # Skip special tokens if requested
                if skip_special_tokens and token in self.special_tokens:
                    continue
                
                tokens.append(token)
        
        return ''.join(tokens)
    
    def encode_batch(self, texts: List[str], add_special_tokens: bool = True) -> List[List[int]]:
        """
        Encode a batch of texts.
        
        Args:
            texts: List of CSS class strings
            add_special_tokens: Whether to add special tokens
            
        Returns:
            List of token ID sequences
        """
        return [self.encode(text, add_special_tokens) for text in texts]
    
    def get_vocab_size(self) -> int:
        """Get current vocabulary size."""
        return len(self.token_to_id)
    
    def get_vocab(self) -> Dict[str, int]:
        """Get vocabulary mapping."""
        return self.token_to_id.copy()
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return self.training_stats.copy()
    
    def save_vocabulary(self, filepath: str) -> None:
        """
        Save vocabulary and BPE merges to file.
        
        Args:
            filepath: Path to save vocabulary
        """
        vocab_data = {
            'vocab': self.token_to_id,
            'bpe_merges': self.bpe_merges,
            'special_tokens': self.special_tokens,
            'training_stats': self.training_stats,
            'vocab_size': self.vocab_size,
            'min_frequency': self.min_frequency
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath: str) -> None:
        """
        Load vocabulary and BPE merges from file.
        
        Args:
            filepath: Path to load vocabulary from
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.token_to_id = vocab_data['vocab']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.bpe_merges = [tuple(pair) for pair in vocab_data['bpe_merges']]
        self.special_tokens = vocab_data['special_tokens']
        self.training_stats = vocab_data.get('training_stats', {})
        self.vocab_size = vocab_data.get('vocab_size', self.vocab_size)
        self.min_frequency = vocab_data.get('min_frequency', self.min_frequency)
        
        logger.info(f"Vocabulary loaded from {filepath}")
    
    def analyze_tokenization(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze tokenization quality on given texts.
        
        Args:
            texts: List of CSS class strings to analyze
            
        Returns:
            Analysis statistics
        """
        total_chars = 0
        total_tokens = 0
        unknown_tokens = 0
        compression_ratios = []
        
        for text in texts:
            total_chars += len(text)
            
            token_ids = self.encode(text, add_special_tokens=False)
            total_tokens += len(token_ids)
            
            # Count unknown tokens
            unknown_tokens += token_ids.count(self.token_to_id['[UNK]'])
            
            # Calculate compression ratio
            if len(text) > 0:
                compression_ratios.append(len(token_ids) / len(text))
        
        avg_compression = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0
        
        return {
            'total_characters': total_chars,
            'total_tokens': total_tokens,
            'unknown_tokens': unknown_tokens,
            'unknown_token_rate': unknown_tokens / max(total_tokens, 1),
            'average_compression_ratio': avg_compression,
            'characters_per_token': total_chars / max(total_tokens, 1),
            'vocabulary_coverage': 1 - (unknown_tokens / max(total_tokens, 1))
        }


if __name__ == "__main__":
    # Example usage
    tokenizer = SubwordTokenizer(vocab_size=1000)
    
    # Sample CSS class strings
    sample_texts = [
        "pt_7r", "t:fs_18x", "h:c_blue", "m:h:pt_auto", "mr_auto",
        "div@mr_auto@ml_auto", "h1@t:fs_18x@h:c_blue", "p@pt_7r",
        "bg_primary", "w_100p", "d_flex", "jc_center", "ai_center"
    ]
    
    print("=== Training BPE Tokenizer ===")
    tokenizer.train(sample_texts)
    
    print(f"\nFinal vocabulary size: {tokenizer.get_vocab_size()}")
    print(f"Training stats: {tokenizer.get_training_stats()}")
    
    print(f"\n=== Tokenization Examples ===")
    for text in sample_texts[:5]:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"'{text}' -> {tokens} -> '{decoded}'")
    
    print(f"\n=== Analysis ===")
    analysis = tokenizer.analyze_tokenization(sample_texts)
    for key, value in analysis.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}") 