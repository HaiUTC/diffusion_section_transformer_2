"""
HTML Structure Processor

Handles HTML structure parsing, feature extraction, and vocabulary building.
Pure processing logic without any file I/O operations.
"""

import json
import re
import logging
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)


class HTMLProcessor:
    """
    Processes HTML structures for layout analysis.
    
    Extracts structural features, builds vocabularies, and prepares data
    for downstream processing. No file operations.
    """
    
    def __init__(self, max_depth: int = 50, content_truncate_length: int = 100, 
                 max_vocab_size: int = 10000):
        """
        Initialize HTML processor with configurable limits.
        
        Args:
            max_depth: Maximum recursion depth to prevent infinite loops
            content_truncate_length: Maximum length for text/src content
            max_vocab_size: Maximum vocabulary size to prevent memory issues
        """
        self.max_depth = max_depth
        self.content_truncate_length = content_truncate_length
        self.max_vocab_size = max_vocab_size
        
        self.tag_vocabulary = set()
        self.class_vocabulary = set()
        self.special_tokens = ['<START>', '<END>', '<PAD>', '<UNK>']
        
        # Track processing state
        self._processing_paths = set()  # For circular reference detection
    
    def parse_structure(self, html_data: Any) -> Dict[str, Any]:
        """
        Parse HTML structure from various input formats.
        
        Args:
            html_data: HTML structure as JSON string or dictionary
            
        Returns:
            Parsed HTML structure as nested dictionary
            
        Raises:
            ValueError: If structure is invalid or malformed
            TypeError: If input type is not supported
        """
        if html_data is None:
            raise ValueError("HTML data cannot be None")
            
        if isinstance(html_data, str):
            if not html_data.strip():
                raise ValueError("HTML data string cannot be empty")
            try:
                structure = json.loads(html_data)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON in HTML structure: {e}")
                raise ValueError(f"Failed to parse HTML structure: {e}")
        elif isinstance(html_data, dict):
            structure = html_data
        else:
            raise TypeError(f"Unsupported HTML data type: {type(html_data)}. Expected str or dict.")
        
        # Validate structure format
        if not isinstance(structure, dict):
            raise ValueError("HTML structure must be a dictionary")
            
        # Check for basic structure validity
        if not structure:
            logger.warning("Empty HTML structure provided")
            
        return structure
    
    def extract_features(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract features from HTML structure.
        
        Args:
            structure: HTML structure as nested dictionary
            
        Returns:
            List of feature dictionaries for each HTML element
            
        Raises:
            ValueError: If structure is invalid or too deep
        """
        if not isinstance(structure, dict):
            raise ValueError("Structure must be a dictionary")
            
        features = []
        self._processing_paths.clear()  # Reset for each extraction
        
        try:
            self._extract_recursive(structure, features, depth=0, current_path="")
        except RecursionError:
            raise ValueError(f"Structure exceeds maximum depth of {self.max_depth}")
        finally:
            self._processing_paths.clear()
            
        return features
    
    def _extract_recursive(self, structure: Dict[str, Any], features: List[Dict[str, Any]], 
                          depth: int, current_path: str = ""):
        """
        Recursively extract features from nested structure.
        
        Args:
            structure: Current level of HTML structure
            features: List to accumulate features
            depth: Current recursion depth
            current_path: Current path for circular reference detection
        """
        # Prevent infinite recursion
        if depth > self.max_depth:
            logger.warning(f"Maximum depth {self.max_depth} exceeded, stopping recursion")
            return
            
        # Detect circular references
        if current_path in self._processing_paths:
            logger.warning(f"Circular reference detected at path: {current_path}")
            return
            
        self._processing_paths.add(current_path)
        
        try:
            for key, value in structure.items():
                if self._is_content_key(key):
                    continue
                    
                # Parse element information
                try:
                    tag, classes = self._parse_element_key(key)
                except Exception as e:
                    logger.warning(f"Failed to parse element key '{key}': {e}")
                    continue
                
                # Update vocabularies with size limits
                self._safe_add_to_vocabulary(self.tag_vocabulary, tag)
                for cls in classes:
                    self._safe_add_to_vocabulary(self.class_vocabulary, cls)
                
                # Build new path for nested processing
                new_path = f"{current_path}/{key}" if current_path else key
                
                # Extract element features
                try:
                    feature = {
                        'tag': tag,
                        'classes': classes,
                        'depth': depth,
                        'path': new_path,
                        'has_text': self._has_text_content(value),
                        'has_src': self._has_src_content(value),
                        'text_content': self._extract_text_content(value),
                        'src_content': self._extract_src_content(value),
                        'child_count': self._count_children(value)
                    }
                    features.append(feature)
                except Exception as e:
                    logger.warning(f"Failed to extract features for element '{key}': {e}")
                    continue
                
                # Process nested elements
                if isinstance(value, dict) and depth < self.max_depth:
                    self._extract_recursive(value, features, depth + 1, new_path)
                    
        finally:
            self._processing_paths.discard(current_path)
    
    def _safe_add_to_vocabulary(self, vocabulary: set, item: str):
        """Safely add item to vocabulary with size limit."""
        if len(vocabulary) < self.max_vocab_size:
            vocabulary.add(item)
        elif item not in vocabulary:
            logger.warning(f"Vocabulary size limit ({self.max_vocab_size}) reached, skipping item: {item}")
    
    def _parse_element_key(self, key: str) -> Tuple[str, List[str]]:
        """
        Parse tag and classes from element key (e.g., 'div@class1@class2').
        
        Args:
            key: Element key string
            
        Returns:
            Tuple of (tag, classes_list)
            
        Raises:
            ValueError: If key format is invalid
        """
        if not isinstance(key, str):
            raise ValueError(f"Element key must be string, got {type(key)}")
            
        if not key.strip():
            raise ValueError("Element key cannot be empty")
            
        # Validate key format (basic HTML tag name validation)
        if '@' in key:
            parts = key.split('@')
            tag = parts[0].strip()
            classes = [cls.strip() for cls in parts[1:] if cls.strip()]
        else:
            tag = key.strip()
            classes = []
            
        # Basic tag validation
        if not tag:
            raise ValueError("Tag name cannot be empty")
            
        # HTML tag name validation (simplified)
        if not re.match(r'^[a-zA-Z][a-zA-Z0-9-]*$', tag):
            logger.warning(f"Potentially invalid HTML tag name: {tag}")
            
        return tag, classes
    
    def _is_content_key(self, key: str) -> bool:
        """Check if key represents content rather than an element."""
        if not isinstance(key, str):
            return False
        return key in ['text', 'src', 'svg']  # Added 'svg' as common content type
    
    def _has_text_content(self, value: Any) -> bool:
        """Check if element has text content."""
        return isinstance(value, dict) and 'text' in value and value['text'] is not None
    
    def _has_src_content(self, value: Any) -> bool:
        """Check if element has src content."""
        return isinstance(value, dict) and 'src' in value and value['src'] is not None
    
    def _extract_text_content(self, value: Any) -> str:
        """Extract text content from element."""
        if self._has_text_content(value):
            try:
                text = str(value['text'])
                return text[:self.content_truncate_length] if text else ""
            except Exception as e:
                logger.warning(f"Failed to extract text content: {e}")
                return ""
        return ""
    
    def _extract_src_content(self, value: Any) -> str:
        """Extract src content from element."""
        if self._has_src_content(value):
            try:
                src = str(value['src'])
                return src[:self.content_truncate_length] if src else ""
            except Exception as e:
                logger.warning(f"Failed to extract src content: {e}")
                return ""
        return ""
    
    def _count_children(self, value: Any) -> int:
        """
        Count number of child elements.
        
        Args:
            value: Element value to check for children
            
        Returns:
            Number of child elements
        """
        if not isinstance(value, dict):
            return 0
            
        try:
            return len([k for k in value.keys() if not self._is_content_key(k)])
        except Exception as e:
            logger.warning(f"Failed to count children: {e}")
            return 0
    
    def build_token_sequence(self, structure: Dict[str, Any]) -> List[str]:
        """
        Convert HTML structure to token sequence.
        
        Args:
            structure: Parsed HTML structure
            
        Returns:
            List of string tokens representing the structure
            
        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(structure, dict):
            raise ValueError("Structure must be a dictionary")
            
        try:
            features = self.extract_features(structure)
        except Exception as e:
            logger.error(f"Failed to extract features for token sequence: {e}")
            raise ValueError(f"Cannot build token sequence: {e}")
            
        tokens = ['<START>']
        
        for feature in features:
            try:
                token_parts = [feature['tag']]
                
                # Add class information
                if feature.get('classes'):
                    token_parts.extend([f"@{cls}" for cls in feature['classes']])
                
                # Add content indicators
                if feature.get('has_text', False):
                    token_parts.append("text")
                if feature.get('has_src', False):
                    token_parts.append("src")
                
                # Add depth indicator
                depth = feature.get('depth', 0)
                token_parts.append(f"d{depth}")
                
                tokens.append('|'.join(token_parts))
                
            except Exception as e:
                logger.warning(f"Failed to build token for feature: {e}")
                tokens.append('<UNK>')  # Add unknown token for failed features
        
        tokens.append('<END>')
        return tokens
    
    def get_vocabulary_statistics(self) -> Dict[str, Any]:
        """
        Get vocabulary statistics.
        
        Returns:
            Dictionary containing vocabulary information
        """
        return {
            'tag_vocabulary': {
                'size': len(self.tag_vocabulary),
                'tags': sorted(list(self.tag_vocabulary))[:100],  # Limit output size
                'total_tags': len(self.tag_vocabulary)
            },
            'class_vocabulary': {
                'size': len(self.class_vocabulary),
                'classes': sorted(list(self.class_vocabulary))[:100],  # Limit output size
                'total_classes': len(self.class_vocabulary)
            },
            'special_tokens': self.special_tokens,
            'total_vocabulary_size': len(self.tag_vocabulary) + len(self.class_vocabulary) + len(self.special_tokens),
            'configuration': {
                'max_depth': self.max_depth,
                'content_truncate_length': self.content_truncate_length,
                'max_vocab_size': self.max_vocab_size
            },
            'vocabulary_limits_reached': {
                'tag_vocabulary': len(self.tag_vocabulary) >= self.max_vocab_size,
                'class_vocabulary': len(self.class_vocabulary) >= self.max_vocab_size
            }
        }
    
    def reset_vocabularies(self):
        """Reset vocabularies for processing new dataset."""
        self.tag_vocabulary.clear()
        self.class_vocabulary.clear()
        self._processing_paths.clear()
        logger.info("Vocabularies reset")
    
    def extract_hierarchy(self, structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract hierarchical relationships from HTML structure.
        
        Args:
            structure: HTML structure as nested dictionary
            
        Returns:
            List of elements with hierarchy information
            
        Raises:
            ValueError: If structure is invalid
        """
        if not isinstance(structure, dict):
            raise ValueError("Structure must be a dictionary")
            
        hierarchy = []
        self._processing_paths.clear()
        
        try:
            self._build_hierarchy(structure, hierarchy, parent_path="", depth=0)
        except RecursionError:
            raise ValueError(f"Structure exceeds maximum depth of {self.max_depth}")
        finally:
            self._processing_paths.clear()
            
        return hierarchy
    
    def _build_hierarchy(self, structure: Dict[str, Any], hierarchy: List[Dict[str, Any]], 
                        parent_path: str, depth: int):
        """
        Build hierarchy information recursively.
        
        Args:
            structure: Current level of structure
            hierarchy: List to accumulate hierarchy info
            parent_path: Path of parent element
            depth: Current depth level
        """
        # Prevent infinite recursion
        if depth > self.max_depth:
            logger.warning(f"Maximum depth {self.max_depth} exceeded in hierarchy building")
            return
            
        # Detect circular references
        if parent_path in self._processing_paths:
            logger.warning(f"Circular reference detected in hierarchy at path: {parent_path}")
            return
            
        self._processing_paths.add(parent_path)
        
        try:
            for key, value in structure.items():
                if self._is_content_key(key):
                    continue
                    
                current_path = f"{parent_path}/{key}" if parent_path else key
                
                try:
                    tag, classes = self._parse_element_key(key)
                except Exception as e:
                    logger.warning(f"Failed to parse element key in hierarchy '{key}': {e}")
                    continue
                
                element_info = {
                    'path': current_path,
                    'tag': tag,
                    'classes': classes,
                    'depth': depth,
                    'parent_path': parent_path or None,
                    'has_children': isinstance(value, dict) and any(
                        not self._is_content_key(k) for k in value.keys()
                    ),
                    'child_count': self._count_children(value)
                }
                hierarchy.append(element_info)
                
                if isinstance(value, dict) and depth < self.max_depth:
                    self._build_hierarchy(value, hierarchy, current_path, depth + 1)
                    
        finally:
            self._processing_paths.discard(parent_path) 