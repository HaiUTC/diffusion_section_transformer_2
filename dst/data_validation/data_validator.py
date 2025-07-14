"""
Data Validator

Main validation orchestrator for task 1.1 that combines HTML and XOJL processing.
Pure processing logic without any file I/O operations.
"""

import logging
from typing import Dict, List, Any, Tuple

from .data_sample import DataSample
from .html_processor import HTMLProcessor
from .xojl_processor import XOJLProcessor

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Main validator that orchestrates HTML and XOJL validation.
    
    Provides comprehensive validation of data samples by combining
    HTML structure processing and XOJL semantic validation.
    No file operations - only data processing.
    """
    
    def __init__(self, html_max_depth: int = 50, html_content_truncate_length: int = 100, 
                 html_max_vocab_size: int = 10000):
        """
        Initialize DataValidator with configurable HTML processing limits.
        
        Args:
            html_max_depth: Maximum recursion depth for HTML processing
            html_content_truncate_length: Maximum length for HTML text/src content
            html_max_vocab_size: Maximum vocabulary size for HTML processing
        """
        self.html_processor = HTMLProcessor(
            max_depth=html_max_depth,
            content_truncate_length=html_content_truncate_length,
            max_vocab_size=html_max_vocab_size
        )
        self.xojl_processor = XOJLProcessor()
        self.validation_statistics = {
            'total_processed': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'error_counts': {},
            'consistency_scores': []
        }
    
    def validate_sample(self, sample: DataSample) -> Dict[str, Any]:
        """
        Validate a complete data sample.
        
        Args:
            sample: DataSample instance to validate
            
        Returns:
            Comprehensive validation result
        """
        validation_result = {
            'sample_id': sample.sample_id,
            'is_valid': True,
            'html_validation': {},
            'xojl_validation': {},
            'consistency_validation': {},
            'overall_score': 0.0,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Validate HTML structure
            html_validation = self._validate_html_structure(sample.html_structure)
            validation_result['html_validation'] = html_validation
            
            # Validate XOJL layout
            xojl_validation = self._validate_xojl_layout(sample.section_layout)
            validation_result['xojl_validation'] = xojl_validation
            
            # Validate consistency between HTML and XOJL
            consistency_validation = self._validate_consistency(
                sample.html_structure, sample.section_layout
            )
            validation_result['consistency_validation'] = consistency_validation
            
            # Calculate overall validation score
            overall_score = self._calculate_overall_score(
                html_validation, xojl_validation, consistency_validation
            )
            validation_result['overall_score'] = overall_score
            
            # Determine if sample is valid
            validation_result['is_valid'] = self._determine_validity(
                html_validation, xojl_validation, consistency_validation
            )
            
            # Update statistics
            self._update_statistics(validation_result)
            
        except Exception as e:
            logger.error(f"Error validating sample {sample.sample_id}: {e}")
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def _validate_html_structure(self, html_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Validate HTML structure using HTMLProcessor."""
        try:
            # Parse and extract features
            parsed_structure = self.html_processor.parse_structure(html_structure)
            features = self.html_processor.extract_features(parsed_structure)
            hierarchy = self.html_processor.extract_hierarchy(parsed_structure)
            tokens = self.html_processor.build_token_sequence(parsed_structure)
            
            return {
                'is_valid': True,
                'features_count': len(features),
                'hierarchy_depth': max([elem['depth'] for elem in hierarchy]) if hierarchy else 0,
                'token_count': len(tokens),
                'vocabulary_stats': self.html_processor.get_vocabulary_statistics(),
                'errors': []
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'features_count': 0,
                'hierarchy_depth': 0,
                'token_count': 0,
                'vocabulary_stats': {},
                'errors': [str(e)]
            }
    
    def _validate_xojl_layout(self, section_layout: Dict[str, Any]) -> Dict[str, Any]:
        """Validate XOJL layout using XOJLProcessor."""
        try:
            # Parse and validate structure
            parsed_layout = self.xojl_processor.parse_layout(section_layout)
            validation_result = self.xojl_processor.validate_structure(parsed_layout)
            
            # Extract additional metrics
            used_elements = self.xojl_processor.extract_used_elements(parsed_layout)
            complexity_metrics = self.xojl_processor.analyze_layout_complexity(parsed_layout)
            layout_category = self.xojl_processor.get_layout_category(parsed_layout)
            
            return {
                'is_valid': validation_result['is_valid'],
                'used_elements': used_elements,
                'element_count': len(used_elements),
                'complexity_metrics': complexity_metrics,
                'layout_category': layout_category,
                'validation_details': validation_result,
                'errors': validation_result['errors'],
                'warnings': validation_result['warnings']
            }
            
        except Exception as e:
            return {
                'is_valid': False,
                'used_elements': [],
                'element_count': 0,
                'complexity_metrics': {},
                'layout_category': 'unknown',
                'validation_details': {},
                'errors': [str(e)],
                'warnings': []
            }
    
    def _validate_consistency(self, html_structure: Dict[str, Any], 
                            section_layout: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consistency between HTML and XOJL structures."""
        try:
            # Extract elements from both structures
            html_features = self.html_processor.extract_features(html_structure)
            html_elements = [feature['tag'] for feature in html_features]
            
            parsed_layout = self.xojl_processor.parse_layout(section_layout)
            
            # Validate semantic consistency
            consistency_result = self.xojl_processor.validate_semantic_consistency(
                parsed_layout, html_elements
            )
            
            return {
                'is_consistent': consistency_result['is_consistent'],
                'consistency_score': consistency_result['consistency_score'],
                'common_elements': consistency_result['common_elements'],
                'semantic_gaps': consistency_result['semantic_only_elements'],
                'html_gaps': consistency_result['html_only_elements'],
                'html_element_count': len(html_elements),
                'semantic_element_count': consistency_result['total_semantic_elements']
            }
            
        except Exception as e:
            return {
                'is_consistent': False,
                'consistency_score': 0.0,
                'common_elements': [],
                'semantic_gaps': [],
                'html_gaps': [],
                'html_element_count': 0,
                'semantic_element_count': 0,
                'error': str(e)
            }
    
    def _calculate_overall_score(self, html_validation: Dict[str, Any], 
                               xojl_validation: Dict[str, Any],
                               consistency_validation: Dict[str, Any]) -> float:
        """Calculate overall validation score."""
        # HTML validation score (40% weight)
        html_score = 1.0 if html_validation['is_valid'] else 0.0
        
        # XOJL validation score (40% weight)
        xojl_score = 1.0 if xojl_validation['is_valid'] else 0.0
        
        # Consistency score (20% weight)
        consistency_score = consistency_validation['consistency_score']
        
        return (html_score * 0.4) + (xojl_score * 0.4) + (consistency_score * 0.2)
    
    def _determine_validity(self, html_validation: Dict[str, Any], 
                          xojl_validation: Dict[str, Any],
                          consistency_validation: Dict[str, Any]) -> bool:
        """Determine if sample is valid based on all validations."""
        return (
            html_validation['is_valid'] and 
            xojl_validation['is_valid'] and 
            consistency_validation['consistency_score'] >= 0.5
        )
    
    def _update_statistics(self, validation_result: Dict[str, Any]):
        """Update validation statistics."""
        self.validation_statistics['total_processed'] += 1
        
        if validation_result['is_valid']:
            self.validation_statistics['valid_samples'] += 1
        else:
            self.validation_statistics['invalid_samples'] += 1
        
        # Track error types
        for error in validation_result.get('errors', []):
            error_type = error.split(':')[0] if ':' in error else 'unknown'
            self.validation_statistics['error_counts'][error_type] = (
                self.validation_statistics['error_counts'].get(error_type, 0) + 1
            )
        
        # Track consistency scores
        consistency_score = validation_result.get('consistency_validation', {}).get('consistency_score', 0)
        self.validation_statistics['consistency_scores'].append(consistency_score)
    
    def validate_batch(self, samples: List[DataSample]) -> Dict[str, Any]:
        """
        Validate a batch of data samples.
        
        Args:
            samples: List of DataSample instances to validate
            
        Returns:
            Batch validation result with statistics
        """
        batch_results = []
        
        for sample in samples:
            result = self.validate_sample(sample)
            batch_results.append(result)
        
        # Calculate batch statistics
        valid_count = sum(1 for result in batch_results if result['is_valid'])
        average_score = sum(result['overall_score'] for result in batch_results) / len(batch_results)
        
        return {
            'total_samples': len(samples),
            'valid_samples': valid_count,
            'invalid_samples': len(samples) - valid_count,
            'validation_rate': valid_count / len(samples),
            'average_score': average_score,
            'results': batch_results
        }
    
    def get_validation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics."""
        stats = self.validation_statistics.copy()
        
        # Calculate additional metrics
        if stats['total_processed'] > 0:
            stats['validation_rate'] = stats['valid_samples'] / stats['total_processed']
            
        if stats['consistency_scores']:
            stats['average_consistency_score'] = (
                sum(stats['consistency_scores']) / len(stats['consistency_scores'])
            )
            stats['min_consistency_score'] = min(stats['consistency_scores'])
            stats['max_consistency_score'] = max(stats['consistency_scores'])
        
        return stats
    
    def reset_statistics(self):
        """Reset validation statistics."""
        self.validation_statistics = {
            'total_processed': 0,
            'valid_samples': 0,
            'invalid_samples': 0,
            'error_counts': {},
            'consistency_scores': []
        }
        
        # Reset processor vocabularies
        self.html_processor.reset_vocabularies()
    
    def get_processor_statistics(self) -> Dict[str, Any]:
        """Get statistics from individual processors."""
        return {
            'html_processor': self.html_processor.get_vocabulary_statistics(),
            'xojl_processor': {
                'valid_elements_count': len(self.xojl_processor.valid_elements),
                'semantic_categories': list(self.xojl_processor.semantic_categories.keys())
            }
        } 