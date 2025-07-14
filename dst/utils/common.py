"""
Common Utilities

Shared utility functions for data validation and image processing tasks.
Pure processing logic without any file I/O operations.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingStats:
    """Statistics for processing operations."""
    total_samples: int
    successful_samples: int
    failed_samples: int
    processing_time: float
    average_time_per_sample: float
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_samples / self.total_samples if self.total_samples > 0 else 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        return self.failed_samples / self.total_samples if self.total_samples > 0 else 0.0


class MemoryTracker:
    """Track memory usage during processing."""
    
    def __init__(self):
        self.peak_memory = 0
        self.current_memory = 0
        
    def update(self):
        """Update memory tracking."""
        if torch.cuda.is_available():
            self.current_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return {
            'current_memory_mb': self.current_memory,
            'peak_memory_mb': self.peak_memory
        }
    
    def reset(self):
        """Reset memory tracking."""
        self.peak_memory = 0
        self.current_memory = 0


class BatchProcessor:
    """Generic batch processor for handling data in chunks."""
    
    def __init__(self, batch_size: int = 32, max_retries: int = 3):
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.memory_tracker = MemoryTracker()
        
    def process_in_batches(self, 
                          data_list: List[Any], 
                          process_function: callable,
                          progress_callback: Optional[callable] = None) -> Tuple[List[Any], ProcessingStats]:
        """
        Process data in batches with error handling and statistics.
        
        Args:
            data_list: List of data items to process
            process_function: Function to apply to each batch
            progress_callback: Optional callback for progress updates
            
        Returns:
            Tuple of (results_list, processing_stats)
        """
        total_samples = len(data_list)
        results = []
        successful_samples = 0
        failed_samples = 0
        start_time = time.time()
        
        # Process in batches
        for i in range(0, total_samples, self.batch_size):
            batch_start = i
            batch_end = min(i + self.batch_size, total_samples)
            batch = data_list[batch_start:batch_end]
            
            # Process batch with retries
            batch_results, batch_success, batch_failures = self._process_batch_with_retry(
                batch, process_function
            )
            
            results.extend(batch_results)
            successful_samples += batch_success
            failed_samples += batch_failures
            
            # Update memory tracking
            self.memory_tracker.update()
            
            # Progress callback
            if progress_callback:
                progress = (batch_end / total_samples) * 100
                progress_callback(progress, batch_end, total_samples)
        
        # Calculate final statistics
        processing_time = time.time() - start_time
        avg_time_per_sample = processing_time / total_samples if total_samples > 0 else 0
        
        stats = ProcessingStats(
            total_samples=total_samples,
            successful_samples=successful_samples,
            failed_samples=failed_samples,
            processing_time=processing_time,
            average_time_per_sample=avg_time_per_sample
        )
        
        return results, stats
    
    def _process_batch_with_retry(self, 
                                 batch: List[Any], 
                                 process_function: callable) -> Tuple[List[Any], int, int]:
        """Process a single batch with retry logic."""
        for attempt in range(self.max_retries):
            try:
                batch_results = process_function(batch)
                return batch_results, len(batch), 0
            except Exception as e:
                logger.warning(f"Batch processing attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    # Return empty results for failed batch
                    return [None] * len(batch), 0, len(batch)
        
        return [], 0, len(batch)


def tensor_memory_size(tensor: torch.Tensor) -> Dict[str, float]:
    """
    Calculate memory size of a tensor.
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        Dictionary with memory size in different units
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor")
    
    bytes_size = tensor.numel() * tensor.element_size()
    
    return {
        'bytes': bytes_size,
        'kb': bytes_size / 1024,
        'mb': bytes_size / (1024 ** 2),
        'gb': bytes_size / (1024 ** 3)
    }


def validate_tensor_shape(tensor: torch.Tensor, 
                         expected_shape: Tuple[int, ...],
                         allow_batch: bool = True) -> bool:
    """
    Validate tensor shape with optional batch dimension.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (without batch dimension)
        allow_batch: Whether to allow an additional batch dimension
        
    Returns:
        True if shape is valid
    """
    if not isinstance(tensor, torch.Tensor):
        return False
    
    actual_shape = tensor.shape
    
    # Check exact match
    if actual_shape == expected_shape:
        return True
    
    # Check with batch dimension
    if allow_batch and len(actual_shape) == len(expected_shape) + 1:
        return actual_shape[1:] == expected_shape
    
    return False


def safe_division(numerator: Union[int, float], 
                 denominator: Union[int, float], 
                 default: float = 0.0) -> float:
    """
    Perform safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Default value if denominator is zero
        
    Returns:
        Division result or default value
    """
    if denominator == 0:
        return default
    return numerator / denominator


def calculate_percentage(part: Union[int, float], 
                        total: Union[int, float], 
                        decimals: int = 2) -> float:
    """
    Calculate percentage with safe handling.
    
    Args:
        part: Part value
        total: Total value
        decimals: Number of decimal places
        
    Returns:
        Percentage value
    """
    if total == 0:
        return 0.0
    return round((part / total) * 100, decimals)


def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple dictionaries safely.
    
    Args:
        *dicts: Variable number of dictionaries to merge
        
    Returns:
        Merged dictionary
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result


def format_time_duration(seconds: float) -> str:
    """
    Format time duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def format_memory_size(bytes_size: int) -> str:
    """
    Format memory size in human-readable format.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    if bytes_size < 1024:
        return f"{bytes_size}B"
    elif bytes_size < 1024**2:
        return f"{bytes_size/1024:.2f}KB"
    elif bytes_size < 1024**3:
        return f"{bytes_size/(1024**2):.2f}MB"
    else:
        return f"{bytes_size/(1024**3):.2f}GB"


class ConfigValidator:
    """Validate configuration dictionaries."""
    
    @staticmethod
    def validate_required_keys(config: Dict[str, Any], 
                             required_keys: List[str]) -> List[str]:
        """
        Validate that all required keys are present.
        
        Args:
            config: Configuration dictionary
            required_keys: List of required keys
            
        Returns:
            List of missing keys
        """
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
        return missing_keys
    
    @staticmethod
    def validate_types(config: Dict[str, Any], 
                      type_specs: Dict[str, type]) -> List[str]:
        """
        Validate types of configuration values.
        
        Args:
            config: Configuration dictionary
            type_specs: Dictionary mapping keys to expected types
            
        Returns:
            List of type validation errors
        """
        errors = []
        for key, expected_type in type_specs.items():
            if key in config:
                if not isinstance(config[key], expected_type):
                    errors.append(
                        f"Key '{key}' expected {expected_type.__name__}, "
                        f"got {type(config[key]).__name__}"
                    )
        return errors
    
    @staticmethod
    def validate_ranges(config: Dict[str, Any], 
                       range_specs: Dict[str, Tuple[float, float]]) -> List[str]:
        """
        Validate that numeric values are within specified ranges.
        
        Args:
            config: Configuration dictionary
            range_specs: Dictionary mapping keys to (min, max) tuples
            
        Returns:
            List of range validation errors
        """
        errors = []
        for key, (min_val, max_val) in range_specs.items():
            if key in config:
                value = config[key]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        errors.append(
                            f"Key '{key}' value {value} not in range [{min_val}, {max_val}]"
                        )
        return errors


def create_summary_report(title: str, 
                         data: Dict[str, Any], 
                         include_timestamp: bool = True) -> str:
    """
    Create a formatted summary report.
    
    Args:
        title: Report title
        data: Data dictionary to include in report
        include_timestamp: Whether to include timestamp
        
    Returns:
        Formatted report string
    """
    lines = [f"=== {title} ==="]
    
    if include_timestamp:
        lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
    
    def format_value(value):
        if isinstance(value, float):
            return f"{value:.4f}"
        elif isinstance(value, dict):
            return f"({len(value)} items)"
        elif isinstance(value, list):
            return f"[{len(value)} items]"
        else:
            return str(value)
    
    for key, value in data.items():
        formatted_value = format_value(value)
        lines.append(f"{key}: {formatted_value}")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    
    # Test tensor memory calculation
    test_tensor = torch.rand(4, 3, 720, 1280)
    memory_info = tensor_memory_size(test_tensor)
    print(f"Tensor memory: {memory_info}")
    
    # Test shape validation
    is_valid = validate_tensor_shape(test_tensor, (3, 720, 1280), allow_batch=True)
    print(f"Shape validation: {is_valid}")
    
    # Test batch processor
    processor = BatchProcessor(batch_size=2)
    test_data = [1, 2, 3, 4, 5]
    
    def dummy_process(batch):
        return [x * 2 for x in batch]
    
    results, stats = processor.process_in_batches(test_data, dummy_process)
    print(f"Processing results: {results}")
    print(f"Processing stats: {stats}")
    
    # Test summary report
    report_data = {
        'total_samples': 100,
        'success_rate': 0.95,
        'average_time': 1.234,
        'memory_usage': {'peak_mb': 512.5}
    }
    report = create_summary_report("Processing Summary", report_data)
    print(f"\n{report}") 