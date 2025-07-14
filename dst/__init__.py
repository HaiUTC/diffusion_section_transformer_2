"""
DST - Diffusion Section Transformer

Main package for multimodal layout generation pipeline.
Contains task 1.1 (data validation) and 1.2 (image processing) components.
"""

from . import data_validation
from . import image_processing
from . import utils

# Import main classes for convenience
from .data_validation import DataSample, DataValidator, HTMLProcessor, XOJLProcessor
from .image_processing import ImageProcessor, DatasetNormalizer, ViTPatchifier
from .utils import ProcessingStats, MemoryTracker, BatchProcessor, ConfigValidator

__version__ = "1.0.0"

__all__ = [
    # Modules
    'data_validation',
    'image_processing', 
    'utils',
    
    # Task 1.1 - Data Validation
    'DataSample',
    'DataValidator',
    'HTMLProcessor',
    'XOJLProcessor',
    
    # Task 1.2 - Image Processing
    'ImageProcessor',
    'DatasetNormalizer', 
    'ViTPatchifier',
    
    # Utilities
    'ProcessingStats',
    'MemoryTracker',
    'BatchProcessor',
    'ConfigValidator'
] 