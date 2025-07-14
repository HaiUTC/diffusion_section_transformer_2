"""
Image Processing Module - Task 1.2

Handles advanced desktop-optimized image preprocessing, normalization,
and ViT patchification for transformer training.
"""

from .image_processor import ImageProcessor
from .dataset_normalizer import DatasetNormalizer
from .vit_patchifier import ViTPatchifier

__all__ = [
    'ImageProcessor',
    'DatasetNormalizer',
    'ViTPatchifier'
] 