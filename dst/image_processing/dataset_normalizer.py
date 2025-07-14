"""
Dataset Normalizer

Computes normalization statistics across entire dataset for task 1.2.
Uses Welford's algorithm for memory-efficient computation.
Pure processing logic without any file I/O operations.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DatasetNormalizer:
    """
    Computes and applies dataset-specific normalization statistics.
    
    Uses Welford's online algorithm for memory-efficient computation
    of mean and standard deviation across large datasets.
    No file operations - only tensor processing.
    """
    
    def __init__(self, channels: int = 3):
        self.channels = channels
        
        # Welford's algorithm state
        self.count = 0
        self.mean = torch.zeros(channels, dtype=torch.float64)
        self.m2 = torch.zeros(channels, dtype=torch.float64)
        
        # ImageNet statistics for comparison
        self.imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        self.imagenet_std = torch.tensor([0.229, 0.224, 0.225])
        
        # Desktop UI color characteristics
        self.ui_characteristics = {
            'expected_mean_range': (0.7, 0.95),  # UI tends to be lighter
            'expected_std_range': (0.15, 0.35),  # UI has less variation
            'background_dominance': True,        # Backgrounds are common
            'text_contrast': True               # High text contrast
        }
        
        # Computed statistics
        self.computed_mean = None
        self.computed_std = None
        self.is_finalized = False
    
    def update(self, tensor_batch: torch.Tensor):
        """
        Update statistics with a batch of tensors.
        
        Args:
            tensor_batch: Batch of tensors with shape (B, C, H, W)
        """
        if tensor_batch.dim() != 4:
            raise ValueError(f"Expected 4D tensor (B,C,H,W), got {tensor_batch.shape}")
        
        if tensor_batch.size(1) != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {tensor_batch.size(1)}")
        
        # Flatten spatial dimensions but keep batch and channel dimensions
        batch_size, channels, height, width = tensor_batch.shape
        flattened = tensor_batch.view(batch_size, channels, -1)
        
        # Update statistics for each sample in batch
        for sample in flattened:
            self._update_welford(sample)
    
    def _update_welford(self, sample: torch.Tensor):
        """Update Welford statistics with a single sample."""
        # sample shape: (C, H*W)
        pixel_values = sample.transpose(0, 1)  # (H*W, C)
        
        for pixel in pixel_values:
            self.count += 1
            delta = pixel.double() - self.mean
            self.mean += delta / self.count
            delta2 = pixel.double() - self.mean
            self.m2 += delta * delta2
    
    def finalize(self) -> Dict[str, torch.Tensor]:
        """
        Finalize statistics computation.
        
        Returns:
            Dictionary containing computed statistics
        """
        if self.count < 2:
            logger.warning("Insufficient data for reliable statistics")
            self.computed_mean = torch.zeros(self.channels)
            self.computed_std = torch.ones(self.channels)
        else:
            self.computed_mean = self.mean.float()
            variance = self.m2 / (self.count - 1)
            self.computed_std = torch.sqrt(variance).float()
        
        self.is_finalized = True
        
        return {
            'mean': self.computed_mean,
            'std': self.computed_std
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive normalization statistics.
        
        Returns:
            Dictionary containing all computed statistics and analysis
        """
        if not self.is_finalized:
            self.finalize()
        
        return {
            'dataset_statistics': {
                'mean': self.computed_mean.tolist(),
                'std': self.computed_std.tolist(),
                'samples_processed': self.count,
                'channels': self.channels
            },
            'imagenet_comparison': self._compare_with_imagenet(),
            'ui_analysis': self._analyze_ui_characteristics(),
            'recommendations': self._get_normalization_recommendations()
        }
    
    def _compare_with_imagenet(self) -> Dict[str, Any]:
        """Compare computed statistics with ImageNet."""
        if not self.is_finalized:
            return {}
        
        mean_diff = torch.abs(self.computed_mean - self.imagenet_mean)
        std_diff = torch.abs(self.computed_std - self.imagenet_std)
        
        # Calculate similarity scores
        mean_similarity = 1.0 - torch.mean(mean_diff).item()
        std_similarity = 1.0 - torch.mean(std_diff).item()
        overall_similarity = (mean_similarity + std_similarity) / 2
        
        return {
            'imagenet_mean': self.imagenet_mean.tolist(),
            'imagenet_std': self.imagenet_std.tolist(),
            'mean_difference': mean_diff.tolist(),
            'std_difference': std_diff.tolist(),
            'mean_similarity': mean_similarity,
            'std_similarity': std_similarity,
            'overall_similarity': overall_similarity,
            'significantly_different': overall_similarity < 0.8
        }
    
    def _analyze_ui_characteristics(self) -> Dict[str, Any]:
        """Analyze UI-specific characteristics."""
        if not self.is_finalized:
            return {}
        
        analysis = {}
        
        # Check if mean is in expected UI range (lighter colors)
        mean_values = self.computed_mean.tolist()
        ui_mean_min, ui_mean_max = self.ui_characteristics['expected_mean_range']
        analysis['mean_in_ui_range'] = all(ui_mean_min <= m <= ui_mean_max for m in mean_values)
        
        # Check if std is in expected UI range (less variation)
        std_values = self.computed_std.tolist()
        ui_std_min, ui_std_max = self.ui_characteristics['expected_std_range']
        analysis['std_in_ui_range'] = all(ui_std_min <= s <= ui_std_max for s in std_values)
        
        # Overall UI conformity
        analysis['conforms_to_ui_patterns'] = (
            analysis['mean_in_ui_range'] and analysis['std_in_ui_range']
        )
        
        # Color channel analysis
        analysis['channel_analysis'] = {
            'red_channel': {'mean': mean_values[0], 'std': std_values[0]},
            'green_channel': {'mean': mean_values[1], 'std': std_values[1]},
            'blue_channel': {'mean': mean_values[2], 'std': std_values[2]}
        }
        
        # Detect color bias
        max_mean_diff = max(mean_values) - min(mean_values)
        analysis['has_color_bias'] = max_mean_diff > 0.1
        analysis['dominant_channel'] = ['red', 'green', 'blue'][np.argmax(mean_values)]
        
        return analysis
    
    def _get_normalization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for normalization strategy."""
        if not self.is_finalized:
            return {}
        
        comparison = self._compare_with_imagenet()
        ui_analysis = self._analyze_ui_characteristics()
        
        recommendations = {
            'use_dataset_specific': True,  # Always recommend dataset-specific for this use case
            'fallback_to_imagenet': False
        }
        
        # Reasoning
        if comparison.get('significantly_different', True):
            recommendations['reason'] = (
                "Dataset statistics significantly different from ImageNet. "
                "Desktop UI images have different color distributions."
            )
        else:
            recommendations['reason'] = (
                "Dataset statistics similar to ImageNet, but desktop UI "
                "characteristics warrant dataset-specific normalization."
            )
        
        # Quality assessment
        if self.count < 1000:
            recommendations['quality_warning'] = (
                f"Limited sample size ({self.count}). Consider more samples for robust statistics."
            )
        
        if ui_analysis.get('conforms_to_ui_patterns', False):
            recommendations['ui_optimized'] = True
            recommendations['confidence'] = 'high'
        else:
            recommendations['ui_optimized'] = False
            recommendations['confidence'] = 'medium'
            recommendations['note'] = "Statistics don't match typical UI patterns"
        
        return recommendations
    
    def normalize_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply normalization to a tensor.
        
        Args:
            tensor: Input tensor to normalize
            
        Returns:
            Normalized tensor
        """
        if not self.is_finalized:
            raise RuntimeError("Must finalize statistics before normalizing")
        
        # Reshape for broadcasting
        if tensor.dim() == 4:  # Batch of images (B, C, H, W)
            mean = self.computed_mean.view(1, -1, 1, 1)
            std = self.computed_std.view(1, -1, 1, 1)
        elif tensor.dim() == 3:  # Single image (C, H, W)
            mean = self.computed_mean.view(-1, 1, 1)
            std = self.computed_std.view(-1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.shape}")
        
        # Apply normalization
        normalized = (tensor - mean) / (std + 1e-8)  # Add epsilon for numerical stability
        
        return normalized
    
    def denormalize_tensor(self, normalized_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reverse normalization on a tensor.
        
        Args:
            normalized_tensor: Normalized tensor to denormalize
            
        Returns:
            Denormalized tensor
        """
        if not self.is_finalized:
            raise RuntimeError("Must finalize statistics before denormalizing")
        
        # Reshape for broadcasting
        if normalized_tensor.dim() == 4:  # Batch of images (B, C, H, W)
            mean = self.computed_mean.view(1, -1, 1, 1)
            std = self.computed_std.view(1, -1, 1, 1)
        elif normalized_tensor.dim() == 3:  # Single image (C, H, W)
            mean = self.computed_mean.view(-1, 1, 1)
            std = self.computed_std.view(-1, 1, 1)
        else:
            raise ValueError(f"Unsupported tensor dimensions: {normalized_tensor.shape}")
        
        # Reverse normalization
        denormalized = (normalized_tensor * std) + mean
        
        return denormalized
    
    def reset(self):
        """Reset all statistics for new dataset."""
        self.count = 0
        self.mean = torch.zeros(self.channels, dtype=torch.float64)
        self.m2 = torch.zeros(self.channels, dtype=torch.float64)
        self.computed_mean = None
        self.computed_std = None
        self.is_finalized = False
    
    def get_welford_state(self) -> Dict[str, Any]:
        """Get current Welford algorithm state for inspection."""
        return {
            'count': self.count,
            'running_mean': self.mean.tolist(),
            'running_m2': self.m2.tolist(),
            'is_finalized': self.is_finalized
        }


if __name__ == "__main__":
    # Example usage
    normalizer = DatasetNormalizer()
    
    # Create example batch
    batch = torch.rand(4, 3, 720, 1280)  # 4 images
    
    # Update statistics
    normalizer.update(batch)
    
    # Finalize and get statistics
    stats = normalizer.finalize()
    print(f"Computed statistics: {stats}")
    
    # Get comprehensive analysis
    analysis = normalizer.get_statistics()
    print(f"Analysis: {analysis}")
    
    # Normalize a tensor
    normalized = normalizer.normalize_tensor(batch[0])
    print(f"Normalized tensor shape: {normalized.shape}") 