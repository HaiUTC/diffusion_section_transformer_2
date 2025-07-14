"""
Advanced Image Processor

Desktop-optimized image preprocessing for task 1.2.
Handles 1280×720 scaling, content-aware transformations, and quality optimization.
Pure processing logic without any file I/O operations.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ImageProcessor:
    """
    Advanced image processor optimized for desktop screenshots.
    
    Provides high-quality scaling, content-aware transformations,
    and desktop UI-specific optimizations. No file operations.
    """
    
    def __init__(self, target_resolution: Tuple[int, int] = (1280, 720)):
        self.target_width, self.target_height = target_resolution
        self.target_aspect_ratio = self.target_width / self.target_height
        
        # Desktop UI optimized padding color (neutral gray)
        self.padding_color = (240, 240, 240)
        
        # Quality settings for different modes
        self.quality_modes = {
            'maximum': {
                'interpolation': Image.Resampling.LANCZOS,
                'enhance_sharpness': True,
                'enhance_contrast': True,
                'noise_reduction': True
            },
            'balanced': {
                'interpolation': Image.Resampling.BICUBIC,
                'enhance_sharpness': True,
                'enhance_contrast': False,
                'noise_reduction': False
            },
            'fast': {
                'interpolation': Image.Resampling.BILINEAR,
                'enhance_sharpness': False,
                'enhance_contrast': False,
                'noise_reduction': False
            }
        }
        
        # Default mode
        self.current_mode = 'maximum'
    
    def set_quality_mode(self, mode: str):
        """Set processing quality mode."""
        if mode not in self.quality_modes:
            raise ValueError(f"Invalid quality mode: {mode}")
        self.current_mode = mode
        logger.info(f"Set quality mode to: {mode}")
    
    def process_image(self, image: Image.Image) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Process a single image with desktop optimizations.
        
        Args:
            image: PIL Image to process
            
        Returns:
            Tuple of (processed_tensor, metadata)
        """
        try:
            # Get quality settings
            settings = self.quality_modes[self.current_mode]
            
            # Store original dimensions
            original_width, original_height = image.size
            original_aspect_ratio = original_width / original_height
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Apply content-aware scaling
            scaled_image = self._content_aware_scale(image, settings)
            
            # Apply padding to reach target resolution
            padded_image = self._apply_smart_padding(scaled_image)
            
            # Apply quality enhancements
            enhanced_image = self._apply_enhancements(padded_image, settings)
            
            # Convert to tensor
            tensor = self._to_tensor(enhanced_image)
            
            # Prepare metadata
            metadata = {
                'original_size': (original_width, original_height),
                'target_size': (self.target_width, self.target_height),
                'original_aspect_ratio': original_aspect_ratio,
                'target_aspect_ratio': self.target_aspect_ratio,
                'scaling_applied': True,
                'padding_applied': True,
                'quality_mode': self.current_mode,
                'tensor_shape': tuple(tensor.shape),
                'tensor_dtype': str(tensor.dtype)
            }
            
            return tensor, metadata
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise ValueError(f"Image processing failed: {e}")
    
    def _content_aware_scale(self, image: Image.Image, settings: Dict[str, Any]) -> Image.Image:
        """Apply content-aware scaling optimized for desktop UIs."""
        original_width, original_height = image.size
        original_aspect_ratio = original_width / original_height
        
        # Determine scaling strategy
        if original_aspect_ratio > self.target_aspect_ratio:
            # Image is wider than target - scale by width
            new_width = self.target_width
            new_height = int(self.target_width / original_aspect_ratio)
        else:
            # Image is taller than target - scale by height
            new_height = self.target_height
            new_width = int(self.target_height * original_aspect_ratio)
        
        # Apply scaling
        scaled_image = image.resize(
            (new_width, new_height),
            resample=settings['interpolation']
        )
        
        # Apply text sharpening for upscaled images
        if (new_width > original_width or new_height > original_height) and settings['enhance_sharpness']:
            scaled_image = self._enhance_text_clarity(scaled_image)
        
        return scaled_image
    
    def _apply_smart_padding(self, image: Image.Image) -> Image.Image:
        """Apply intelligent padding to reach target resolution."""
        current_width, current_height = image.size
        
        # Calculate padding needed
        pad_width = max(0, self.target_width - current_width)
        pad_height = max(0, self.target_height - current_height)
        
        if pad_width == 0 and pad_height == 0:
            return image
        
        # Create new image with target dimensions
        padded_image = Image.new('RGB', (self.target_width, self.target_height), self.padding_color)
        
        # Calculate centering position
        paste_x = (self.target_width - current_width) // 2
        paste_y = (self.target_height - current_height) // 2
        
        # Paste original image
        padded_image.paste(image, (paste_x, paste_y))
        
        return padded_image
    
    def _apply_enhancements(self, image: Image.Image, settings: Dict[str, Any]) -> Image.Image:
        """Apply quality enhancements based on settings."""
        enhanced_image = image
        
        # Apply sharpness enhancement
        if settings['enhance_sharpness']:
            enhancer = ImageEnhance.Sharpness(enhanced_image)
            enhanced_image = enhancer.enhance(1.1)  # Subtle sharpening
        
        # Apply contrast enhancement
        if settings['enhance_contrast']:
            enhancer = ImageEnhance.Contrast(enhanced_image)
            enhanced_image = enhancer.enhance(1.05)  # Subtle contrast boost
        
        # Apply noise reduction
        if settings['noise_reduction']:
            enhanced_image = enhanced_image.filter(ImageFilter.SMOOTH_MORE)
        
        return enhanced_image
    
    def _enhance_text_clarity(self, image: Image.Image) -> Image.Image:
        """Enhance text clarity for desktop screenshots."""
        # Apply unsharp mask for text clarity
        enhanced = image.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        return enhanced
    
    def _to_tensor(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to normalized tensor."""
        # Convert to tensor with values in [0, 1]
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        tensor = transform(image)
        
        # Ensure tensor is in the correct format (C, H, W)
        if tensor.dim() == 3:
            return tensor  # Already (C, H, W)
        else:
            raise ValueError(f"Unexpected tensor dimensions: {tensor.shape}")
    
    def process_batch(self, images: list) -> Tuple[torch.Tensor, list]:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Images
            
        Returns:
            Tuple of (batch_tensor, metadata_list)
        """
        tensors = []
        metadata_list = []
        
        for i, image in enumerate(images):
            try:
                tensor, metadata = self.process_image(image)
                tensors.append(tensor)
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Error processing image {i}: {e}")
                # Create a black tensor as fallback
                fallback_tensor = torch.zeros(3, self.target_height, self.target_width)
                fallback_metadata = {
                    'error': str(e),
                    'is_fallback': True,
                    'tensor_shape': tuple(fallback_tensor.shape)
                }
                tensors.append(fallback_tensor)
                metadata_list.append(fallback_metadata)
        
        # Stack tensors into batch
        batch_tensor = torch.stack(tensors)
        
        return batch_tensor, metadata_list
    
    def calculate_scaling_metrics(self, original_size: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate scaling metrics for analysis."""
        original_width, original_height = original_size
        original_pixels = original_width * original_height
        target_pixels = self.target_width * self.target_height
        
        scaling_factor = np.sqrt(target_pixels / original_pixels)
        aspect_ratio_change = abs(
            (original_width / original_height) - self.target_aspect_ratio
        )
        
        return {
            'original_size': original_size,
            'target_size': (self.target_width, self.target_height),
            'scaling_factor': scaling_factor,
            'is_upscaling': scaling_factor > 1.0,
            'is_downscaling': scaling_factor < 1.0,
            'aspect_ratio_change': aspect_ratio_change,
            'requires_padding': aspect_ratio_change > 0.01
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor configuration information."""
        return {
            'target_resolution': (self.target_width, self.target_height),
            'target_aspect_ratio': self.target_aspect_ratio,
            'current_quality_mode': self.current_mode,
            'available_modes': list(self.quality_modes.keys()),
            'padding_color': self.padding_color,
            'optimized_for': 'desktop_screenshots'
        }


if __name__ == "__main__":
    # Example usage
    processor = ImageProcessor()
    
    # Create example image
    example_image = Image.new('RGB', (1920, 1080), (255, 255, 255))
    
    # Process image
    tensor, metadata = processor.process_image(example_image)
    
    print(f"Processed tensor shape: {tensor.shape}")
    print(f"Metadata: {metadata}")
    print(f"Processor info: {processor.get_processor_info()}") 