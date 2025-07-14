"""
ViT Patchifier

Extracts patches from images for Vision Transformer training in task 1.2.
Handles 16×16 patches with position embeddings and UI-aware features.
Pure processing logic without any file I/O operations.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import logging
import math

logger = logging.getLogger(__name__)


class ViTPatchifier:
    """
    Extracts patches from images for Vision Transformer processing.
    
    Converts images into sequences of patches with position embeddings,
    optimized for desktop UI analysis. No file operations.
    """
    
    def __init__(self, 
                 patch_size: int = 16,
                 image_size: Tuple[int, int] = (720, 1280),
                 embedding_dim: int = 768):
        
        self.patch_size = patch_size
        self.image_height, self.image_width = image_size
        self.embedding_dim = embedding_dim
        
        # Calculate patch grid dimensions
        self.patches_height = self.image_height // patch_size
        self.patches_width = self.image_width // patch_size
        self.num_patches = self.patches_height * self.patches_width
        
        # Patch dimension (channels * patch_height * patch_width)
        self.patch_dim = 3 * patch_size * patch_size
        
        # Validate dimensions
        if self.image_height % patch_size != 0 or self.image_width % patch_size != 0:
            raise ValueError(
                f"Image size {image_size} not divisible by patch size {patch_size}"
            )
        
        # Position embedding types
        self.position_embedding_types = ['learnable', 'sinusoidal', 'none']
        self.current_embedding_type = 'learnable'
        
        # UI-aware features
        self.ui_aware_features = {
            'text_regions': True,      # Detect likely text regions
            'attention_masks': True,   # Create attention masks for UI elements
            'spatial_grouping': True   # Group patches by UI components
        }
        
        logger.info(f"Initialized ViTPatchifier: {self.patches_height}×{self.patches_width} = {self.num_patches} patches")
    
    def extract_patches(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Extract patches from image tensor.
        
        Args:
            image_tensor: Input tensor with shape (C, H, W) or (B, C, H, W)
            
        Returns:
            Tuple of (patches_tensor, patch_metadata)
        """
        original_shape = image_tensor.shape
        
        # Handle both single image and batch
        if image_tensor.dim() == 3:
            # Single image (C, H, W)
            batch_size = 1
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        elif image_tensor.dim() == 4:
            # Batch of images (B, C, H, W)
            batch_size = image_tensor.size(0)
        else:
            raise ValueError(f"Expected 3D or 4D tensor, got shape {original_shape}")
        
        # Validate tensor dimensions
        if image_tensor.size(-2) != self.image_height or image_tensor.size(-1) != self.image_width:
            raise ValueError(
                f"Expected image size {(self.image_height, self.image_width)}, "
                f"got {(image_tensor.size(-2), image_tensor.size(-1))}"
            )
        
        # Extract patches using unfold operation
        patches = self._extract_patches_unfold(image_tensor)
        
        # Generate position embeddings
        position_embeddings = self._generate_position_embeddings(batch_size)
        
        # Extract UI-aware features
        ui_features = self._extract_ui_features(patches) if self.ui_aware_features['text_regions'] else {}
        
        # Create attention masks
        attention_masks = self._create_attention_masks(patches) if self.ui_aware_features['attention_masks'] else None
        
        # Prepare metadata
        metadata = {
            'original_shape': original_shape,
            'patches_shape': tuple(patches.shape),
            'num_patches': self.num_patches,
            'patch_size': self.patch_size,
            'patch_dim': self.patch_dim,
            'patches_grid': (self.patches_height, self.patches_width),
            'position_embeddings_shape': tuple(position_embeddings.shape) if position_embeddings is not None else None,
            'embedding_type': self.current_embedding_type,
            'ui_features': ui_features,
            'has_attention_masks': attention_masks is not None
        }
        
        # Add attention masks to output if generated
        if attention_masks is not None:
            metadata['attention_masks'] = attention_masks
        
        return patches, metadata
    
    def _extract_patches_unfold(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """Extract patches using PyTorch unfold operation."""
        batch_size, channels, height, width = image_tensor.shape
        
        # Use unfold to extract patches
        # unfold(dimension, size, step)
        patches = image_tensor.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape to (batch_size, patches_height, patches_width, channels, patch_size, patch_size)
        patches = patches.contiguous()
        
        # Flatten patch dimensions and rearrange
        # Final shape: (batch_size, num_patches, patch_dim)
        patches = patches.view(
            batch_size, 
            self.patches_height, 
            self.patches_width, 
            channels * self.patch_size * self.patch_size
        )
        
        # Flatten spatial patch dimensions
        patches = patches.view(batch_size, self.num_patches, self.patch_dim)
        
        return patches
    
    def _generate_position_embeddings(self, batch_size: int) -> Optional[torch.Tensor]:
        """Generate position embeddings for patches."""
        if self.current_embedding_type == 'none':
            return None
        elif self.current_embedding_type == 'learnable':
            return self._generate_learnable_embeddings(batch_size)
        elif self.current_embedding_type == 'sinusoidal':
            return self._generate_sinusoidal_embeddings(batch_size)
        else:
            raise ValueError(f"Unknown embedding type: {self.current_embedding_type}")
    
    def _generate_learnable_embeddings(self, batch_size: int) -> torch.Tensor:
        """Generate learnable position embeddings."""
        # Initialize random embeddings that would be learned during training
        position_embeddings = torch.randn(batch_size, self.num_patches, self.embedding_dim) * 0.02
        return position_embeddings
    
    def _generate_sinusoidal_embeddings(self, batch_size: int) -> torch.Tensor:
        """Generate sinusoidal position embeddings."""
        position_embeddings = torch.zeros(self.num_patches, self.embedding_dim)
        
        # Create position indices
        position = torch.arange(0, self.num_patches, dtype=torch.float).unsqueeze(1)
        
        # Create dimension indices
        div_term = torch.exp(torch.arange(0, self.embedding_dim, 2).float() * 
                           (-math.log(10000.0) / self.embedding_dim))
        
        # Apply sine to even indices
        position_embeddings[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        if self.embedding_dim % 2 == 1:
            position_embeddings[:, 1::2] = torch.cos(position * div_term)
        else:
            position_embeddings[:, 1::2] = torch.cos(position * div_term)
        
        # Expand for batch size
        position_embeddings = position_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        
        return position_embeddings
    
    def _extract_ui_features(self, patches: torch.Tensor) -> Dict[str, Any]:
        """Extract UI-aware features from patches."""
        batch_size, num_patches, patch_dim = patches.shape
        
        # Reshape patches to analyze pixel content
        patches_reshaped = patches.view(batch_size, num_patches, 3, self.patch_size, self.patch_size)
        
        features = {}
        
        # Detect likely text regions based on high contrast
        if self.ui_aware_features['text_regions']:
            features['text_likelihood'] = self._detect_text_regions(patches_reshaped)
        
        # Detect uniform regions (likely backgrounds)
        features['uniformity_scores'] = self._calculate_uniformity_scores(patches_reshaped)
        
        # Calculate color diversity
        features['color_diversity'] = self._calculate_color_diversity(patches_reshaped)
        
        # Detect edge density (useful for UI elements)
        features['edge_density'] = self._calculate_edge_density(patches_reshaped)
        
        return features
    
    def _detect_text_regions(self, patches: torch.Tensor) -> torch.Tensor:
        """Detect patches likely to contain text based on contrast patterns."""
        batch_size, num_patches, channels, patch_h, patch_w = patches.shape
        
        # Convert to grayscale for contrast analysis
        grayscale = torch.mean(patches, dim=2)  # (batch, num_patches, patch_h, patch_w)
        
        # Calculate local variance as proxy for text likelihood
        variance = torch.var(grayscale.view(batch_size, num_patches, -1), dim=2)
        
        # Normalize to [0, 1] range
        text_likelihood = torch.sigmoid(variance * 10 - 5)  # Sigmoid to normalize
        
        return text_likelihood
    
    def _calculate_uniformity_scores(self, patches: torch.Tensor) -> torch.Tensor:
        """Calculate uniformity scores for patches (higher = more uniform)."""
        batch_size, num_patches, channels, patch_h, patch_w = patches.shape
        
        # Calculate standard deviation across pixels in each patch
        patches_flat = patches.view(batch_size, num_patches, -1)
        std_dev = torch.std(patches_flat, dim=2)
        
        # Invert so higher score means more uniform
        uniformity = 1.0 / (1.0 + std_dev)
        
        return uniformity
    
    def _calculate_color_diversity(self, patches: torch.Tensor) -> torch.Tensor:
        """Calculate color diversity in patches."""
        batch_size, num_patches, channels, patch_h, patch_w = patches.shape
        
        # Calculate variance across color channels
        channel_means = torch.mean(patches, dim=(3, 4))  # (batch, num_patches, channels)
        color_diversity = torch.var(channel_means, dim=2)
        
        return color_diversity
    
    def _calculate_edge_density(self, patches: torch.Tensor) -> torch.Tensor:
        """Calculate edge density in patches using gradient magnitude."""
        batch_size, num_patches, channels, patch_h, patch_w = patches.shape
        
        # Convert to grayscale
        grayscale = torch.mean(patches, dim=2)  # (batch, num_patches, patch_h, patch_w)
        
        # Calculate gradients
        grad_x = torch.abs(grayscale[:, :, :, 1:] - grayscale[:, :, :, :-1])
        grad_y = torch.abs(grayscale[:, :, 1:, :] - grayscale[:, :, :-1, :])
        
        # Calculate average gradient magnitude
        avg_grad_x = torch.mean(grad_x, dim=(2, 3))
        avg_grad_y = torch.mean(grad_y, dim=(2, 3))
        
        edge_density = (avg_grad_x + avg_grad_y) / 2
        
        return edge_density
    
    def _create_attention_masks(self, patches: torch.Tensor) -> torch.Tensor:
        """Create attention masks for UI-aware processing."""
        batch_size, num_patches, patch_dim = patches.shape
        
        # Create simple attention mask based on patch variance
        patches_reshaped = patches.view(batch_size, num_patches, 3, self.patch_size, self.patch_size)
        variance = torch.var(patches_reshaped.view(batch_size, num_patches, -1), dim=2)
        
        # Mask low-variance patches (likely empty regions)
        threshold = torch.quantile(variance, 0.1, dim=1, keepdim=True)
        attention_mask = (variance > threshold).float()
        
        return attention_mask
    
    def reconstruct_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct image from patches for visualization/validation.
        
        Args:
            patches: Patches tensor with shape (batch_size, num_patches, patch_dim)
            
        Returns:
            Reconstructed image tensor with shape (batch_size, C, H, W)
        """
        batch_size, num_patches, patch_dim = patches.shape
        
        if num_patches != self.num_patches:
            raise ValueError(f"Expected {self.num_patches} patches, got {num_patches}")
        
        # Reshape patches to include spatial dimensions
        patches_spatial = patches.view(
            batch_size, 
            self.patches_height, 
            self.patches_width, 
            3, 
            self.patch_size, 
            self.patch_size
        )
        
        # Reconstruct image using fold operation (inverse of unfold)
        image = torch.zeros(batch_size, 3, self.image_height, self.image_width)
        
        for i in range(self.patches_height):
            for j in range(self.patches_width):
                h_start = i * self.patch_size
                h_end = h_start + self.patch_size
                w_start = j * self.patch_size
                w_end = w_start + self.patch_size
                
                image[:, :, h_start:h_end, w_start:w_end] = patches_spatial[:, i, j, :, :, :]
        
        return image
    
    def set_position_embedding_type(self, embedding_type: str):
        """Set the type of position embeddings to use."""
        if embedding_type not in self.position_embedding_types:
            raise ValueError(f"Invalid embedding type. Choose from: {self.position_embedding_types}")
        self.current_embedding_type = embedding_type
        logger.info(f"Set position embedding type to: {embedding_type}")
    
    def get_patch_grid_coordinates(self) -> torch.Tensor:
        """Get 2D coordinates for each patch in the grid."""
        coords = torch.zeros(self.num_patches, 2)
        
        for i in range(self.patches_height):
            for j in range(self.patches_width):
                patch_idx = i * self.patches_width + j
                coords[patch_idx, 0] = i  # row
                coords[patch_idx, 1] = j  # column
        
        return coords
    
    def get_patchifier_info(self) -> Dict[str, Any]:
        """Get patchifier configuration information."""
        return {
            'patch_size': self.patch_size,
            'image_size': (self.image_height, self.image_width),
            'patch_grid': (self.patches_height, self.patches_width),
            'num_patches': self.num_patches,
            'patch_dim': self.patch_dim,
            'embedding_dim': self.embedding_dim,
            'current_embedding_type': self.current_embedding_type,
            'ui_aware_features': self.ui_aware_features
        }


if __name__ == "__main__":
    # Example usage
    patchifier = ViTPatchifier()
    
    # Create example image tensor
    image = torch.rand(3, 720, 1280)
    
    # Extract patches
    patches, metadata = patchifier.extract_patches(image)
    
    print(f"Original image shape: {image.shape}")
    print(f"Patches shape: {patches.shape}")
    print(f"Metadata: {metadata}")
    
    # Reconstruct image
    reconstructed = patchifier.reconstruct_image(patches)
    print(f"Reconstructed image shape: {reconstructed.shape}")
    
    # Check reconstruction accuracy
    mse = torch.mean((image.unsqueeze(0) - reconstructed) ** 2)
    print(f"Reconstruction MSE: {mse.item()}")
    
    print(f"Patchifier info: {patchifier.get_patchifier_info()}") 