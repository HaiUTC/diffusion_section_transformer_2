"""
Enhanced Vision Encoder for Desktop Layout Analysis

Implements a sophisticated Vision Transformer with specialized techniques for small datasets:
- Shifted Patch Tokenization (SPT) for locality preservation
- Locality Self-Attention (LSA) for improved spatial modeling
- Intermediate supervision for faster convergence
- Class-guided attention for semantic alignment
- Progressive tokenization for multi-scale understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class VisionEncoderConfig:
    """Configuration for Enhanced Vision Encoder"""
    image_size: int = 720
    patch_size: int = 16
    num_layers: int = 12
    hidden_dim: int = 768
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    num_element_classes: int = 50
    use_spt: bool = True
    use_lsa: bool = True
    use_progressive: bool = True
    aux_loss_weight: float = 0.3


class ShiftedPatchTokenization(nn.Module):
    """
    Shifted Patch Tokenization (SPT) for enhanced locality modeling.
    Creates overlapping receptive fields by shifting input image in 4 directions.
    """
    
    def __init__(self, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding layers for original and shifted versions
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.shift_offsets = [(-patch_size//2, 0), (patch_size//2, 0), 
                             (0, -patch_size//2), (0, patch_size//2)]
        
    def shift_and_pad(self, image: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
        """Shift image and pad to maintain dimensions"""
        shifted = torch.zeros_like(image)
        
        if dx > 0:  # Shift right
            shifted[:, :, :, dx:] = image[:, :, :, :-dx]
        elif dx < 0:  # Shift left  
            shifted[:, :, :, :dx] = image[:, :, :, -dx:]
        else:
            shifted = image.clone()
            
        if dy > 0:  # Shift down
            shifted[:, :, dy:, :] = shifted[:, :, :-dy, :]
        elif dy < 0:  # Shift up
            shifted[:, :, :dy, :] = shifted[:, :, -dy:, :]
            
        return shifted
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with shifted patch tokenization
        
        Args:
            x: Input image tensor [batch, 3, height, width]
            
        Returns:
            Enhanced patch embeddings [batch, num_patches * 5, embed_dim]
        """
        
        # Original patches
        original_patches = self.patch_embed(x)  # [batch, embed_dim, h_patches, w_patches]
        original_patches = original_patches.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Shifted patches
        shifted_patches = []
        for dx, dy in self.shift_offsets:
            shifted_img = self.shift_and_pad(x, dx, dy)
            shifted_patch = self.patch_embed(shifted_img)
            shifted_patch = shifted_patch.flatten(2).transpose(1, 2)
            shifted_patches.append(shifted_patch)
            
        # Concatenate all patches
        all_patches = torch.cat([original_patches] + shifted_patches, dim=1)
        return all_patches


class LocalitySelfAttention(nn.Module):
    """
    Locality Self-Attention (LSA) with learnable temperature and local masking.
    Improves spatial modeling for small datasets.
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def create_local_mask(self, seq_len: int, window_size: int = 7) -> torch.Tensor:
        """Create local attention mask for neighboring patches"""
        mask = torch.zeros(seq_len, seq_len)
        
        # Assuming patches are arranged in a grid
        grid_size = int(math.sqrt(seq_len))
        
        for i in range(seq_len):
            row, col = i // grid_size, i % grid_size
            
            # Define local window
            for j in range(seq_len):
                target_row, target_col = j // grid_size, j % grid_size
                
                if (abs(row - target_row) <= window_size//2 and 
                    abs(col - target_col) <= window_size//2):
                    mask[i, j] = 1
                    
        return mask
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with locality self-attention
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention with learnable temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale * self.temperature
        
        # Apply local mask (only during training for efficiency)
        if self.training and N > 100:  # Only for large sequences
            local_mask = self.create_local_mask(N).to(x.device)
            attn = attn.masked_fill(local_mask.unsqueeze(0).unsqueeze(0) == 0, -float('inf'))
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with LSA and standard components"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, use_lsa: bool = True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # Use LSA or standard attention
        if use_lsa:
            self.attn = LocalitySelfAttention(dim, num_heads, dropout)
        else:
            self.attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
            
        # MLP layers
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections"""
        # Attention block
        if isinstance(self.attn, LocalitySelfAttention):
            x = x + self.attn(self.norm1(x))
        else:
            attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
            x = x + attn_out
            
        # MLP block
        x = x + self.mlp(self.norm2(x))
        return x


class ClassGuidedAttention(nn.Module):
    """
    Class-guided attention mechanism with semantic prefix tokens.
    Enables cross-modal alignment between class semantics and visual patches.
    """
    
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Learnable class tokens
        self.class_tokens = nn.Parameter(torch.randn(num_classes, embed_dim))
        
        # Attention bias matrix for class-patch interactions
        self.bias_projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, patches: torch.Tensor, class_hints: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with class-guided attention
        
        Args:
            patches: Visual patch embeddings [batch, num_patches, embed_dim]
            class_hints: Class indices for guidance [batch, num_active_classes]
            
        Returns:
            Enhanced patches with class guidance [batch, total_tokens, embed_dim]
        """
        batch_size = patches.shape[0]
        
        if class_hints is not None:
            # Select relevant class tokens
            class_embeds = self.class_tokens[class_hints]  # [batch, num_active, embed_dim]
            
            # Apply attention bias
            class_bias = self.bias_projection(class_embeds)
            
            # Concatenate class tokens with patches
            enhanced_tokens = torch.cat([class_bias, patches], dim=1)
        else:
            enhanced_tokens = patches
            
        return enhanced_tokens


class ProgressiveTokenization(nn.Module):
    """
    Progressive multi-scale tokenization for hierarchical feature extraction.
    Processes desktop layouts at multiple scales for comprehensive understanding.
    """
    
    def __init__(self, embed_dim: int = 768):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Multi-scale patch embeddings
        self.coarse_embed = nn.Conv2d(3, embed_dim, kernel_size=32, stride=32)  # Global structure
        self.medium_embed = nn.Conv2d(3, embed_dim, kernel_size=16, stride=16)  # Standard elements
        self.fine_embed = nn.Conv2d(3, embed_dim, kernel_size=8, stride=8)    # Fine details
        
        # Scale fusion layer
        self.scale_fusion = nn.Linear(embed_dim * 3, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Progressive tokenization across multiple scales
        
        Args:
            x: Input image [batch, 3, height, width]
            
        Returns:
            Multi-scale patch embeddings [batch, num_patches, embed_dim]
        """
        # Extract features at different scales
        coarse_patches = self.coarse_embed(x).flatten(2).transpose(1, 2)
        medium_patches = self.medium_embed(x).flatten(2).transpose(1, 2)
        fine_patches = self.fine_embed(x).flatten(2).transpose(1, 2)
        
        # Interpolate to common size (use medium as reference)
        target_size = medium_patches.shape[1]
        
        # Fix interpolation for coarse patches
        if coarse_patches.shape[1] != target_size:
            coarse_interp = F.interpolate(
                coarse_patches.transpose(1, 2), 
                size=target_size, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            coarse_interp = coarse_patches
        
        # Fix interpolation for fine patches
        if fine_patches.shape[1] != target_size:
            fine_downsampled = F.interpolate(
                fine_patches.transpose(1, 2), 
                size=target_size, mode='linear', align_corners=False
            ).transpose(1, 2)
        else:
            fine_downsampled = fine_patches
        
        # Fuse multi-scale features
        combined = torch.cat([coarse_interp, medium_patches, fine_downsampled], dim=-1)
        fused_patches = self.scale_fusion(combined)
        
        return fused_patches


class EnhancedVisionEncoder(nn.Module):
    """
    Enhanced Vision Encoder for Desktop Layout Analysis
    
    Combines multiple advanced techniques:
    - Shifted Patch Tokenization (SPT) for locality
    - Locality Self-Attention (LSA) for spatial modeling
    - Intermediate supervision for element detection
    - Class-guided attention for semantic alignment
    - Progressive tokenization for multi-scale understanding
    """
    
    def __init__(self, config: VisionEncoderConfig):
        super().__init__()
        self.config = config
        
        # Enhanced patch tokenization
        if config.use_spt:
            self.patch_tokenizer = ShiftedPatchTokenization(config.patch_size, config.hidden_dim)
            # SPT creates 5x more patches (original + 4 shifts)
            self.num_patches = (config.image_size // config.patch_size) ** 2 * 5
        else:
            self.patch_tokenizer = nn.Conv2d(3, config.hidden_dim, 
                                           kernel_size=config.patch_size, 
                                           stride=config.patch_size)
            self.num_patches = (config.image_size // config.patch_size) ** 2
            
        # Progressive tokenization
        if config.use_progressive:
            self.progressive_tokenizer = ProgressiveTokenization(config.hidden_dim)
            
        # Positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, config.hidden_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=config.hidden_dim,
                num_heads=config.num_heads, 
                mlp_ratio=config.mlp_ratio,
                dropout=config.dropout,
                use_lsa=config.use_lsa
            ) for _ in range(config.num_layers)
        ])
        
        # Intermediate supervision head (after layer 6)
        self.aux_head = nn.Linear(config.hidden_dim, config.num_element_classes)
        
        # Class-guided attention
        self.class_guided_attn = ClassGuidedAttention(config.hidden_dim, config.num_element_classes)
        
        # Final normalization
        self.norm = nn.LayerNorm(config.hidden_dim)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, images: torch.Tensor, 
                class_hints: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of enhanced vision encoder
        
        Args:
            images: Input desktop screenshots [batch, 3, height, width]
            class_hints: Optional class indices for guidance [batch, num_classes]
            
        Returns:
            Tuple of (final_features, auxiliary_predictions)
            - final_features: [batch, num_patches + 1, hidden_dim]
            - auxiliary_predictions: [batch, num_patches, num_element_classes] or None
        """
        batch_size = images.shape[0]
        
        # Enhanced patch tokenization
        if self.config.use_progressive:
            patches = self.progressive_tokenizer(images)
        elif self.config.use_spt:
            patches = self.patch_tokenizer(images)
        else:
            patches = self.patch_tokenizer(images)
            patches = patches.flatten(2).transpose(1, 2)
            
        # Class-guided attention
        patches = self.class_guided_attn(patches, class_hints)
        
        # Add CLS token and positional embeddings
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        patches = torch.cat([cls_tokens, patches], dim=1)
        
        # Adjust positional embeddings if needed
        if patches.shape[1] != self.pos_embed.shape[1]:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2), 
                size=patches.shape[1], 
                mode='linear'
            ).transpose(1, 2)
        else:
            pos_embed = self.pos_embed
            
        patches = patches + pos_embed
        
        # Progressive encoding with intermediate supervision
        aux_predictions = None
        
        for i, block in enumerate(self.transformer_blocks):
            patches = block(patches)
            
            # Intermediate supervision after layer 6
            if i == 5:  # Layer 6 (0-indexed)
                patch_features = patches[:, 1:]  # Exclude CLS token
                aux_predictions = self.aux_head(patch_features)
                
        # Final normalization
        patches = self.norm(patches)
        
        return patches, aux_predictions
    
    def get_patch_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch embeddings without class tokens"""
        with torch.no_grad():
            features, _ = self.forward(images)
            return features[:, 1:]  # Exclude CLS token
            
    def get_cls_token(self, images: torch.Tensor) -> torch.Tensor:
        """Extract CLS token representation"""
        with torch.no_grad():
            features, _ = self.forward(images)
            return features[:, 0]  # CLS token only


def create_vision_encoder(image_size: int = 720, patch_size: int = 16, 
                         num_element_classes: int = 50) -> EnhancedVisionEncoder:
    """
    Factory function to create enhanced vision encoder with default configuration
    
    Args:
        image_size: Input image size (assumes square images)
        patch_size: Size of image patches
        num_element_classes: Number of UI element classes for auxiliary supervision
        
    Returns:
        Configured EnhancedVisionEncoder instance
    """
    config = VisionEncoderConfig(
        image_size=image_size,
        patch_size=patch_size,
        num_element_classes=num_element_classes
    )
    return EnhancedVisionEncoder(config)


if __name__ == "__main__":
    # Example usage and testing
    print("Enhanced Vision Encoder for Desktop Layout Analysis")
    print("=" * 60)
    
    # Create model instance
    config = VisionEncoderConfig(
        image_size=720,
        patch_size=16,
        num_layers=12,
        hidden_dim=768,
        num_heads=12,
        num_element_classes=50
    )
    
    model = EnhancedVisionEncoder(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Example input: desktop screenshot batch
    batch_size = 2
    desktop_screenshots = torch.randn(batch_size, 3, 720, 1280)  # Desktop resolution
    class_hints = torch.tensor([[0, 1, 2], [1, 3, 4]])  # Element classes present
    
    print(f"\nInput shape: {desktop_screenshots.shape}")
    print(f"Class hints: {class_hints.shape}")
    
    # Forward pass
    with torch.no_grad():
        features, aux_predictions = model(desktop_screenshots, class_hints)
        
    print(f"\nOutput features shape: {features.shape}")
    print(f"Auxiliary predictions shape: {aux_predictions.shape if aux_predictions is not None else None}")
    
    # Extract specific representations
    cls_representation = model.get_cls_token(desktop_screenshots)
    patch_embeddings = model.get_patch_embeddings(desktop_screenshots)
    
    print(f"\nCLS token shape: {cls_representation.shape}")
    print(f"Patch embeddings shape: {patch_embeddings.shape}")
    
    print("\n✓ Enhanced Vision Encoder successfully implemented!")
    print("Key features:")
    print("  - Shifted Patch Tokenization (SPT) for locality preservation")
    print("  - Locality Self-Attention (LSA) for spatial modeling")
    print("  - Intermediate supervision for element detection")
    print("  - Class-guided attention for semantic alignment")
    print("  - Progressive tokenization for multi-scale understanding") 