import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math

class VisualFeaturePyramid(nn.Module):
    """Feature Pyramid Network for multi-scale visual understanding of desktop layouts"""
    
    def __init__(self, input_dim: int = 768, output_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Extract from ViT layers 3, 6, 9, 12
        self.fpn_layers = [3, 6, 9, 12]
        
        # Lateral connections to reduce channel dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=False)
            for _ in self.fpn_layers
        ])
        
        # Top-down pathway convolutions
        self.fpn_convs = nn.ModuleList([
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)
            for _ in self.fpn_layers
        ])
        
        # Batch normalization for stable training
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm2d(output_dim) for _ in self.fpn_layers
        ])
        
    def forward(self, vit_layer_features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Input: List of ViT layer features [batch, seq_len, dim]
        Output: List of FPN features [batch, channels, height, width]
        """
        # Convert sequence features to spatial maps
        batch_size = vit_layer_features[0].size(0)
        seq_len = vit_layer_features[0].size(1) - 1  # Exclude CLS token
        spatial_size = int(math.sqrt(seq_len))  # Assume square patches
        
        spatial_features = []
        for feat in vit_layer_features:
            # Remove CLS token and reshape to spatial
            spatial_feat = feat[:, 1:, :].transpose(1, 2)  # [batch, dim, seq_len]
            spatial_feat = spatial_feat.view(batch_size, self.input_dim, spatial_size, spatial_size)
            spatial_features.append(spatial_feat)
        
        # Build top-down feature pyramid
        fpn_features = []
        for i, features in enumerate(reversed(spatial_features)):
            layer_idx = len(spatial_features) - 1 - i
            
            if i == 0:
                # Highest level - just apply lateral connection
                fpn_feat = self.lateral_convs[layer_idx](features)
            else:
                # Lower levels - combine lateral and top-down
                lateral = self.lateral_convs[layer_idx](features)
                top_down = F.interpolate(fpn_features[-1], scale_factor=2, mode='nearest')
                fpn_feat = lateral + top_down
            
            # Apply final convolution and batch norm
            fpn_feat = self.fpn_convs[layer_idx](fpn_feat)
            fpn_feat = self.bn_layers[layer_idx](fpn_feat)
            fpn_feat = F.relu(fpn_feat)
            
            fpn_features.append(fpn_feat)
        
        return list(reversed(fpn_features))  # Return in original order

class CrossModalAttention(nn.Module):
    """Cross-modal attention between visual and HTML structure features"""
    
    def __init__(self, d_model: int = 768, n_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Projection layers for common embedding space
        self.vision_proj = nn.Linear(d_model, d_model)
        self.html_proj = nn.Linear(d_model, d_model)
        
        # Cross-attention mechanisms
        self.vision_to_html_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.html_to_vision_attention = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # Layer normalization
        self.vision_norm = nn.LayerNorm(d_model)
        self.html_norm = nn.LayerNorm(d_model)
        
        # Feed-forward networks
        self.vision_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.html_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(self, visual_features: torch.Tensor, html_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Input: 
            visual_features: [batch, visual_seq_len, d_model]
            html_features: [batch, html_seq_len, d_model]
        Output:
            enhanced_visual: [batch, visual_seq_len, d_model]
            enhanced_html: [batch, html_seq_len, d_model]
            attention_weights: [batch, n_heads, html_seq_len, visual_seq_len]
        """
        # Project to common embedding space
        visual_proj = self.vision_proj(visual_features)
        html_proj = self.html_proj(html_features)
        
        # HTML queries attend to visual keys/values
        html_enhanced, html_to_vision_weights = self.html_to_vision_attention(
            query=html_proj,
            key=visual_proj,
            value=visual_proj,
            need_weights=True
        )
        
        # Visual queries attend to HTML keys/values
        visual_enhanced, vision_to_html_weights = self.vision_to_html_attention(
            query=visual_proj,
            key=html_proj,
            value=html_proj,
            need_weights=False
        )
        
        # Residual connections and normalization
        html_enhanced = self.html_norm(html_features + html_enhanced)
        visual_enhanced = self.vision_norm(visual_features + visual_enhanced)
        
        # Feed-forward networks
        html_enhanced = html_enhanced + self.html_ffn(html_enhanced)
        visual_enhanced = visual_enhanced + self.vision_ffn(visual_enhanced)
        
        return visual_enhanced, html_enhanced, html_to_vision_weights

class ContrastiveAlignment(nn.Module):
    """Contrastive learning for robust cross-modal alignment"""
    
    def __init__(self, temperature: float = 0.07, projection_dim: int = 128):
        super().__init__()
        self.temperature = temperature
        self.projection_dim = projection_dim
        
        # Projection heads for contrastive learning
        self.visual_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
        
        self.html_projector = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, projection_dim)
        )
    
    def create_positive_mask(self, batch_size: int, visual_seq_len: int, html_seq_len: int) -> torch.Tensor:
        """Create mask for positive pairs based on spatial-semantic correspondence"""
        # For simplicity, assume uniform correspondence
        # In practice, this would use ground truth alignments
        mask = torch.zeros(batch_size, visual_seq_len, html_seq_len)
        
        # Create sparse positive correspondences
        min_len = min(visual_seq_len, html_seq_len)
        for i in range(min_len):
            mask[:, i, i] = 1.0
        
        return mask
    
    def forward(self, visual_features: torch.Tensor, html_features: torch.Tensor) -> torch.Tensor:
        """
        Input:
            visual_features: [batch, visual_seq_len, 768]
            html_features: [batch, html_seq_len, 768]
        Output:
            contrastive_loss: scalar tensor
        """
        batch_size, visual_seq_len, _ = visual_features.shape
        html_seq_len = html_features.shape[1]
        
        # Project to contrastive space
        visual_proj = self.visual_projector(visual_features)  # [batch, visual_seq_len, projection_dim]
        html_proj = self.html_projector(html_features)      # [batch, html_seq_len, projection_dim]
        
        # Normalize features
        visual_norm = F.normalize(visual_proj, dim=-1)
        html_norm = F.normalize(html_proj, dim=-1)
        
        # Compute similarity matrix
        similarity = torch.matmul(visual_norm, html_norm.transpose(-2, -1)) / self.temperature
        
        # Create positive mask
        positive_mask = self.create_positive_mask(batch_size, visual_seq_len, html_seq_len)
        positive_mask = positive_mask.to(visual_features.device)
        
        # InfoNCE loss computation
        exp_sim = torch.exp(similarity)
        positive_sim = exp_sim * positive_mask
        
        # Sum over negative samples (all samples in denominator)
        denominator = exp_sim.sum(dim=-1, keepdim=True)
        
        # Avoid division by zero
        positive_sim = positive_sim.sum(dim=-1, keepdim=True)
        positive_sim = torch.clamp(positive_sim, min=1e-8)
        
        loss = -torch.log(positive_sim / denominator)
        
        # Average over valid positive pairs
        valid_pairs = positive_mask.sum(dim=-1, keepdim=True)
        valid_pairs = torch.clamp(valid_pairs, min=1)
        
        loss = (loss * valid_pairs).sum() / valid_pairs.sum()
        
        return loss

class SpatialAttentionFusion(nn.Module):
    """Spatial attention for focused layout region processing"""
    
    def __init__(self, channels: int = 256):
        super().__init__()
        self.channels = channels
        
        # Spatial attention mechanism
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        # Channel attention mechanism
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Input: feature_map [batch, channels, height, width]
        Output: attention_weighted_features [batch, channels, height, width]
        """
        # Channel attention
        channel_weight = self.channel_attention(feature_map)
        feature_map = feature_map * channel_weight
        
        # Spatial attention
        avg_pool = torch.mean(feature_map, dim=1, keepdim=True)  # [batch, 1, H, W]
        max_pool = torch.max(feature_map, dim=1, keepdim=True)[0]  # [batch, 1, H, W]
        
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [batch, 2, H, W]
        spatial_weight = self.spatial_attention(spatial_input)  # [batch, 1, H, W]
        
        return feature_map * spatial_weight

class MultimodalLayoutFusion(nn.Module):
    """Complete multimodal fusion architecture for layout-to-JSON generation"""
    
    def __init__(
        self,
        vision_encoder,
        html_encoder,
        d_model: int = 768,
        n_heads: int = 12,
        num_classes: int = 50,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.num_classes = num_classes
        
        # Modality-specific encoders (passed as parameters)
        self.vision_encoder = vision_encoder
        self.html_encoder = html_encoder
        
        # Feature pyramid for multi-scale processing
        self.visual_fpn = VisualFeaturePyramid(input_dim=d_model, output_dim=256)
        
        # Cross-modal components
        self.cross_attention = CrossModalAttention(d_model, n_heads, dropout)
        self.contrastive_alignment = ContrastiveAlignment(temperature=0.07)
        self.spatial_attention = SpatialAttentionFusion(channels=256)
        
        # Class-guided attention tokens
        self.class_tokens = nn.Parameter(torch.randn(num_classes, d_model))
        
        # Final fusion layers
        self.fusion_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Output projections
        self.output_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        screenshot: torch.Tensor,
        html_structure: torch.Tensor,
        element_hints: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Input:
            screenshot: [batch, 3, height, width]
            html_structure: [batch, html_seq_len, d_model] (pre-encoded)
            element_hints: [batch, num_hints] (optional class indices)
        Output:
            Dictionary containing all fusion outputs
        """
        batch_size = screenshot.size(0)
        
        # Extract modality-specific features
        vision_output = self.vision_encoder(screenshot, element_hints)
        visual_features = vision_output['visual_features']  # [batch, visual_seq_len, d_model]
        aux_predictions = vision_output.get('aux_predictions', None)
        
        html_features = self.html_encoder(html_structure)  # [batch, html_seq_len, d_model]
        
        # Build visual feature pyramid (extract layer features for FPN)
        # Note: This requires the vision encoder to return intermediate features
        if hasattr(self.vision_encoder, 'get_layer_features'):
            layer_features = self.vision_encoder.get_layer_features(screenshot)
            fpn_features = self.visual_fpn(layer_features)
        else:
            fpn_features = []
        
        # Class-guided attention if element hints provided
        if element_hints is not None:
            class_embeds = self.class_tokens[element_hints]  # [batch, num_hints, d_model]
            visual_features = torch.cat([class_embeds, visual_features], dim=1)
        
        # Cross-modal attention
        visual_enhanced, html_enhanced, attention_weights = self.cross_attention(
            visual_features, html_features
        )
        
        # Apply spatial attention to FPN features
        enhanced_fpn_features = []
        for fpn_feat in fpn_features:
            enhanced_fpn_feat = self.spatial_attention(fpn_feat)
            enhanced_fpn_features.append(enhanced_fpn_feat)
        
        # Contrastive alignment loss
        contrastive_loss = self.contrastive_alignment(visual_enhanced, html_enhanced)
        
        # Final fusion through transformer
        # Concatenate enhanced features
        fused_features = torch.cat([visual_enhanced, html_enhanced], dim=1)
        fused_features = self.dropout(fused_features)
        
        # Apply final transformer layers
        fused_output = self.fusion_transformer(fused_features)
        
        # Output projection
        final_output = self.output_projection(fused_output)
        
        return {
            'fused_features': final_output,
            'visual_features': visual_enhanced,
            'html_features': html_enhanced,
            'fpn_features': enhanced_fpn_features,
            'attention_weights': attention_weights,
            'aux_predictions': aux_predictions,
            'contrastive_loss': contrastive_loss
        }

class MultimodalLossFunction(nn.Module):
    """Integrated loss function for multimodal training"""
    
    def __init__(
        self,
        lambda_aux: float = 0.3,
        lambda_contrastive: float = 0.1,
        lambda_spatial: float = 0.05
    ):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.lambda_contrastive = lambda_contrastive
        self.lambda_spatial = lambda_spatial
        
        # Loss components
        self.main_loss = nn.CrossEntropyLoss()
        self.aux_loss = nn.BCEWithLogitsLoss()
        self.spatial_loss = nn.MSELoss()
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute integrated loss from all components
        """
        losses = {}
        
        # Main generation loss
        if 'main_logits' in outputs and 'main_targets' in targets:
            losses['main_loss'] = self.main_loss(outputs['main_logits'], targets['main_targets'])
        
        # Auxiliary element detection loss
        if 'aux_predictions' in outputs and 'element_targets' in targets:
            losses['aux_loss'] = self.aux_loss(outputs['aux_predictions'], targets['element_targets'])
        
        # Contrastive alignment loss
        if 'contrastive_loss' in outputs:
            losses['contrastive_loss'] = outputs['contrastive_loss']
        
        # Spatial attention regularization
        if 'attention_weights' in outputs and 'spatial_targets' in targets:
            losses['spatial_loss'] = self.spatial_loss(
                outputs['attention_weights'], targets['spatial_targets']
            )
        
        # Compute total loss
        total_loss = 0.0
        if 'main_loss' in losses:
            total_loss += losses['main_loss']
        if 'aux_loss' in losses:
            total_loss += self.lambda_aux * losses['aux_loss']
        if 'contrastive_loss' in losses:
            total_loss += self.lambda_contrastive * losses['contrastive_loss']
        if 'spatial_loss' in losses:
            total_loss += self.lambda_spatial * losses['spatial_loss']
        
        losses['total_loss'] = total_loss
        
        return losses

# Example usage and testing
if __name__ == "__main__":
    # Mock encoders for testing
    class MockVisionEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(3 * 224 * 224, 768)
            
        def forward(self, x, element_hints=None):
            batch_size = x.size(0)
            x_flat = x.view(batch_size, -1)
            features = self.linear(x_flat).unsqueeze(1)  # [batch, 1, 768]
            return {
                'visual_features': features,
                'aux_predictions': torch.randn(batch_size, 1, 50)
            }
    
    class MockHTMLEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1000, 768)
            
        def forward(self, x):
            return self.embedding(x)
    
    # Initialize components
    vision_encoder = MockVisionEncoder()
    html_encoder = MockHTMLEncoder()
    
    # Create multimodal fusion model
    fusion_model = MultimodalLayoutFusion(
        vision_encoder=vision_encoder,
        html_encoder=html_encoder,
        d_model=768,
        n_heads=12,
        num_classes=50
    )
    
    # Create loss function
    loss_function = MultimodalLossFunction()
    
    # Test forward pass
    batch_size = 2
    screenshot = torch.randn(batch_size, 3, 224, 224)
    html_structure = torch.randint(0, 1000, (batch_size, 50))
    element_hints = torch.randint(0, 50, (batch_size, 3))
    
    # Forward pass
    outputs = fusion_model(screenshot, html_structure, element_hints)
    
    # Print output shapes
    print("Multimodal Fusion Outputs:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
        elif isinstance(value, list):
            print(f"  {key}: List of {len(value)} tensors")
        else:
            print(f"  {key}: {type(value)}")
    
    # Test loss computation
    targets = {
        'main_targets': torch.randint(0, 1000, (batch_size,)),
        'element_targets': torch.randn(batch_size, 1, 50),
        'spatial_targets': torch.randn_like(outputs['attention_weights'])
    }
    
    # Add main logits to outputs for loss computation
    outputs['main_logits'] = torch.randn(batch_size, 1000)
    
    losses = loss_function(outputs, targets)
    
    print("\nLoss Components:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")
    
    print(f"\nTotal parameters: {sum(p.numel() for p in fusion_model.parameters()):,}") 