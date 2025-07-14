#!/usr/bin/env python3
"""
Debug script to identify the unpacking error
"""

import sys
import torch
import traceback

# Add paths
sys.path.append('../models')
sys.path.append('../data')

from real_dataset import RealDataset, real_collate_fn
from torch.utils.data import DataLoader

def test_dataset():
    """Test dataset loading"""
    print("Testing dataset...")
    
    dataset = RealDataset("../../data", max_html_length=128, max_layout_length=64)
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print("Sample keys:", list(sample.keys()))
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Test data loader
        loader = DataLoader(dataset, batch_size=1, collate_fn=real_collate_fn)
        batch = next(iter(loader))
        print("Batch keys:", list(batch.keys()))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        return batch
    return None

def test_models(batch):
    """Test model components"""
    print("\nTesting models...")
    
    # Simple vision encoder
    class SimpleVisionEncoder(torch.nn.Module):
        def __init__(self, d_model=256):
            super().__init__()
            self.cnn = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d((8, 8))
            )
            self.projection = torch.nn.Linear(64, d_model)
            self.pos_encoding = torch.nn.Parameter(torch.randn(64, d_model))
            
        def forward(self, x):
            batch_size = x.size(0)
            features = self.cnn(x)  # [batch, 64, 8, 8]
            features = features.view(batch_size, 64, -1).transpose(1, 2)  # [batch, 64, 64]
            features = self.projection(features)  # [batch, 64, d_model]
            features = features + self.pos_encoding.unsqueeze(0)
            return features
    
    # Simple HTML encoder
    class SimpleHTMLEncoder(torch.nn.Module):
        def __init__(self, vocab_size, d_model=256):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab_size, d_model)
            
        def forward(self, tokens, mask=None):
            return self.embedding(tokens)
    
    # Test vision encoder
    vision_encoder = SimpleVisionEncoder()
    visual_features = vision_encoder(batch['images'])
    print(f"Visual features shape: {visual_features.shape}")
    
    # Test HTML encoder
    vocab_size = 1000  # Use fixed vocab for testing
    html_encoder = SimpleHTMLEncoder(vocab_size)
    
    # Clamp HTML tokens to vocab size
    html_tokens_clamped = torch.clamp(batch['html_tokens'], 0, vocab_size - 1)
    html_features = html_encoder(html_tokens_clamped)
    print(f"HTML features shape: {html_features.shape}")
    
    return visual_features, html_features

def test_output_model(visual_features, html_features, batch):
    """Test output generation model"""
    print("\nTesting output model...")
    
    try:
        from output_generation import OutputGenerationModule, MultiTaskLossFunction
        
        # Create fused features (simple concatenation)
        fused_features = torch.cat([visual_features, html_features], dim=1)
        print(f"Fused features shape: {fused_features.shape}")
        
        # Create small output model
        vocab_size = 1000
        num_semantic_classes = 50
        num_element_types = 22  # Match the element_labels size
        
        model = OutputGenerationModule(
            d_model=256,
            num_decoder_layers=2,
            num_heads=4,
            vocab_size=vocab_size,
            num_semantic_classes=num_semantic_classes,
            num_element_types=num_element_types,
            window_size=64,
            top_k=16
        )
        
        print("Created output model")
        
        # Clamp targets to correct ranges
        layout_targets_clamped = torch.clamp(batch['layout_targets'], 0, vocab_size - 1)
        
        # Test forward pass
        print("Testing forward pass...")
        outputs = model(
            fused_features=fused_features,
            target_sequence=layout_targets_clamped[:, :-1],
            inference_mode='dual'
        )
        
        print("Forward pass successful!")
        print("Output keys:", list(outputs.keys()))
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.shape}")
        
        # Test loss with properly clamped targets
        print("Testing loss...")
        loss_fn = MultiTaskLossFunction()
        
        targets = {
            'detail_targets': torch.clamp(layout_targets_clamped[:, 1:], 0, vocab_size - 1),
            'semantic_targets': torch.clamp(layout_targets_clamped[:, 1:], 0, num_semantic_classes - 1),
            'element_targets': batch['element_labels']  # Already in correct format [0, 1]
        }
        
        print(f"Target shapes:")
        print(f"  detail_targets: {targets['detail_targets'].shape} (range: 0-{vocab_size-1})")
        print(f"  semantic_targets: {targets['semantic_targets'].shape} (range: 0-{num_semantic_classes-1})")
        print(f"  element_targets: {targets['element_targets'].shape}")
        
        losses = loss_fn(outputs, targets)
        print("Loss computation successful!")
        print("Loss keys:", list(losses.keys()))
        for key, value in losses.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error in output model: {e}")
        traceback.print_exc()
        return False

def main():
    print("=== DEBUGGING REAL DATA TRAINING ===")
    
    try:
        # Test dataset
        batch = test_dataset()
        if batch is None:
            print("No data available")
            return
        
        # Test model components
        visual_features, html_features = test_models(batch)
        
        # Test output model
        success = test_output_model(visual_features, html_features, batch)
        
        if success:
            print("\n✓ All components working!")
        else:
            print("\n✗ Some components failed")
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 