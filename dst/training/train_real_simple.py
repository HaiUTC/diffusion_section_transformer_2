#!/usr/bin/env python3
"""
Simplified Real Data Training Script - macOS CPU Optimized
Trains the model using actual screenshot images and layout data with basic encoders.
Optimized for CPU training on macOS.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import time
import logging
from datetime import datetime

# Import our modules
import sys
from pathlib import Path

# Add the correct paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent / 'models'))
sys.path.append(str(current_dir.parent / 'data'))

from output_generation import OutputGenerationModule, MultiTaskLossFunction

# Import real dataset
from real_dataset import RealDataset, create_real_dataset_splits, real_collate_fn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_training_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SimpleVisionEncoder(nn.Module):
    """Simple vision encoder for real screenshots - CPU optimized"""
    
    def __init__(self, d_model: int = 256):  # Reduced from 768
        super().__init__()
        self.d_model = d_model
        
        # Smaller CNN backbone for CPU
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=4, padding=3),  # Larger stride for speed
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),  # Reduced from 256
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # Smaller 4x4 for CPU
        )
        
        # Project to model dimension
        self.projection = nn.Linear(128, d_model)  # Reduced from 256
        self.pos_encoding = nn.Parameter(torch.randn(16, d_model))  # 4x4 = 16 positions
        
    def forward(self, screenshot: torch.Tensor) -> torch.Tensor:
        batch_size = screenshot.size(0)
        
        # Extract CNN features
        features = self.cnn(screenshot)  # [batch, 128, 4, 4]
        
        # Reshape to sequence
        features = features.view(batch_size, 128, -1).transpose(1, 2)  # [batch, 16, 128]
        
        # Project to model dimension
        features = self.projection(features)  # [batch, 16, d_model]
        
        # Add positional encoding
        features = features + self.pos_encoding.unsqueeze(0)
        
        return features

class SimpleHTMLEncoder(nn.Module):
    """Simple HTML structure encoder - CPU optimized"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, max_seq_len: int = 256):  # Reduced dimensions
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=512, dropout=0.1),  # Reduced heads and ff
            num_layers=2  # Reduced from 3
        )
        
    def forward(self, html_tokens: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        seq_len = html_tokens.size(1)
        
        # Embed tokens
        embedded = self.embedding(html_tokens)  # [batch, seq_len, d_model]
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Create attention mask for transformer
        if attention_mask is not None:
            # Convert to boolean mask (True for masked positions)
            mask = (attention_mask == 0)
        else:
            mask = None
        
        # Apply transformer
        embedded = embedded.transpose(0, 1)  # [seq_len, batch, d_model]
        output = self.transformer(embedded, src_key_padding_mask=mask)
        output = output.transpose(0, 1)  # [batch, seq_len, d_model]
        
        return output

class SimpleFusionModel(nn.Module):
    """Simple fusion of vision and HTML features - CPU optimized"""
    
    def __init__(self, d_model: int = 256):  # Reduced from 768
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(d_model, 4, dropout=0.1)  # Reduced heads
        self.fusion_layer = nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=512)  # Reduced dimensions
        
    def forward(self, visual_features: torch.Tensor, html_features: torch.Tensor) -> torch.Tensor:
        # Cross attention: visual features attend to HTML
        visual_enhanced, _ = self.cross_attention(
            visual_features.transpose(0, 1),
            html_features.transpose(0, 1),
            html_features.transpose(0, 1)
        )
        visual_enhanced = visual_enhanced.transpose(0, 1)
        
        # Concatenate features
        fused = torch.cat([visual_enhanced, html_features], dim=1)
        
        # Apply fusion layer
        fused = fused.transpose(0, 1)
        fused = self.fusion_layer(fused)
        fused = fused.transpose(0, 1)
        
        return fused

class SimpleRealTrainer:
    """Simplified trainer for real data - macOS CPU optimized"""
    
    def __init__(self, device: str = 'cpu'):  # Default to CPU
        self.device = device
        self.best_val_loss = float('inf')
        self.patience = 5  # Reduced patience for faster testing
        self.patience_counter = 0
        
        # Initialize models
        self.vision_encoder = None
        self.html_encoder = None
        self.fusion_model = None
        self.output_model = None
        self.loss_function = None
        self.optimizer = None
        self.scheduler = None
        # No GradScaler for CPU
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_element_f1': [],
            'val_element_f1': [],
            'learning_rate': []
        }
        
        logger.info(f"Trainer initialized for device: {device}")
    
    def create_models(self, vocab_size: int, element_types: int, d_model: int = 256):  # Reduced default
        """Create all models - CPU optimized"""
        logger.info("Creating CPU-optimized models...")
        
        # Create encoders with smaller dimensions
        self.vision_encoder = SimpleVisionEncoder(d_model).to(self.device)
        self.html_encoder = SimpleHTMLEncoder(vocab_size, d_model).to(self.device)
        self.fusion_model = SimpleFusionModel(d_model).to(self.device)
        
        # Create output generation model with smaller dimensions
        self.output_model = OutputGenerationModule(
            d_model=d_model,
            num_decoder_layers=2,  # Reduced for CPU
            num_heads=4,  # Reduced for CPU
            vocab_size=vocab_size,
            num_semantic_classes=50,
            num_element_types=element_types,
            window_size=64,  # Reduced for CPU
            top_k=16,  # Reduced for CPU
            dropout=0.1
        ).to(self.device)
        
        # Loss function
        self.loss_function = MultiTaskLossFunction(
            lambda_detail=0.5,
            lambda_semantic=0.3,
            lambda_element=0.2,
            lambda_consistency=0.0  # Disable for simplicity
        )
        
        # Count parameters
        total_params = (
            sum(p.numel() for p in self.vision_encoder.parameters()) +
            sum(p.numel() for p in self.html_encoder.parameters()) +
            sum(p.numel() for p in self.fusion_model.parameters()) +
            sum(p.numel() for p in self.output_model.parameters())
        )
        
        logger.info(f"Total parameters: {total_params:,} (CPU optimized)")
    
    def setup_training(self, learning_rate: float = 5e-4, weight_decay: float = 0.01):  # Higher LR for CPU
        """Setup optimizer and scheduler"""
        # Collect all parameters
        all_params = (
            list(self.vision_encoder.parameters()) +
            list(self.html_encoder.parameters()) +
            list(self.fusion_model.parameters()) +
            list(self.output_model.parameters())
        )
        
        # Optimizer
        self.optimizer = optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=200, eta_min=learning_rate * 0.1  # Shorter schedule
        )
        
        logger.info(f"Training setup complete. Learning rate: {learning_rate}")
    
    def compute_element_f1(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute F1 score for element classification"""
        try:
            preds_binary = (torch.sigmoid(predictions) > 0.5).float()
            tp = (preds_binary * targets).sum().item()
            fp = (preds_binary * (1 - targets)).sum().item()
            fn = ((1 - preds_binary) * targets).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            return f1
        except:
            return 0.0
    
    def train_epoch(self, train_loader: DataLoader, vocab_size: int) -> Dict[str, float]:
        """Train for one epoch - CPU optimized"""
        self.vision_encoder.train()
        self.html_encoder.train()
        self.fusion_model.train()
        self.output_model.train()
        
        epoch_losses = []
        epoch_f1_scores = []
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move to device (CPU)
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass (no autocast for CPU)
                # Encode vision and HTML
                visual_features = self.vision_encoder(batch['images'])
                html_features = self.html_encoder(
                    batch['html_tokens'], 
                    batch['html_attention_mask']
                )
                
                # Fuse features
                fused_features = self.fusion_model(visual_features, html_features)
                
                # Generate output
                outputs = self.output_model(
                    fused_features=fused_features,
                    target_sequence=batch['layout_targets'][:, :-1],
                    inference_mode='dual'
                )
                
                # Prepare targets with proper clamping
                targets = {
                    'detail_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, vocab_size - 1),
                    'semantic_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, 49),  # num_semantic_classes - 1
                    'element_targets': batch['element_labels']
                }
                
                # Compute loss
                losses = self.loss_function(outputs, targets)
                total_loss = losses['total_loss']
                
                # Backward pass (no GradScaler for CPU)
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(
                    list(self.vision_encoder.parameters()) + 
                    list(self.html_encoder.parameters()) + 
                    list(self.fusion_model.parameters()) + 
                    list(self.output_model.parameters()), 
                    max_norm=1.0
                )
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Track metrics
                epoch_losses.append(total_loss.item())
                f1 = self.compute_element_f1(outputs['element_logits'], targets['element_targets'])
                epoch_f1_scores.append(f1)
                
                # Log progress (every batch for small datasets)
                if batch_idx % 1 == 0:  # Log every batch
                    lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f'Batch {batch_idx+1}/{len(train_loader)}, Loss: {total_loss.item():.4f}, '
                              f'F1: {f1:.4f}, LR: {lr:.2e}')
                    
            except Exception as e:
                logger.error(f"Error in batch {batch_idx}: {e}")
                continue
        
        return {
            'avg_loss': np.mean(epoch_losses) if epoch_losses else float('inf'),
            'avg_element_f1': np.mean(epoch_f1_scores) if epoch_f1_scores else 0.0
        }
    
    def validate_epoch(self, val_loader: DataLoader, vocab_size: int) -> Dict[str, float]:
        """Validate for one epoch - CPU optimized"""
        self.vision_encoder.eval()
        self.html_encoder.eval()
        self.fusion_model.eval()
        self.output_model.eval()
        
        val_losses = []
        val_f1_scores = []
        
        with torch.no_grad():
            for batch in val_loader:
                try:
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    visual_features = self.vision_encoder(batch['images'])
                    html_features = self.html_encoder(
                        batch['html_tokens'], 
                        batch['html_attention_mask']
                    )
                    fused_features = self.fusion_model(visual_features, html_features)
                    
                    outputs = self.output_model(
                        fused_features=fused_features,
                        target_sequence=batch['layout_targets'][:, :-1],
                        inference_mode='dual'
                    )
                    
                    # Prepare targets with proper clamping
                    targets = {
                        'detail_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, vocab_size - 1),
                        'semantic_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, 49),  # num_semantic_classes - 1
                        'element_targets': batch['element_labels']
                    }
                    
                    losses = self.loss_function(outputs, targets)
                    val_losses.append(losses['total_loss'].item())
                    
                    f1 = self.compute_element_f1(outputs['element_logits'], targets['element_targets'])
                    val_f1_scores.append(f1)
                    
                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue
        
        return {
            'avg_loss': np.mean(val_losses) if val_losses else float('inf'),
            'avg_element_f1': np.mean(val_f1_scores) if val_f1_scores else 0.0
        }
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, checkpoint_dir: str, vocab_size: int):
        """Complete training loop - CPU optimized"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting CPU training for {num_epochs} epochs")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        logger.info(f"Vocabulary size: {vocab_size}")
        logger.info(f"Device: {self.device}")
        
        total_start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 40)
            
            # Training
            train_metrics = self.train_epoch(train_loader, vocab_size)
            
            # Validation
            val_metrics = self.validate_epoch(val_loader, vocab_size)
            
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.history['train_loss'].append(train_metrics['avg_loss'])
            self.history['val_loss'].append(val_metrics['avg_loss'])
            self.history['train_element_f1'].append(train_metrics['avg_element_f1'])
            self.history['val_element_f1'].append(val_metrics['avg_element_f1'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print metrics
            logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s")
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            logger.info(f"Train F1: {train_metrics['avg_element_f1']:.4f}")
            logger.info(f"Val F1: {val_metrics['avg_element_f1']:.4f}")
            
            # Early stopping
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                self.patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'vision_encoder': self.vision_encoder.state_dict(),
                    'html_encoder': self.html_encoder.state_dict(),
                    'fusion_model': self.fusion_model.state_dict(),
                    'output_model': self.output_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'history': self.history
                }
                torch.save(checkpoint, Path(checkpoint_dir) / 'best_model.pt')
                logger.info("✓ Saved best model")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    logger.info("Early stopping triggered")
                    break
        
        total_time = time.time() - total_start_time
        logger.info(f"\nTraining completed in {total_time:.2f}s")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        
        return self.history

def main():
    parser = argparse.ArgumentParser(description='Simple Real Data Training - macOS CPU')
    parser.add_argument('--data_dir', type=str, default='../../data', help='Data directory')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (small for CPU)')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs (small for testing)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate (higher for CPU)')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension (reduced for CPU)')
    parser.add_argument('--output_dir', type=str, default='./experiments', help='Output directory')
    args = parser.parse_args()
    
    # Force CPU device for macOS
    device = 'cpu'
    logger.info(f"Running on macOS with device: {device}")
    
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = Path(args.output_dir) / f"simple_real_cpu_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create datasets with smaller dimensions for CPU
        logger.info("Creating datasets...")
        train_dataset, val_dataset, _ = create_real_dataset_splits(
            data_dir=args.data_dir,
            train_ratio=0.8,
            val_ratio=0.2,
            test_ratio=0.0,
            max_html_length=128,  # Reduced for CPU
            max_layout_length=64   # Reduced for CPU
        )
        
        # Data loaders with CPU-optimized settings
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            collate_fn=real_collate_fn, num_workers=0  # 0 for macOS compatibility
        )
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, 
            collate_fn=real_collate_fn, num_workers=0
        )
        
        # Create trainer
        trainer = SimpleRealTrainer(device=device)
        trainer.create_models(
            vocab_size=len(train_dataset.tokenizer.token_to_id),
            element_types=len(train_dataset.tokenizer.element_types),
            d_model=args.d_model
        )
        trainer.setup_training(args.learning_rate)
        
        # Train
        history = trainer.train(train_loader, val_loader, args.num_epochs, str(exp_dir), len(train_dataset.tokenizer.token_to_id))
        
        # Save results
        with open(exp_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Results saved to: {exp_dir}")
        
        # Print summary for Google Colab preparation
        print("\n" + "="*60)
        print("🎉 macOS CPU TRAINING SUCCESSFUL!")
        print("="*60)
        print(f"✓ Processed {len(train_dataset)} training samples")
        print(f"✓ Vocabulary size: {len(train_dataset.tokenizer.token_to_id)}")
        print(f"✓ Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"✓ Model saved to: {exp_dir}")
        print("\n🚀 Ready for Google Colab with train_real_data.py!")
        print("Recommended Colab settings:")
        print("  --batch_size 4")
        print("  --num_epochs 50") 
        print("  --learning_rate 1e-4")
        print("  --d_model 512")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main()) 