#!/usr/bin/env python3
"""
Real Data Training Script for Layout Generation
Trains the model using actual screenshot images and layout data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
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
sys.path.append('../models')
from output_generation import OutputGenerationModule, MultiTaskLossFunction
from multimodal_fusion import MultimodalLayoutFusion

# Import real dataset
sys.path.append('../data')
from real_dataset import RealDataset, create_real_dataset_splits, real_collate_fn

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('real_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RealDataTrainer:
    """Enhanced trainer for real data with multimodal fusion"""
    
    def __init__(self, model: OutputGenerationModule, 
                 multimodal_fusion: MultimodalLayoutFusion,
                 loss_function: MultiTaskLossFunction,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.multimodal_fusion = multimodal_fusion.to(device)
        self.loss_function = loss_function
        self.device = device
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if device == 'cuda' else None
        
        # Tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_bleu': [],
            'val_bleu': [],
            'train_element_f1': [],
            'val_element_f1': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_bleu = 0.0
        self.patience = 15
        self.patience_counter = 0
    
    def setup_training(self, learning_rate: float = 1e-4, 
                      weight_decay: float = 0.01,
                      total_steps: int = 10000):
        """Setup optimizer and scheduler"""
        # Combined parameters from both models
        all_params = list(self.model.parameters()) + list(self.multimodal_fusion.parameters())
        
        # AdamW optimizer
        self.optimizer = optim.AdamW(
            all_params,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing with warmup
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=learning_rate * 0.01
        )
        
        # Warmup scheduler
        self.warmup_scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
    
    def compute_bleu_score(self, predictions: torch.Tensor, targets: torch.Tensor, 
                          tokenizer, max_samples: int = 10) -> float:
        """Compute approximate BLEU score"""
        try:
            bleu_scores = []
            
            # Convert to text for a subset of samples
            num_samples = min(predictions.size(0), max_samples)
            
            for i in range(num_samples):
                pred_ids = predictions[i].cpu().numpy().tolist()
                target_ids = targets[i].cpu().numpy().tolist()
                
                # Remove padding tokens
                pred_ids = [id for id in pred_ids if id not in [0, 1, 2, 3, 4, 5, 6]]  # Remove special tokens
                target_ids = [id for id in target_ids if id not in [0, 1, 2, 3, 4, 5, 6]]
                
                # Simple token-level BLEU approximation
                if len(target_ids) > 0:
                    matches = sum(1 for p, t in zip(pred_ids, target_ids) if p == t)
                    bleu = matches / max(len(target_ids), 1)
                    bleu_scores.append(bleu)
            
            return np.mean(bleu_scores) if bleu_scores else 0.0
            
        except Exception as e:
            logger.warning(f"Error computing BLEU score: {e}")
            return 0.0
    
    def compute_element_f1(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute F1 score for element classification"""
        try:
            # Convert to binary predictions
            preds_binary = (torch.sigmoid(predictions) > 0.5).float()
            
            # Calculate F1
            tp = (preds_binary * targets).sum().item()
            fp = (preds_binary * (1 - targets)).sum().item()
            fn = ((1 - preds_binary) * targets).sum().item()
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            return f1
        except Exception as e:
            logger.warning(f"Error computing element F1: {e}")
            return 0.0
    
    def train_epoch(self, train_loader: DataLoader, epoch: int, tokenizer) -> Dict[str, float]:
        """Train for one epoch with real data"""
        self.model.train()
        self.multimodal_fusion.train()
        
        epoch_losses = []
        epoch_bleu_scores = []
        epoch_element_f1_scores = []
        epoch_start_time = time.time()
        
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with autocast(enabled=self.scaler is not None):
                    # Multimodal fusion
                    fusion_outputs = self.multimodal_fusion(
                        screenshot=batch['images'],
                        html_structure=batch['html_tokens'],
                        element_hints=None  # Could use category_ids here
                    )
                    
                    # Output generation
                    outputs = self.model(
                        fused_features=fusion_outputs['visual_features'],
                        target_sequence=batch['layout_targets'][:, :-1],  # Teacher forcing
                        inference_mode='dual'
                    )
                    
                    # Prepare targets
                    targets = {
                        'detail_targets': batch['layout_targets'][:, 1:],  # Shifted for autoregressive
                        'semantic_targets': batch['layout_targets'][:, 1:],  # Same target for both heads
                        'element_targets': batch['element_labels']
                    }
                    
                    # Compute loss
                    losses = self.loss_function(outputs, targets)
                    total_loss = losses['total_loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                
                if self.scaler is not None:
                    self.scaler.scale(total_loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.multimodal_fusion.parameters()), 
                        max_norm=1.0
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.multimodal_fusion.parameters()), 
                        max_norm=1.0
                    )
                    self.optimizer.step()
                
                # Update learning rate
                current_step = epoch * total_batches + batch_idx
                if current_step < 1000:  # Warmup phase
                    self.warmup_scheduler.step()
                else:
                    self.scheduler.step()
                
                # Compute metrics
                epoch_losses.append(total_loss.item())
                
                # Compute BLEU score (subset of samples)
                if batch_idx % 10 == 0:  # Compute less frequently for efficiency
                    detail_preds = torch.argmax(outputs['detail_logits'], dim=-1)
                    bleu_score = self.compute_bleu_score(
                        detail_preds, targets['detail_targets'], tokenizer
                    )
                    epoch_bleu_scores.append(bleu_score)
                
                # Compute element F1
                element_f1 = self.compute_element_f1(
                    outputs['element_logits'], targets['element_targets']
                )
                epoch_element_f1_scores.append(element_f1)
                
                # Log progress
                if batch_idx % 20 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    logger.info(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}, '
                              f'Loss: {total_loss.item():.4f}, '
                              f'Element F1: {element_f1:.4f}, '
                              f'LR: {current_lr:.2e}')
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        epoch_time = time.time() - epoch_start_time
        self.training_history['epoch_times'].append(epoch_time)
        
        # Return epoch metrics
        return {
            'avg_loss': np.mean(epoch_losses) if epoch_losses else float('inf'),
            'avg_bleu': np.mean(epoch_bleu_scores) if epoch_bleu_scores else 0.0,
            'avg_element_f1': np.mean(epoch_element_f1_scores) if epoch_element_f1_scores else 0.0
        }
    
    def validate_epoch(self, val_loader: DataLoader, tokenizer) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.multimodal_fusion.eval()
        
        val_losses = []
        val_bleu_scores = []
        val_element_f1_scores = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                try:
                    # Move batch to device
                    batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                            for k, v in batch.items()}
                    
                    # Forward pass
                    fusion_outputs = self.multimodal_fusion(
                        screenshot=batch['images'],
                        html_structure=batch['html_tokens'],
                        element_hints=None
                    )
                    
                    outputs = self.model(
                        fused_features=fusion_outputs['visual_features'],
                        target_sequence=batch['layout_targets'][:, :-1],
                        inference_mode='dual'
                    )
                    
                    # Prepare targets
                    targets = {
                        'detail_targets': batch['layout_targets'][:, 1:],
                        'semantic_targets': batch['layout_targets'][:, 1:],
                        'element_targets': batch['element_labels']
                    }
                    
                    # Compute loss
                    losses = self.loss_function(outputs, targets)
                    val_losses.append(losses['total_loss'].item())
                    
                    # Compute metrics
                    detail_preds = torch.argmax(outputs['detail_logits'], dim=-1)
                    bleu_score = self.compute_bleu_score(
                        detail_preds, targets['detail_targets'], tokenizer
                    )
                    val_bleu_scores.append(bleu_score)
                    
                    element_f1 = self.compute_element_f1(
                        outputs['element_logits'], targets['element_targets']
                    )
                    val_element_f1_scores.append(element_f1)
                    
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        return {
            'avg_loss': np.mean(val_losses) if val_losses else float('inf'),
            'avg_bleu': np.mean(val_bleu_scores) if val_bleu_scores else 0.0,
            'avg_element_f1': np.mean(val_element_f1_scores) if val_element_f1_scores else 0.0
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'fusion_state_dict': self.multimodal_fusion.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_bleu': self.best_val_bleu,
            'training_history': self.training_history,
            'metrics': metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, checkpoint_dir: str, tokenizer) -> Dict[str, List]:
        """Complete training loop"""
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train samples: {len(train_loader.dataset)}")
        logger.info(f"Val samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch, tokenizer)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader, tokenizer)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['avg_loss'])
            self.training_history['val_loss'].append(val_metrics['avg_loss'])
            self.training_history['train_bleu'].append(train_metrics['avg_bleu'])
            self.training_history['val_bleu'].append(val_metrics['avg_bleu'])
            self.training_history['train_element_f1'].append(train_metrics['avg_element_f1'])
            self.training_history['val_element_f1'].append(val_metrics['avg_element_f1'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            logger.info(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            logger.info(f"Val Loss: {val_metrics['avg_loss']:.4f}")
            logger.info(f"Train BLEU: {train_metrics['avg_bleu']:.4f}")
            logger.info(f"Val BLEU: {val_metrics['avg_bleu']:.4f}")
            logger.info(f"Train Element F1: {train_metrics['avg_element_f1']:.4f}")
            logger.info(f"Val Element F1: {val_metrics['avg_element_f1']:.4f}")
            
            # Early stopping and checkpointing
            improved = False
            if val_metrics['avg_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_loss']
                improved = True
            
            if val_metrics['avg_bleu'] > self.best_val_bleu:
                self.best_val_bleu = val_metrics['avg_bleu']
                improved = True
            
            if improved:
                self.patience_counter = 0
                # Save best model
                best_model_path = Path(checkpoint_dir) / 'best_model.pt'
                self.save_checkpoint(epoch, val_metrics, str(best_model_path))
                logger.info("✓ Saved best model")
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pt'
                self.save_checkpoint(epoch, val_metrics, str(checkpoint_path))
        
        return self.training_history

def setup_experiment(args):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"real_data_{args.exp_name}_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir

def main():
    parser = argparse.ArgumentParser(description='Real Data Training for Layout Generation')
    
    # Data settings
    parser.add_argument('--data_dir', type=str, default='../../data',
                      help='Directory containing the real data')
    parser.add_argument('--image_size', type=int, nargs=2, default=[720, 1280],
                      help='Image size (height, width)')
    parser.add_argument('--max_html_length', type=int, default=512,
                      help='Maximum HTML sequence length')
    parser.add_argument('--max_layout_length', type=int, default=256,
                      help='Maximum layout sequence length')
    
    # Training settings
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size (smaller for real images)')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                      help='Learning rate (lower for real data)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    
    # Model settings
    parser.add_argument('--d_model', type=int, default=768,
                      help='Model dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                      help='Number of decoder layers')
    parser.add_argument('--vocab_size', type=int, default=50000,
                      help='Vocabulary size')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='layout_generation',
                      help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                      help='Output directory')
    parser.add_argument('--device', type=str, 
                      default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use')
    
    # Data split settings
    parser.add_argument('--train_ratio', type=float, default=0.8,
                      help='Training data ratio')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                      help='Validation data ratio')
    
    args = parser.parse_args()
    
    # Setup experiment
    args.exp_dir = setup_experiment(args)
    
    try:
        # Create datasets
        logger.info("Creating real datasets...")
        train_dataset, val_dataset, _ = create_real_dataset_splits(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=0.0,  # No test split for now
            image_size=tuple(args.image_size),
            max_html_length=args.max_html_length,
            max_layout_length=args.max_layout_length,
            vocab_size=args.vocab_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=real_collate_fn,
            num_workers=2,
            pin_memory=True if args.device == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=real_collate_fn,
            num_workers=2,
            pin_memory=True if args.device == 'cuda' else False
        )
        
        # Get vocabulary size from dataset
        actual_vocab_size = len(train_dataset.tokenizer.token_to_id)
        logger.info(f"Actual vocabulary size: {actual_vocab_size}")
        
        # Create models
        logger.info("Creating models...")
        
        # Multimodal fusion model
        multimodal_fusion = MultimodalLayoutFusion()
        
        # Output generation model
        model = OutputGenerationModule(
            d_model=args.d_model,
            num_decoder_layers=args.num_decoder_layers,
            num_heads=12,
            vocab_size=actual_vocab_size,
            num_semantic_classes=50,
            num_element_types=len(train_dataset.tokenizer.element_types),
            window_size=256,
            top_k=32,
            dropout=0.1
        )
        
        # Loss function
        loss_function = MultiTaskLossFunction(
            lambda_detail=0.4,
            lambda_semantic=0.3,
            lambda_element=0.2,
            lambda_consistency=0.1
        )
        
        # Create trainer
        trainer = RealDataTrainer(model, multimodal_fusion, loss_function, args.device)
        
        # Setup training
        total_steps = len(train_loader) * args.num_epochs
        trainer.setup_training(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            total_steps=total_steps
        )
        
        # Train model
        logger.info("Starting training...")
        history = trainer.train(
            train_loader, val_loader, 
            args.num_epochs, 
            str(args.exp_dir / 'checkpoints'),
            train_dataset.tokenizer
        )
        
        # Save results
        with open(args.exp_dir / 'training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Best validation BLEU: {trainer.best_val_bleu:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 