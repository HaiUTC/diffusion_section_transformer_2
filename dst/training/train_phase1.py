#!/usr/bin/env python3
"""
Phase 1 Training Script for Decoder Architecture
Demonstrates complete training pipeline with mixed precision, scheduling, and evaluation.
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

# Import test components (reuse for real training)
sys.path.append('../../tests')
from test_phase1_training import MockDataset, collate_fn, MetricsCalculator, Phase1Trainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def setup_experiment(args):
    """Setup experiment directory and logging"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = f"{args.exp_name}_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(exp_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Experiment directory: {exp_dir}")
    return exp_dir

def create_datasets(args):
    """Create training and validation datasets"""
    logger.info("Creating datasets...")
    
    # For now, using MockDataset - replace with your actual dataset
    train_dataset = MockDataset(
        num_samples=args.train_samples,
        seq_len=args.sequence_length,
        vocab_size=args.vocab_size,
        num_semantic_classes=args.num_semantic_classes,
        num_element_types=args.num_element_types
    )
    
    val_dataset = MockDataset(
        num_samples=args.val_samples,
        seq_len=args.sequence_length,
        vocab_size=args.vocab_size,
        num_semantic_classes=args.num_semantic_classes,
        num_element_types=args.num_element_types
    )
    
    logger.info(f"Created train dataset with {len(train_dataset)} samples")
    logger.info(f"Created val dataset with {len(val_dataset)} samples")
    
    return train_dataset, val_dataset

def create_data_loaders(train_dataset, val_dataset, args):
    """Create data loaders with proper batching"""
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True if args.device == 'cuda' else False,
        drop_last=False
    )
    
    logger.info(f"Created train loader with {len(train_loader)} batches")
    logger.info(f"Created val loader with {len(val_loader)} batches")
    
    return train_loader, val_loader

def create_model(args):
    """Create the output generation model"""
    logger.info("Creating model...")
    
    model = OutputGenerationModule(
        d_model=args.d_model,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        vocab_size=args.vocab_size,
        num_semantic_classes=args.num_semantic_classes,
        num_element_types=args.num_element_types,
        window_size=args.window_size,
        top_k=args.top_k,
        dropout=args.dropout
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created with {total_params:,} total parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model

def create_loss_function(args):
    """Create multi-task loss function"""
    return MultiTaskLossFunction(
        lambda_detail=args.lambda_detail,
        lambda_semantic=args.lambda_semantic,
        lambda_element=args.lambda_element,
        lambda_consistency=args.lambda_consistency
    )

def run_small_training_test(model, loss_function, train_loader, val_loader, args):
    """Run small training test for debugging"""
    logger.info("Running small training test (1-2 epochs)...")
    
    # Create trainer
    trainer = Phase1Trainer(model, loss_function, device=args.device)
    
    # Setup training
    trainer.setup_training(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        total_steps=len(train_loader) * 2
    )
    
    # Train for 2 epochs
    test_exp_dir = args.exp_dir / 'small_test'
    test_exp_dir.mkdir(exist_ok=True)
    
    history = trainer.train(
        train_loader, val_loader, 
        num_epochs=2, 
        checkpoint_dir=str(test_exp_dir)
    )
    
    logger.info("Small training test completed!")
    logger.info(f"Final train loss: {history['train_loss'][-1]:.4f}")
    logger.info(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return history

def run_full_training(model, loss_function, train_loader, val_loader, args):
    """Run full training pipeline"""
    logger.info(f"Starting full training for {args.num_epochs} epochs...")
    
    # Create trainer
    trainer = Phase1Trainer(model, loss_function, device=args.device)
    
    # Setup training
    total_steps = len(train_loader) * args.num_epochs
    trainer.setup_training(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        total_steps=total_steps
    )
    
    # Setup TensorBoard logging
    tb_writer = SummaryWriter(args.exp_dir / 'tensorboard')
    
    # Train with TensorBoard logging
    history = trainer.train(
        train_loader, val_loader,
        num_epochs=args.num_epochs,
        checkpoint_dir=str(args.exp_dir / 'checkpoints')
    )
    
    # Log to TensorBoard
    for epoch, (train_loss, val_loss, lr) in enumerate(zip(
        history['train_loss'], history['val_loss'], history['learning_rate']
    )):
        tb_writer.add_scalar('Loss/Train', train_loss, epoch)
        tb_writer.add_scalar('Loss/Val', val_loss, epoch)
        tb_writer.add_scalar('Learning_Rate', lr, epoch)
    
    tb_writer.close()
    
    # Save training history
    with open(args.exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Full training completed!")
    logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
    
    return history

def evaluate_model(model, loss_function, val_loader, args):
    """Evaluate model against target thresholds"""
    logger.info("Evaluating model on validation set...")
    
    trainer = Phase1Trainer(model, loss_function, device=args.device)
    
    # Load best checkpoint if available
    best_checkpoint = args.exp_dir / 'checkpoints' / 'best_model.pt'
    if best_checkpoint.exists():
        logger.info(f"Loading best checkpoint: {best_checkpoint}")
        trainer.load_checkpoint(str(best_checkpoint))
    
    # Run validation
    val_metrics = trainer.validate_epoch(val_loader)
    
    # Define target thresholds
    target_thresholds = {
        'mock_bleu': 0.30,
        'element_f1': 0.85,
        'semantic_accuracy': 0.85
    }
    
    # Log results
    logger.info("="*50)
    logger.info("EVALUATION RESULTS")
    logger.info("="*50)
    
    results = {}
    for metric, threshold in target_thresholds.items():
        value = val_metrics.get(metric, 0.0)
        passed = value >= threshold
        status = "✓ PASSED" if passed else "✗ FAILED"
        
        logger.info(f"{metric.upper()}: {value:.4f} (target: ≥{threshold:.2f}) {status}")
        results[metric] = {
            'value': value,
            'threshold': threshold,
            'passed': passed
        }
    
    # Additional metrics
    logger.info("-"*50)
    logger.info("ADDITIONAL METRICS")
    logger.info("-"*50)
    
    for key, value in val_metrics.items():
        if key not in target_thresholds:
            logger.info(f"{key}: {value:.4f}")
    
    # Save evaluation results
    with open(args.exp_dir / 'evaluation_results.json', 'w') as f:
        json.dump({
            'target_thresholds': target_thresholds,
            'results': results,
            'all_metrics': val_metrics
        }, f, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Phase 1 Decoder Training')
    
    # Experiment settings
    parser.add_argument('--exp_name', type=str, default='phase1_decoder',
                      help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                      help='Output directory for experiments')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for training')
    
    # Data settings
    parser.add_argument('--train_samples', type=int, default=5000,
                      help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                      help='Number of validation samples')
    parser.add_argument('--sequence_length', type=int, default=100,
                      help='Input sequence length')
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loader workers')
    
    # Model settings
    parser.add_argument('--d_model', type=int, default=768,
                      help='Model dimension')
    parser.add_argument('--num_decoder_layers', type=int, default=6,
                      help='Number of decoder layers')
    parser.add_argument('--num_heads', type=int, default=12,
                      help='Number of attention heads')
    parser.add_argument('--vocab_size', type=int, default=50000,
                      help='Vocabulary size')
    parser.add_argument('--num_semantic_classes', type=int, default=50,
                      help='Number of semantic classes')
    parser.add_argument('--num_element_types', type=int, default=30,
                      help='Number of element types')
    parser.add_argument('--window_size', type=int, default=512,
                      help='Sparse attention window size')
    parser.add_argument('--top_k', type=int, default=64,
                      help='Sparse attention top-k')
    parser.add_argument('--dropout', type=float, default=0.1,
                      help='Dropout rate')
    
    # Training settings
    parser.add_argument('--num_epochs', type=int, default=50,
                      help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                      help='Weight decay')
    
    # Loss settings
    parser.add_argument('--lambda_detail', type=float, default=0.4,
                      help='Detail loss weight')
    parser.add_argument('--lambda_semantic', type=float, default=0.3,
                      help='Semantic loss weight')
    parser.add_argument('--lambda_element', type=float, default=0.2,
                      help='Element loss weight')
    parser.add_argument('--lambda_consistency', type=float, default=0.1,
                      help='Consistency loss weight')
    
    # Training modes
    parser.add_argument('--run_small_test', action='store_true',
                      help='Run small training test (1-2 epochs)')
    parser.add_argument('--run_full_training', action='store_true',
                      help='Run full training')
    parser.add_argument('--evaluate_only', action='store_true',
                      help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    # Setup experiment
    args.exp_dir = setup_experiment(args)
    
    # Create datasets and data loaders
    train_dataset, val_dataset = create_datasets(args)
    train_loader, val_loader = create_data_loaders(train_dataset, val_dataset, args)
    
    # Create model and loss function
    model = create_model(args)
    loss_function = create_loss_function(args)
    
    # Move model to device
    model = model.to(args.device)
    
    # Run training/evaluation based on arguments
    if args.run_small_test:
        logger.info("Running small training test...")
        run_small_training_test(model, loss_function, train_loader, val_loader, args)
    
    if args.run_full_training:
        logger.info("Running full training...")
        run_full_training(model, loss_function, train_loader, val_loader, args)
    
    # Always evaluate at the end
    if args.run_full_training or args.evaluate_only:
        logger.info("Running evaluation...")
        evaluate_model(model, loss_function, val_loader, args)
    
    logger.info("Training pipeline completed!")

if __name__ == "__main__":
    main() 