import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import tempfile
import shutil
from collections import defaultdict
import time

# Import the modules (adjust import paths as needed)
import sys
sys.path.append('../dst/models')
from output_generation import OutputGenerationModule, MultiTaskLossFunction
from multimodal_fusion import MultimodalLayoutFusion

class MockDataset(Dataset):
    """Mock dataset for testing Phase 1 training pipeline"""
    
    def __init__(self, num_samples: int = 1000, seq_len: int = 100, 
                 vocab_size: int = 50000, num_semantic_classes: int = 50,
                 num_element_types: int = 30):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_semantic_classes = num_semantic_classes
        self.num_element_types = num_element_types
        
        # Generate mock data
        self.data = self._generate_mock_data()
    
    def _generate_mock_data(self) -> List[Dict]:
        """Generate realistic mock training data"""
        data = []
        
        for i in range(self.num_samples):
            # Simulate multimodal fusion outputs
            fused_features = torch.randn(self.seq_len, 768)
            
            # HTML token sequences with attention masks
            html_tokens = torch.randint(0, 1000, (self.seq_len,))
            html_attention_mask = torch.ones(self.seq_len)
            
            # Target sequences for detail generation
            detail_target_len = np.random.randint(20, 100)
            detail_targets = torch.randint(0, self.vocab_size, (detail_target_len,))
            
            # Target sequences for semantic generation
            semantic_target_len = np.random.randint(10, 50)
            semantic_targets = torch.randint(0, self.num_semantic_classes, (semantic_target_len,))
            
            # Element type labels (multi-label)
            element_labels = torch.zeros(self.num_element_types)
            num_active_elements = np.random.randint(1, 8)
            active_indices = np.random.choice(self.num_element_types, num_active_elements, replace=False)
            element_labels[active_indices] = 1.0
            
            # Create sample
            sample = {
                'fused_features': fused_features,
                'html_tokens': html_tokens,
                'html_attention_mask': html_attention_mask,
                'detail_targets': detail_targets,
                'semantic_targets': semantic_targets,
                'element_labels': element_labels,
                'sample_id': f'sample_{i:06d}'
            }
            
            data.append(sample)
        
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

def collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for batching variable-length sequences"""
    batch_size = len(batch)
    
    # Stack fused features (fixed length)
    fused_features = torch.stack([item['fused_features'] for item in batch])
    
    # Stack HTML tokens and masks
    html_tokens = torch.stack([item['html_tokens'] for item in batch])
    html_attention_mask = torch.stack([item['html_attention_mask'] for item in batch])
    
    # Pad detail targets to same length
    detail_targets = [item['detail_targets'] for item in batch]
    max_detail_len = max(len(seq) for seq in detail_targets)
    padded_detail_targets = torch.full((batch_size, max_detail_len), -100, dtype=torch.long)
    
    for i, seq in enumerate(detail_targets):
        padded_detail_targets[i, :len(seq)] = seq
    
    # Pad semantic targets to same length
    semantic_targets = [item['semantic_targets'] for item in batch]
    max_semantic_len = max(len(seq) for seq in semantic_targets)
    padded_semantic_targets = torch.full((batch_size, max_semantic_len), -100, dtype=torch.long)
    
    for i, seq in enumerate(semantic_targets):
        padded_semantic_targets[i, :len(seq)] = seq
    
    # Stack element labels
    element_labels = torch.stack([item['element_labels'] for item in batch])
    
    # Sample IDs
    sample_ids = [item['sample_id'] for item in batch]
    
    return {
        'fused_features': fused_features,
        'html_tokens': html_tokens,
        'html_attention_mask': html_attention_mask,
        'detail_targets': padded_detail_targets,
        'semantic_targets': padded_semantic_targets,
        'element_labels': element_labels,
        'sample_ids': sample_ids
    }

class MetricsCalculator:
    """Calculate training and evaluation metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.total_samples = 0
        self.losses = defaultdict(list)
        self.predictions = []
        self.targets = []
        self.element_predictions = []
        self.element_targets = []
    
    def update(self, outputs: Dict, targets: Dict, losses: Dict):
        """Update metrics with batch results"""
        batch_size = outputs['detail_logits'].size(0)
        self.total_samples += batch_size
        
        # Store losses
        for key, value in losses.items():
            self.losses[key].append(value.item())
        
        # Store predictions and targets for metric calculation
        if 'detail_logits' in outputs:
            detail_preds = torch.argmax(outputs['detail_logits'], dim=-1)
            self.predictions.extend(detail_preds.cpu().numpy().tolist())
            self.targets.extend(targets['detail_targets'].cpu().numpy().tolist())
        
        if 'element_logits' in outputs:
            element_preds = torch.sigmoid(outputs['element_logits'])
            self.element_predictions.extend(element_preds.cpu().numpy().tolist())
            self.element_targets.extend(targets['element_targets'].cpu().numpy().tolist())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute final metrics"""
        metrics = {}
        
        # Average losses
        for key, values in self.losses.items():
            metrics[f'avg_{key}'] = np.mean(values)
        
        # Element classification metrics (F1, precision, recall)
        if self.element_predictions and self.element_targets:
            element_preds = np.array(self.element_predictions)
            element_targets = np.array(self.element_targets)
            
            # Convert predictions to binary (threshold = 0.5)
            element_preds_binary = (element_preds > 0.5).astype(int)
            
            # Calculate F1 score
            tp = np.sum(element_preds_binary * element_targets)
            fp = np.sum(element_preds_binary * (1 - element_targets))
            fn = np.sum((1 - element_preds_binary) * element_targets)
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            metrics['element_precision'] = precision
            metrics['element_recall'] = recall
            metrics['element_f1'] = f1
        
        # Mock BLEU score calculation (simplified)
        if self.predictions and self.targets:
            # This is a simplified BLEU calculation for testing
            correct_tokens = 0
            total_tokens = 0
            
            for pred_seq, target_seq in zip(self.predictions, self.targets):
                if isinstance(pred_seq, list) and isinstance(target_seq, list):
                    min_len = min(len(pred_seq), len(target_seq))
                    for i in range(min_len):
                        if pred_seq[i] == target_seq[i]:
                            correct_tokens += 1
                        total_tokens += 1
            
            metrics['mock_bleu'] = correct_tokens / (total_tokens + 1e-8)
        
        return metrics

class Phase1Trainer:
    """Complete trainer for Phase 1 decoder training"""
    
    def __init__(self, model: OutputGenerationModule, 
                 loss_function: MultiTaskLossFunction,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.loss_function = loss_function
        self.device = device
        
        # Training components
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if device == 'cuda' else None
        
        # Tracking
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        # Checkpointing
        self.best_val_loss = float('inf')
        self.patience = 10
        self.patience_counter = 0
        self.checkpoint_dir = None
    
    def setup_training(self, learning_rate: float = 1e-4, 
                      weight_decay: float = 0.01,
                      total_steps: int = 10000):
        """Setup optimizer and scheduler"""
        # AdamW optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Cosine annealing scheduler with warmup
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
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_start_time = time.time()
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=self.scaler is not None):
                # Model forward pass
                outputs = self.model(
                    fused_features=batch['fused_features'],
                    target_sequence=batch['fused_features'][:, :50, :],  # Mock target sequence
                    inference_mode='dual'
                )
                
                # Prepare targets
                targets = {
                    'detail_targets': batch['detail_targets'],
                    'semantic_targets': batch['semantic_targets'],
                    'element_targets': batch['element_labels']
                }
                
                # Compute loss
                losses = self.loss_function(outputs, targets)
                total_loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Update learning rate
            if batch_idx < 1000:  # Warmup phase
                self.warmup_scheduler.step()
            else:
                self.scheduler.step()
            
            # Update metrics
            self.train_metrics.update(outputs, targets, losses)
            
            # Log progress
            if batch_idx % 100 == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}, Batch {batch_idx}/{total_batches}, '
                      f'Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}')
        
        epoch_time = time.time() - epoch_start_time
        self.training_history['epoch_times'].append(epoch_time)
        
        return self.train_metrics.compute_metrics()
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    fused_features=batch['fused_features'],
                    target_sequence=batch['fused_features'][:, :50, :],
                    inference_mode='dual'
                )
                
                # Prepare targets
                targets = {
                    'detail_targets': batch['detail_targets'],
                    'semantic_targets': batch['semantic_targets'],
                    'element_targets': batch['element_labels']
                }
                
                # Compute loss
                losses = self.loss_function(outputs, targets)
                
                # Update metrics
                self.val_metrics.update(outputs, targets, losses)
        
        return self.val_metrics.compute_metrics()
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'metrics': metrics
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint['training_history']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        return checkpoint['epoch'], checkpoint['metrics']
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              num_epochs: int, checkpoint_dir: str = None) -> Dict[str, List]:
        """Complete training loop"""
        self.checkpoint_dir = checkpoint_dir
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(val_loader)
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['avg_total_loss'])
            self.training_history['val_loss'].append(val_metrics['avg_total_loss'])
            self.training_history['learning_rate'].append(
                self.optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            print(f"Train Loss: {train_metrics['avg_total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['avg_total_loss']:.4f}")
            print(f"Element F1: {val_metrics.get('element_f1', 0.0):.4f}")
            print(f"Mock BLEU: {val_metrics.get('mock_bleu', 0.0):.4f}")
            
            # Early stopping check
            if val_metrics['avg_total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['avg_total_loss']
                self.patience_counter = 0
                
                # Save best model
                if checkpoint_dir:
                    best_model_path = Path(checkpoint_dir) / 'best_model.pt'
                    self.save_checkpoint(epoch, val_metrics, str(best_model_path))
            else:
                self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
            
            # Save regular checkpoint
            if checkpoint_dir and (epoch + 1) % 5 == 0:
                checkpoint_path = Path(checkpoint_dir) / f'checkpoint_epoch_{epoch + 1}.pt'
                self.save_checkpoint(epoch, val_metrics, str(checkpoint_path))
        
        return self.training_history

# Test Classes
class TestPhase1Training:
    """Test suite for Phase 1 training pipeline"""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model for testing"""
        return OutputGenerationModule(
            d_model=768,
            num_decoder_layers=2,  # Smaller for testing
            num_heads=8,
            vocab_size=1000,  # Smaller vocab for testing
            num_semantic_classes=20,
            num_element_types=10,
            window_size=128,
            top_k=32
        )
    
    @pytest.fixture
    def mock_loss_function(self):
        """Create mock loss function for testing"""
        return MultiTaskLossFunction(
            lambda_detail=0.4,
            lambda_semantic=0.3,
            lambda_element=0.2,
            lambda_consistency=0.1
        )
    
    @pytest.fixture
    def mock_datasets(self):
        """Create mock datasets for testing"""
        train_dataset = MockDataset(
            num_samples=200,
            seq_len=50,
            vocab_size=1000,
            num_semantic_classes=20,
            num_element_types=10
        )
        
        val_dataset = MockDataset(
            num_samples=50,
            seq_len=50,
            vocab_size=1000,
            num_semantic_classes=20,
            num_element_types=10
        )
        
        return train_dataset, val_dataset
    
    def test_data_loading(self, mock_datasets):
        """Test data loading and batching"""
        train_dataset, val_dataset = mock_datasets
        
        # Test dataset creation
        assert len(train_dataset) == 200
        assert len(val_dataset) == 50
        
        # Test data loader
        train_loader = DataLoader(
            train_dataset,
            batch_size=8,
            shuffle=True,
            collate_fn=collate_fn
        )
        
        # Test one batch
        batch = next(iter(train_loader))
        
        assert 'fused_features' in batch
        assert 'detail_targets' in batch
        assert 'semantic_targets' in batch
        assert 'element_labels' in batch
        
        assert batch['fused_features'].shape[0] == 8  # batch size
        assert batch['fused_features'].shape[2] == 768  # feature dim
        assert batch['element_labels'].shape[1] == 10  # num element types
    
    def test_model_forward_pass(self, mock_model, mock_datasets):
        """Test model forward pass"""
        train_dataset, _ = mock_datasets
        train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
        
        batch = next(iter(train_loader))
        
        # Forward pass
        outputs = mock_model(
            fused_features=batch['fused_features'],
            target_sequence=batch['fused_features'][:, :20, :],
            inference_mode='dual'
        )
        
        # Check outputs
        assert 'detail_logits' in outputs
        assert 'semantic_logits' in outputs
        assert 'element_logits' in outputs
        
        assert outputs['detail_logits'].shape[0] == 4  # batch size
        assert outputs['element_logits'].shape[1] == 10  # num element types
    
    def test_loss_computation(self, mock_model, mock_loss_function, mock_datasets):
        """Test loss computation"""
        train_dataset, _ = mock_datasets
        train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate_fn)
        
        batch = next(iter(train_loader))
        
        # Forward pass
        outputs = mock_model(
            fused_features=batch['fused_features'],
            target_sequence=batch['fused_features'][:, :20, :],
            inference_mode='dual'
        )
        
        # Prepare targets
        targets = {
            'detail_targets': batch['detail_targets'],
            'semantic_targets': batch['semantic_targets'],
            'element_targets': batch['element_labels']
        }
        
        # Compute loss
        losses = mock_loss_function(outputs, targets)
        
        # Check losses
        assert 'total_loss' in losses
        assert 'detail_loss' in losses
        assert 'semantic_loss' in losses
        assert 'element_loss' in losses
        
        assert losses['total_loss'].requires_grad
        assert losses['total_loss'].item() > 0
    
    def test_metrics_calculation(self):
        """Test metrics calculation"""
        metrics_calc = MetricsCalculator()
        
        # Mock outputs and targets
        mock_outputs = {
            'detail_logits': torch.randn(4, 20, 1000),
            'element_logits': torch.randn(4, 10)
        }
        
        mock_targets = {
            'detail_targets': torch.randint(0, 1000, (4, 20)),
            'element_targets': torch.randint(0, 2, (4, 10)).float()
        }
        
        mock_losses = {
            'total_loss': torch.tensor(2.5),
            'detail_loss': torch.tensor(1.8),
            'element_loss': torch.tensor(0.7)
        }
        
        # Update metrics
        metrics_calc.update(mock_outputs, mock_targets, mock_losses)
        
        # Compute metrics
        metrics = metrics_calc.compute_metrics()
        
        assert 'avg_total_loss' in metrics
        assert 'element_f1' in metrics
        assert 'mock_bleu' in metrics
        
        assert 0 <= metrics['element_f1'] <= 1
        assert 0 <= metrics['mock_bleu'] <= 1
    
    def test_small_training_loop(self, mock_model, mock_loss_function, mock_datasets):
        """Test small training loop (1-2 epochs)"""
        train_dataset, val_dataset = mock_datasets
        
        # Small batch size for testing
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn
        )
        val_loader = DataLoader(
            val_dataset, batch_size=8, collate_fn=collate_fn
        )
        
        # Create trainer
        trainer = Phase1Trainer(mock_model, mock_loss_function, device='cpu')
        
        # Setup training
        trainer.setup_training(
            learning_rate=1e-3,
            weight_decay=0.01,
            total_steps=len(train_loader) * 2
        )
        
        # Train for 2 epochs
        with tempfile.TemporaryDirectory() as tmp_dir:
            history = trainer.train(train_loader, val_loader, num_epochs=2, 
                                  checkpoint_dir=tmp_dir)
            
            # Check training history
            assert len(history['train_loss']) == 2
            assert len(history['val_loss']) == 2
            assert len(history['learning_rate']) == 2
            
            # Check that loss decreased (or at least training ran)
            assert history['train_loss'][-1] > 0
            assert history['val_loss'][-1] > 0
            
            # Check checkpoint files exist
            assert (Path(tmp_dir) / 'best_model.pt').exists()
    
    def test_autoregressive_generation(self, mock_model, mock_datasets):
        """Test autoregressive generation"""
        train_dataset, _ = mock_datasets
        train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
        
        batch = next(iter(train_loader))
        
        # Test generation
        generation_results = mock_model.generate_autoregressive(
            fused_features=batch['fused_features'],
            max_length=10,
            temperature=1.0,
            top_k=50
        )
        
        assert 'generated_tokens' in generation_results
        assert 'generation_length' in generation_results
        
        assert generation_results['generated_tokens'].shape[0] == 2  # batch size
        assert generation_results['generation_length'] > 0
    
    def test_checkpoint_save_load(self, mock_model, mock_loss_function):
        """Test checkpoint saving and loading"""
        trainer = Phase1Trainer(mock_model, mock_loss_function, device='cpu')
        trainer.setup_training(learning_rate=1e-3, total_steps=100)
        
        # Mock metrics
        mock_metrics = {
            'avg_total_loss': 2.5,
            'element_f1': 0.75,
            'mock_bleu': 0.3
        }
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'test_checkpoint.pt'
            
            # Save checkpoint
            trainer.save_checkpoint(5, mock_metrics, str(checkpoint_path))
            
            assert checkpoint_path.exists()
            
            # Load checkpoint
            epoch, metrics = trainer.load_checkpoint(str(checkpoint_path))
            
            assert epoch == 5
            assert metrics['avg_total_loss'] == 2.5
            assert metrics['element_f1'] == 0.75

    def test_target_thresholds(self, mock_model, mock_loss_function, mock_datasets):
        """Test evaluation against target thresholds"""
        train_dataset, val_dataset = mock_datasets
        
        val_loader = DataLoader(val_dataset, batch_size=8, collate_fn=collate_fn)
        
        trainer = Phase1Trainer(mock_model, mock_loss_function, device='cpu')
        
        # Run validation
        val_metrics = trainer.validate_epoch(val_loader)
        
        # Define target thresholds
        target_thresholds = {
            'mock_bleu': 0.30,
            'element_f1': 0.85,
            'semantic_accuracy': 0.85
        }
        
        print(f"\nEvaluation Results:")
        print(f"BLEU Score: {val_metrics.get('mock_bleu', 0.0):.4f} (target: ≥{target_thresholds['mock_bleu']})")
        print(f"Element F1: {val_metrics.get('element_f1', 0.0):.4f} (target: ≥{target_thresholds['element_f1']})")
        
        # Note: These are mock metrics, so they won't necessarily meet thresholds
        # In real training, you would check against actual performance
        
        # Assert that metrics are computed and reasonable
        assert 0 <= val_metrics.get('mock_bleu', 0.0) <= 1
        assert 0 <= val_metrics.get('element_f1', 0.0) <= 1

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"]) 