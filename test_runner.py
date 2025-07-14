#!/usr/bin/env python3
"""
Quick Test Runner for Phase 1 Implementation
Validates the complete training pipeline with minimal setup.
"""

import sys
import torch
import traceback
from pathlib import Path

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    try:
        # Test Phase 1 imports
        sys.path.append('dst/models')
        from output_generation import OutputGenerationModule, MultiTaskLossFunction
        from multimodal_fusion import MultimodalLayoutFusion
        
        sys.path.append('tests')
        from test_phase1_training import MockDataset, Phase1Trainer, MetricsCalculator
        
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        traceback.print_exc()
        return False

def test_model_creation():
    """Test model creation and basic functionality"""
    print("\nTesting model creation...")
    try:
        # Create model
        model = OutputGenerationModule(
            d_model=128,  # Smaller for quick testing
            num_decoder_layers=2,
            num_heads=4,
            vocab_size=1000,
            num_semantic_classes=10,
            num_element_types=5,
            window_size=32,
            top_k=16
        )
        
        # Test forward pass
        batch_size = 2
        seq_len = 20
        fused_features = torch.randn(batch_size, seq_len, 128)
        
        outputs = model(fused_features, inference_mode='dual')
        
        # Check outputs
        assert 'detail_logits' in outputs
        assert 'semantic_logits' in outputs
        assert 'element_logits' in outputs
        
        print(f"✓ Model created successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Output shapes: detail_logits={outputs['detail_logits'].shape}")
        return True
    except Exception as e:
        print(f"✗ Model creation error: {e}")
        traceback.print_exc()
        return False

def test_data_loading():
    """Test data loading and batching"""
    print("\nTesting data loading...")
    try:
        from torch.utils.data import DataLoader
        from test_phase1_training import collate_fn
        
        # Create dataset
        dataset = MockDataset(
            num_samples=20,
            seq_len=30,
            vocab_size=1000,
            num_semantic_classes=10,
            num_element_types=5
        )
        
        # Create data loader
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # Test one batch
        batch = next(iter(loader))
        
        assert 'fused_features' in batch
        assert 'detail_targets' in batch
        assert 'element_labels' in batch
        
        print(f"✓ Data loading successful")
        print(f"  - Dataset size: {len(dataset)}")
        print(f"  - Batch shape: {batch['fused_features'].shape}")
        return True
    except Exception as e:
        print(f"✗ Data loading error: {e}")
        traceback.print_exc()
        return False

def test_loss_computation():
    """Test loss computation"""
    print("\nTesting loss computation...")
    try:
        # Create model and loss function
        model = OutputGenerationModule(
            d_model=128, num_decoder_layers=2, num_heads=4,
            vocab_size=1000, num_semantic_classes=10, num_element_types=5
        )
        
        loss_fn = MultiTaskLossFunction()
        
        # Create mock data
        batch_size = 2
        seq_len = 20
        fused_features = torch.randn(batch_size, seq_len, 128)
        
        # Forward pass
        outputs = model(fused_features, inference_mode='dual')
        
        # Create targets
        targets = {
            'detail_targets': torch.randint(0, 1000, (batch_size, seq_len)),
            'semantic_targets': torch.randint(0, 10, (batch_size, seq_len)),
            'element_targets': torch.randn(batch_size, 5)
        }
        
        # Compute loss
        losses = loss_fn(outputs, targets)
        
        assert 'total_loss' in losses
        assert losses['total_loss'].requires_grad
        
        print(f"✓ Loss computation successful")
        print(f"  - Total loss: {losses['total_loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Loss computation error: {e}")
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\nTesting training step...")
    try:
        from torch.utils.data import DataLoader
        from test_phase1_training import collate_fn
        
        # Create components
        model = OutputGenerationModule(
            d_model=128, num_decoder_layers=2, num_heads=4,
            vocab_size=1000, num_semantic_classes=10, num_element_types=5
        )
        
        loss_fn = MultiTaskLossFunction()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Create data
        dataset = MockDataset(num_samples=8, seq_len=30, vocab_size=1000)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        batch = next(iter(loader))
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        outputs = model(batch['fused_features'], inference_mode='dual')
        targets = {
            'detail_targets': batch['detail_targets'],
            'semantic_targets': batch['semantic_targets'],
            'element_targets': batch['element_labels']
        }
        
        losses = loss_fn(outputs, targets)
        losses['total_loss'].backward()
        optimizer.step()
        
        print(f"✓ Training step successful")
        print(f"  - Loss: {losses['total_loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Training step error: {e}")
        traceback.print_exc()
        return False

def test_metrics_calculation():
    """Test metrics calculation"""
    print("\nTesting metrics calculation...")
    try:
        metrics_calc = MetricsCalculator()
        
        # Mock data
        outputs = {
            'detail_logits': torch.randn(2, 10, 1000),
            'element_logits': torch.randn(2, 5)
        }
        
        targets = {
            'detail_targets': torch.randint(0, 1000, (2, 10)),
            'element_targets': torch.randint(0, 2, (2, 5)).float()
        }
        
        losses = {
            'total_loss': torch.tensor(2.5),
            'detail_loss': torch.tensor(1.8)
        }
        
        # Update and compute metrics
        metrics_calc.update(outputs, targets, losses)
        metrics = metrics_calc.compute_metrics()
        
        assert 'avg_total_loss' in metrics
        assert 'element_f1' in metrics
        
        print(f"✓ Metrics calculation successful")
        print(f"  - Average total loss: {metrics['avg_total_loss']:.4f}")
        print(f"  - Element F1: {metrics['element_f1']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Metrics calculation error: {e}")
        traceback.print_exc()
        return False

def run_quick_integration_test():
    """Run a quick integration test"""
    print("\nRunning quick integration test...")
    try:
        # Import trainer
        trainer = Phase1Trainer(
            model=OutputGenerationModule(
                d_model=128, num_decoder_layers=2, num_heads=4,
                vocab_size=1000, num_semantic_classes=10, num_element_types=5
            ),
            loss_function=MultiTaskLossFunction(),
            device='cpu'
        )
        
        trainer.setup_training(learning_rate=1e-3, total_steps=10)
        
        # Create minimal dataset
        from torch.utils.data import DataLoader
        from test_phase1_training import collate_fn
        
        dataset = MockDataset(num_samples=8, seq_len=20, vocab_size=1000)
        loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
        
        # Run one training step
        trainer.model.train()
        batch = next(iter(loader))
        
        # Forward pass
        outputs = trainer.model(batch['fused_features'], inference_mode='dual')
        targets = {
            'detail_targets': batch['detail_targets'],
            'semantic_targets': batch['semantic_targets'],
            'element_targets': batch['element_labels']
        }
        
        losses = trainer.loss_function(outputs, targets)
        
        print(f"✓ Integration test successful")
        print(f"  - Forward pass completed")
        print(f"  - Loss computed: {losses['total_loss'].item():.4f}")
        return True
    except Exception as e:
        print(f"✗ Integration test error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("PHASE 1 IMPLEMENTATION VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_model_creation,
        test_data_loading,
        test_loss_computation,
        test_training_step,
        test_metrics_calculation,
        run_quick_integration_test
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Success rate: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("🎉 All tests passed! Phase 1 implementation is ready.")
        print("\nNext steps:")
        print("1. Run unit tests: cd tests && python -m pytest test_phase1_training.py -v")
        print("2. Run small training test: cd dst/training && python train_phase1.py --run_small_test")
        print("3. Run full training: python train_phase1.py --run_full_training")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nTroubleshooting:")
        print("1. Ensure all dependencies are installed")
        print("2. Check file paths and imports")
        print("3. Verify PyTorch installation")
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 