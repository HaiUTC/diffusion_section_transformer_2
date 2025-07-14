#!/usr/bin/env python3
"""
Quick Test Script for macOS CPU Training
Tests the setup before running full training.
"""

import sys
import torch
import time
from pathlib import Path

def test_system():
    """Test system capabilities"""
    print("🔍 TESTING macOS SYSTEM")
    print("=" * 50)
    
    # Check PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"CPU cores: {torch.get_num_threads()}")
    
    # Test tensor operations
    start_time = time.time()
    x = torch.randn(100, 100)
    y = torch.matmul(x, x)
    cpu_time = time.time() - start_time
    print(f"CPU matrix multiplication (100x100): {cpu_time:.4f}s")
    
    return True

def test_data_loading():
    """Test data loading"""
    print("\n🗃️ TESTING DATA LOADING")
    print("=" * 50)
    
    try:
        # Fix import path
        sys.path.append('dst/data')
        from real_dataset import RealDataset
        
        data_dir = "data"
        if not Path(data_dir).exists():
            print(f"❌ Data directory {data_dir} not found")
            return False
        
        dataset = RealDataset(data_dir, max_html_length=64, max_layout_length=32)
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"✅ Vocabulary size: {len(dataset.tokenizer.token_to_id)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"✅ Sample loaded: {sample['image'].shape}")
            return True
        else:
            print("⚠️ No samples in dataset - add data samples to data/ directory")
            print("Expected structure: data/example_001/screenshot.png and data.json")
            return False
            
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_model_creation():
    """Test model creation and forward pass"""
    print("\n🧠 TESTING MODEL CREATION")
    print("=" * 50)
    
    try:
        # Fix import paths
        sys.path.append('dst/training')
        sys.path.append('dst/models')
        
        from train_real_simple import SimpleVisionEncoder, SimpleHTMLEncoder, SimpleFusionModel
        from output_generation import OutputGenerationModule
        
        # Test small models
        d_model = 128  # Very small for testing
        vocab_size = 1000
        
        # Create models
        vision_encoder = SimpleVisionEncoder(d_model)
        html_encoder = SimpleHTMLEncoder(vocab_size, d_model)
        fusion_model = SimpleFusionModel(d_model)
        output_model = OutputGenerationModule(
            d_model=d_model,
            num_decoder_layers=1,
            num_heads=2,
            vocab_size=vocab_size,
            num_semantic_classes=50,
            num_element_types=22,
            window_size=32,
            top_k=8
        )
        
        print(f"✅ Models created")
        
        # Test forward pass
        batch_size = 1
        
        # Test vision encoder
        dummy_image = torch.randn(batch_size, 3, 720, 1280)
        start_time = time.time()
        visual_features = vision_encoder(dummy_image)
        vision_time = time.time() - start_time
        print(f"✅ Vision encoder: {visual_features.shape} in {vision_time:.4f}s")
        
        # Test HTML encoder
        dummy_html = torch.randint(0, vocab_size, (batch_size, 64))
        html_features = html_encoder(dummy_html)
        print(f"✅ HTML encoder: {html_features.shape}")
        
        # Test fusion
        fused_features = fusion_model(visual_features, html_features)
        print(f"✅ Fusion model: {fused_features.shape}")
        
        # Test output model
        dummy_targets = torch.randint(0, vocab_size, (batch_size, 32))
        outputs = output_model(fused_features, dummy_targets, inference_mode='dual')
        print(f"✅ Output model: {list(outputs.keys())}")
        
        # Count parameters
        total_params = (
            sum(p.numel() for p in vision_encoder.parameters()) +
            sum(p.numel() for p in html_encoder.parameters()) +
            sum(p.numel() for p in fusion_model.parameters()) +
            sum(p.numel() for p in output_model.parameters())
        )
        print(f"✅ Total parameters: {total_params:,}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🏋️ TESTING TRAINING STEP")
    print("=" * 50)
    
    try:
        # Fix import paths
        sys.path.append('dst/training')
        sys.path.append('dst/models')
        
        from train_real_simple import SimpleRealTrainer
        
        # Create small trainer
        trainer = SimpleRealTrainer(device='cpu')
        trainer.create_models(vocab_size=1000, element_types=22, d_model=128)
        trainer.setup_training(learning_rate=1e-3)
        
        print("✅ Trainer created")
        
        # Create dummy batch
        batch = {
            'images': torch.randn(1, 3, 720, 1280),
            'html_tokens': torch.randint(0, 1000, (1, 64)),
            'html_attention_mask': torch.ones(1, 64),
            'layout_targets': torch.randint(0, 1000, (1, 32)),
            'element_labels': torch.rand(1, 22)
        }
        
        # Test training step
        start_time = time.time()
        trainer.vision_encoder.train()
        trainer.html_encoder.train()
        trainer.fusion_model.train()
        trainer.output_model.train()
        
        # Forward pass
        visual_features = trainer.vision_encoder(batch['images'])
        html_features = trainer.html_encoder(batch['html_tokens'], batch['html_attention_mask'])
        fused_features = trainer.fusion_model(visual_features, html_features)
        outputs = trainer.output_model(fused_features, batch['layout_targets'][:, :-1], inference_mode='dual')
        
        # Loss computation
        targets = {
            'detail_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, 999),
            'semantic_targets': torch.clamp(batch['layout_targets'][:, 1:], 0, 49),
            'element_targets': batch['element_labels']
        }
        losses = trainer.loss_function(outputs, targets)
        
        # Backward pass
        trainer.optimizer.zero_grad()
        losses['total_loss'].backward()
        trainer.optimizer.step()
        
        step_time = time.time() - start_time
        print(f"✅ Training step: {step_time:.4f}s")
        print(f"✅ Loss: {losses['total_loss'].item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🍎 macOS CPU TRAINING TEST")
    print("=" * 60)
    
    tests = [
        ("System Check", test_system),
        ("Data Loading", test_data_loading),
        ("Model Creation", test_model_creation),
        ("Training Step", test_training_step)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(tests)}")
    
    if passed == len(tests):
        print("\n🎉 ALL TESTS PASSED!")
        print("Ready to run training:")
        print("cd dst/training")
        print("python3 train_real_simple.py --batch_size 1 --num_epochs 3")
    else:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 