#!/usr/bin/env python3
"""
Test Script for Real Data Implementation
Validates the real dataset and training pipeline with actual data.
"""

import sys
import torch
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_dataset():
    """Test the real dataset implementation"""
    print("=" * 60)
    print("TESTING REAL DATASET IMPLEMENTATION")
    print("=" * 60)
    
    try:
        # Import real dataset
        sys.path.append('dst/data')
        from real_dataset import RealDataset, real_collate_fn
        from torch.utils.data import DataLoader
        
        # Test dataset creation
        data_dir = "data"
        dataset = RealDataset(data_dir)
        
        print(f"✓ Dataset created successfully")
        print(f"  - Number of samples: {len(dataset)}")
        print(f"  - Vocabulary size: {len(dataset.tokenizer.token_to_id)}")
        print(f"  - Element types: {len(dataset.tokenizer.element_types)}")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"\nSample structure:")
            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"  - {key}: {type(value)} ({value if isinstance(value, str) else 'data'})")
            
            # Test data loader
            loader = DataLoader(dataset, batch_size=2, collate_fn=real_collate_fn)
            batch = next(iter(loader))
            
            print(f"\nBatch structure:")
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {value.shape}")
                elif isinstance(value, list):
                    print(f"  - {key}: list of {len(value)} items")
            
            print(f"\n✓ Data loading successful")
            return True
        else:
            print("⚠ No data samples found")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_integration():
    """Test model integration with real data"""
    print("\n" + "=" * 60)
    print("TESTING MODEL INTEGRATION")
    print("=" * 60)
    
    try:
        # Import models
        sys.path.append('dst/models')
        from output_generation import OutputGenerationModule, MultiTaskLossFunction
        from multimodal_fusion import MultimodalLayoutFusion
        
        # Import dataset
        sys.path.append('dst/data')
        from real_dataset import RealDataset, real_collate_fn
        from torch.utils.data import DataLoader
        
        # Create dataset
        dataset = RealDataset("data", max_html_length=128, max_layout_length=64)
        
        if len(dataset) == 0:
            print("⚠ No data available for testing")
            return False
        
        # Get vocabulary size
        vocab_size = len(dataset.tokenizer.token_to_id)
        element_types = len(dataset.tokenizer.element_types)
        
        # Create models with smaller dimensions for testing
        multimodal_fusion = MultimodalLayoutFusion()
        
        model = OutputGenerationModule(
            d_model=256,  # Smaller for testing
            num_decoder_layers=2,
            num_heads=4,
            vocab_size=vocab_size,
            num_semantic_classes=50,
            num_element_types=element_types,
            window_size=64,
            top_k=16,
            dropout=0.1
        )
        
        loss_fn = MultiTaskLossFunction()
        
        print(f"✓ Models created successfully")
        print(f"  - Vocabulary size: {vocab_size}")
        print(f"  - Element types: {element_types}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        loader = DataLoader(dataset, batch_size=1, collate_fn=real_collate_fn)
        batch = next(iter(loader))
        
        # Forward pass
        fusion_outputs = multimodal_fusion(
            screenshot=batch['images'],
            html_structure=batch['html_tokens'],
            element_hints=None
        )
        
        outputs = model(
            fused_features=fusion_outputs['visual_features'],
            target_sequence=batch['layout_targets'][:, :-1],
            inference_mode='dual'
        )
        
        # Test loss computation
        targets = {
            'detail_targets': batch['layout_targets'][:, 1:],
            'semantic_targets': batch['layout_targets'][:, 1:],
            'element_targets': batch['element_labels']
        }
        
        losses = loss_fn(outputs, targets)
        
        print(f"✓ Forward pass successful")
        print(f"  - Total loss: {losses['total_loss'].item():.4f}")
        print(f"  - Detail logits shape: {outputs['detail_logits'].shape}")
        print(f"  - Element logits shape: {outputs['element_logits'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def prepare_training_command():
    """Generate training command for user"""
    print("\n" + "=" * 60)
    print("TRAINING COMMANDS")
    print("=" * 60)
    
    # Check data availability
    data_dir = Path("data")
    example_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('example_')]
    num_samples = len(example_dirs)
    
    print(f"Data directory: {data_dir}")
    print(f"Available samples: {num_samples}")
    
    if num_samples > 0:
        print(f"\n🚀 READY TO TRAIN!")
        print(f"\nQuick test command (small training):")
        print(f"cd dst/training")
        print(f"python3 train_real_data.py --data_dir ../../data --batch_size 2 --num_epochs 5 --learning_rate 1e-4")
        
        print(f"\nFull training command:")
        print(f"python3 train_real_data.py --data_dir ../../data --batch_size 4 --num_epochs 50 --learning_rate 5e-5")
        
        print(f"\nMonitoring:")
        print(f"tail -f real_training.log")
        
    else:
        print(f"\n⚠ Need more data samples!")
        print(f"Currently found: {num_samples} samples")
        print(f"Expected: 200 samples")
        print(f"\nEach sample should be in a directory like:")
        print(f"  data/example_001/")
        print(f"  ├── data.json")
        print(f"  └── screenshot.png")

def create_additional_examples():
    """Create placeholder examples to demonstrate multiple data samples"""
    print("\n" + "=" * 60)
    print("CREATING DEMO DATA SAMPLES")
    print("=" * 60)
    
    try:
        import json
        import shutil
        from PIL import Image
        
        data_dir = Path("data")
        
        # Check if example_001 exists
        example_001 = data_dir / "example_001"
        if not example_001.exists():
            print("⚠ example_001 not found, cannot create demo samples")
            return
        
        # Create a few more example directories for demo
        for i in range(2, 6):  # Create example_002 to example_005
            new_example = data_dir / f"example_{i:03d}"
            
            if not new_example.exists():
                new_example.mkdir(parents=True, exist_ok=True)
                
                # Copy and modify the data.json
                with open(example_001 / "data.json", 'r') as f:
                    data = json.load(f)
                
                # Modify data slightly for variety
                data['category'] = ['hero', 'content', 'footer', 'gallery'][i % 4]
                data['screenshot'] = str(new_example / "screenshot.png")
                
                # Save modified data
                with open(new_example / "data.json", 'w') as f:
                    json.dump(data, f, indent=2)
                
                # Copy screenshot
                shutil.copy2(example_001 / "screenshot.png", new_example / "screenshot.png")
                
                print(f"✓ Created {new_example.name}")
        
        print(f"\n✓ Demo data samples created")
        print(f"You now have samples for testing the training pipeline")
        
    except Exception as e:
        print(f"✗ Error creating demo samples: {e}")

def main():
    """Run all tests"""
    print("🔍 REAL DATA VALIDATION")
    
    # Test 1: Dataset implementation
    dataset_ok = test_real_dataset()
    
    # Test 2: Model integration
    model_ok = test_model_integration() if dataset_ok else False
    
    # Create demo samples if needed
    data_dir = Path("data")
    example_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.startswith('example_')]
    if len(example_dirs) < 5:
        create_additional_examples()
    
    # Show training commands
    prepare_training_command()
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    if dataset_ok and model_ok:
        print("🎉 All tests passed! Ready to train with real data.")
        print("\nNext steps:")
        print("1. Add your 200 data samples to the data/ directory")
        print("2. Run the training command shown above")
        print("3. Monitor training progress with the log file")
    else:
        print("❌ Some tests failed. Check the errors above.")
        
    return dataset_ok and model_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 