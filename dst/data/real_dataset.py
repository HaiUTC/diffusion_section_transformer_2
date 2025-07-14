#!/usr/bin/env python3
"""
Real Dataset Implementation for Layout Generation
Loads actual screenshot images and layout data from the data directory.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import re

logger = logging.getLogger(__name__)

class LayoutTokenizer:
    """Tokenizer for layout JSON strings and HTML structures"""
    
    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size
        self.token_to_id = {}
        self.id_to_token = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[UNK]': 1,
            '[CLS]': 2,
            '[SEP]': 3,
            '[MASK]': 4,
            '[START]': 5,
            '[END]': 6
        }
        
        # Initialize special tokens
        for token, idx in self.special_tokens.items():
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        self.next_id = len(self.special_tokens)
        
        # CSS attribute patterns for tokenization
        self.css_patterns = [
            r'@[^@]+',  # CSS classes with @
            r'[a-zA-Z_][a-zA-Z0-9_]*',  # Identifiers
            r'\d+(?:\.\d+)?[rpxem%]*',  # Numbers with units
            r'[{}()\[\]:,.]',  # Structural characters
            r'#[0-9a-fA-F]+',  # Color codes
            r'[<>=/]+'  # Operators
        ]
        
        # Element type mapping
        self.element_types = [
            'section', 'grid', 'column', 'wrapper', 'freedom', 'heading', 'paragraph', 'button', 'icon', 'image', 'video', 'list', 'map', 'counter', 'divider', 'qr', 'carousel', 'accordion', 'tab', 'gallery', 'masonry', 'social'
        ]
        
        # Build element type to ID mapping
        self.element_to_id = {elem: i for i, elem in enumerate(self.element_types)}
        self.id_to_element = {i: elem for i, elem in enumerate(self.element_types)}
    
    def tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using regex patterns"""
        tokens = []
        for pattern in self.css_patterns:
            matches = re.findall(pattern, text)
            tokens.extend(matches)
        return tokens
    
    def add_tokens(self, tokens: List[str]):
        """Add new tokens to vocabulary"""
        for token in tokens:
            if token not in self.token_to_id and self.next_id < self.vocab_size:
                self.token_to_id[token] = self.next_id
                self.id_to_token[self.next_id] = token
                self.next_id += 1
    
    def encode(self, text: str, max_length: int = 512) -> List[int]:
        """Encode text to token IDs"""
        tokens = self.tokenize_text(text)
        
        # Add tokens to vocabulary if needed
        self.add_tokens(tokens)
        
        # Convert to IDs
        token_ids = [self.special_tokens['[START]']]
        for token in tokens[:max_length-2]:  # Leave space for START and END
            token_id = self.token_to_id.get(token, self.special_tokens['[UNK]'])
            token_ids.append(token_id)
        token_ids.append(self.special_tokens['[END]'])
        
        # Pad to max_length
        while len(token_ids) < max_length:
            token_ids.append(self.special_tokens['[PAD]'])
        
        return token_ids[:max_length]
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                token = self.id_to_token[token_id]
                if token not in ['[PAD]', '[START]', '[END]']:
                    tokens.append(token)
        return ' '.join(tokens)
    
    def encode_elements(self, elements: List[str]) -> torch.Tensor:
        """Encode element list to multi-hot vector"""
        element_vector = torch.zeros(len(self.element_types))
        for element in elements:
            if element in self.element_to_id:
                element_vector[self.element_to_id[element]] = 1.0
        return element_vector

class RealDataset(Dataset):
    """Dataset class for real layout generation data"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (720, 1280),
                 max_html_length: int = 512,
                 max_layout_length: int = 256,
                 vocab_size: int = 50000,
                 transform: Optional[transforms.Compose] = None):
        
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.max_html_length = max_html_length
        self.max_layout_length = max_layout_length
        self.vocab_size = vocab_size
        
        # Initialize tokenizer
        self.tokenizer = LayoutTokenizer(vocab_size)
        
        # Image transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]   # ImageNet stds
                )
            ])
        else:
            self.transform = transform
        
        # Load data samples
        self.samples = self._load_samples()
        
        # Build vocabulary from all samples
        self._build_vocabulary()
        
        logger.info(f"Loaded {len(self.samples)} samples for {split}")
        logger.info(f"Vocabulary size: {len(self.tokenizer.token_to_id)}")
    
    def _load_samples(self) -> List[Dict]:
        """Load all data samples from directory"""
        samples = []
        
        # Find all directories that contain data.json and a PNG file
        all_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        logger.info(f"Found {len(all_dirs)} directories to check")
        
        for example_dir in sorted(all_dirs):
            data_file = example_dir / 'data.json'
            
            # Find any PNG file in the directory
            png_files = list(example_dir.glob('*.png'))
            screenshot_file = png_files[0] if png_files else None
            
            if data_file.exists() and screenshot_file:
                try:
                    with open(data_file, 'r') as f:
                        data = json.load(f)
                    
                    # Validate required fields
                    required_fields = ['html', 'section_layout', 'elements_used', 'category']
                    if all(field in data for field in required_fields):
                        sample = {
                            'data_file': str(data_file),
                            'screenshot_file': str(screenshot_file),
                            'html': data['html'],
                            'section_layout': data['section_layout'],
                            'used_elements': data['elements_used'],  # Map from elements_used to used_elements
                            'category': data['category'],
                            'sample_id': example_dir.name
                        }
                        samples.append(sample)
                    else:
                        logger.warning(f"Missing required fields in {data_file}")
                        
                except Exception as e:
                    logger.error(f"Error loading {data_file}: {e}")
            else:
                if not data_file.exists():
                    logger.debug(f"No data.json in {example_dir}")
                if not screenshot_file:
                    logger.debug(f"No PNG file in {example_dir}")
        
        logger.info(f"Successfully loaded {len(samples)} valid samples")
        return samples
    
    def _build_vocabulary(self):
        """Build vocabulary from all samples"""
        logger.info("Building vocabulary from dataset...")
        
        all_html_tokens = []
        all_layout_tokens = []
        
        for sample in self.samples:
            # Tokenize HTML
            html_tokens = self.tokenizer.tokenize_text(sample['html'])
            all_html_tokens.extend(html_tokens)
            
            # Tokenize section layout
            layout_tokens = self.tokenizer.tokenize_text(sample['section_layout'])
            all_layout_tokens.extend(layout_tokens)
        
        # Add all tokens to vocabulary
        unique_tokens = set(all_html_tokens + all_layout_tokens)
        self.tokenizer.add_tokens(list(unique_tokens))
        
        logger.info(f"Built vocabulary with {len(self.tokenizer.token_to_id)} tokens")
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """Load and process image"""
        try:
            image = Image.open(image_path).convert('RGB')
            return self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return dummy image if loading fails
            return torch.zeros(3, *self.image_size)
    
    def _create_attention_mask(self, token_ids: List[int]) -> List[int]:
        """Create attention mask for token sequence"""
        mask = []
        for token_id in token_ids:
            if token_id == self.tokenizer.special_tokens['[PAD]']:
                mask.append(0)
            else:
                mask.append(1)
        return mask
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load and process image
        image = self._load_image(sample['screenshot_file'])
        
        # Tokenize HTML structure
        html_token_ids = self.tokenizer.encode(sample['html'], self.max_html_length)
        html_attention_mask = self._create_attention_mask(html_token_ids)
        
        # Tokenize section layout (target)
        layout_token_ids = self.tokenizer.encode(sample['section_layout'], self.max_layout_length)
        layout_attention_mask = self._create_attention_mask(layout_token_ids)
        
        # Encode used elements
        element_labels = self.tokenizer.encode_elements(sample['used_elements'])
        
        # Category encoding (simple label encoding for now)
        categories = ['hero', 'content', 'footer', 'navigation', 'sidebar', 'gallery', 'form']
        category_id = categories.index(sample['category']) if sample['category'] in categories else 0
        
        return {
            'image': image,
            'html_tokens': torch.tensor(html_token_ids, dtype=torch.long),
            'html_attention_mask': torch.tensor(html_attention_mask, dtype=torch.long),
            'layout_targets': torch.tensor(layout_token_ids, dtype=torch.long),
            'layout_attention_mask': torch.tensor(layout_attention_mask, dtype=torch.long),
            'element_labels': element_labels,
            'category_id': torch.tensor(category_id, dtype=torch.long),
            'sample_id': sample['sample_id'],
            'raw_html': sample['html'],
            'raw_layout': sample['section_layout']
        }

def create_real_dataset_splits(data_dir: str, 
                              train_ratio: float = 0.8,
                              val_ratio: float = 0.1,
                              test_ratio: float = 0.1,
                              **dataset_kwargs) -> Tuple[RealDataset, RealDataset, RealDataset]:
    """Create train/val/test splits from real data"""
    
    # Get all available samples
    temp_dataset = RealDataset(data_dir, split='all', **dataset_kwargs)
    all_samples = temp_dataset.samples
    
    # Shuffle samples
    np.random.seed(42)  # For reproducible splits
    indices = np.random.permutation(len(all_samples))
    
    # Calculate split sizes
    n_train = int(len(all_samples) * train_ratio)
    n_val = int(len(all_samples) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create split datasets
    train_samples = [all_samples[i] for i in train_indices]
    val_samples = [all_samples[i] for i in val_indices]
    test_samples = [all_samples[i] for i in test_indices]
    
    # Create datasets with shared vocabulary
    train_dataset = RealDataset(data_dir, split='train', **dataset_kwargs)
    train_dataset.samples = train_samples
    
    val_dataset = RealDataset(data_dir, split='val', **dataset_kwargs)
    val_dataset.samples = val_samples
    val_dataset.tokenizer = train_dataset.tokenizer  # Share vocabulary
    
    test_dataset = RealDataset(data_dir, split='test', **dataset_kwargs)
    test_dataset.samples = test_samples
    test_dataset.tokenizer = train_dataset.tokenizer  # Share vocabulary
    
    logger.info(f"Created splits: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}")
    
    return train_dataset, val_dataset, test_dataset

def real_collate_fn(batch: List[Dict]) -> Dict:
    """Custom collate function for real data batching"""
    batch_size = len(batch)
    
    # Stack images
    images = torch.stack([item['image'] for item in batch])
    
    # Stack HTML tokens and masks
    html_tokens = torch.stack([item['html_tokens'] for item in batch])
    html_attention_mask = torch.stack([item['html_attention_mask'] for item in batch])
    
    # Stack layout targets and masks
    layout_targets = torch.stack([item['layout_targets'] for item in batch])
    layout_attention_mask = torch.stack([item['layout_attention_mask'] for item in batch])
    
    # Stack element labels
    element_labels = torch.stack([item['element_labels'] for item in batch])
    
    # Stack category IDs
    category_ids = torch.stack([item['category_id'] for item in batch])
    
    # Collect sample IDs and raw data
    sample_ids = [item['sample_id'] for item in batch]
    raw_html = [item['raw_html'] for item in batch]
    raw_layout = [item['raw_layout'] for item in batch]
    
    return {
        'images': images,
        'html_tokens': html_tokens,
        'html_attention_mask': html_attention_mask,
        'layout_targets': layout_targets,
        'layout_attention_mask': layout_attention_mask,
        'element_labels': element_labels,
        'category_ids': category_ids,
        'sample_ids': sample_ids,
        'raw_html': raw_html,
        'raw_layout': raw_layout
    }

if __name__ == "__main__":
    # Test the dataset implementation
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Test dataset creation
    data_dir = "../../data"  # Adjust path as needed
    
    try:
        dataset = RealDataset(data_dir)
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        if len(dataset) > 0:
            # Test first sample
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"Image shape: {sample['image'].shape}")
            print(f"HTML tokens shape: {sample['html_tokens'].shape}")
            print(f"Layout targets shape: {sample['layout_targets'].shape}")
            print(f"Element labels shape: {sample['element_labels'].shape}")
            print(f"Sample ID: {sample['sample_id']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc() 