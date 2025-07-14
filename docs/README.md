# DST - Diffusion Section Transformer

Multimodal generative AI model for transforming website screenshots and HTML structures into semantic layout descriptions.

## Project Structure

```
dst/                          # Main implementation package
├── data_validation/          # Task 1.1: Dataset expansion & annotation
├── image_processing/         # Task 1.2: Advanced image preprocessing
└── utils/                    # Shared utilities

configs/                      # Configuration files
docs/                         # Documentation
tests/                        # Test files
data/                         # Dataset storage
```

## Task Overview

### Task 1.1: Data Validation

- Dataset expansion and annotation
- HTML structure processing
- XOJL semantic layout validation
- Quality control and error detection

### Task 1.2: Image Processing

- Desktop-optimized preprocessing (1280×720)
- Dataset-specific normalization
- ViT patchification (16×16 patches)
- Content-aware scaling and enhancement

## Usage

The project is organized into focused, independent modules that can be used separately or together for the complete pipeline.

Each task module contains its own processors, validators, and batch processing utilities.
