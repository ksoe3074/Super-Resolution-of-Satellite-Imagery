# Super-Resolution for Multispectral Satellite Imagery

This repository contains implementations of four deep learning models for super-resolution of multispectral satellite imagery using panchromatic guidance.

## Overview

This project implements and compares four super-resolution models:
- **PanNet**: Residual network with 7 residual blocks
- **SRCNN**: Shallow convolutional network with 3 layers
- **VDSR**: Very deep super-resolution network with 20 layers
- **SRGAN**: Generative adversarial network for super-resolution

All models are designed to upsample 6-band multispectral (MS) imagery from 48×48 to 96×96 pixels using panchromatic (PAN) band guidance.

## Repository Structure

```
super_resolution_project/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
│
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── pannet.py               # PanNet model
│   ├── srcnn.py                # SRCNN model
│   ├── vdsr.py                 # VDSR model
│   └── srgan.py                # SRGAN model
│
├── training/                    # Training modules
│   ├── __init__.py
│   ├── pannet_trainer.py       # PanNet trainer
│   ├── srcnn_trainer.py        # SRCNN trainer
│   ├── vdsr_trainer.py         # VDSR trainer
│   └── srgan_trainer.py         # SRGAN trainer
│
├── utils/                       # Utility modules
│   ├── __init__.py
│   ├── data_loader.py           # Dataset class
│   ├── metrics.py               # Evaluation metrics
│   └── visualization.py         # Visualization utilities
│
├── scripts/                      # Execution scripts
│   ├── run_all_models_imported.py  # Main execution script
│   ├── preprocessing.py         # Data preprocessing (optional)
│   └── simulate_pan.py          # PAN simulation (optional)
│
├── examples/                     # Example output images
│   ├── example_pannet.png
│   └── example_srgan.png
│
└── models/                       # Saved model checkpoints (empty in repo)
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd super_resolution_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Requirements

- Acquire Copernicus Sentinel-2 Level-2A tiles (bands B2-B7) and prepare them according to the pipeline described in the thesis.
- Downsample bands to 30m resolution (Landsat-style) and simulate a 15m panchromatic band if required.
- Organize the processed tiles in a structure compatible with `utils/data_loader.py` before running any training scripts.

## Usage

### Quick Test Mode (~40 minutes)

Run all models with minimal epochs for quick verification:

```bash
python scripts/run_all_models_imported.py --quick
```

This will:
- Train each model for a few epochs
- Evaluate on test set
- Generate comparison images
- Save results

### Full Training Mode (~20 hours)

Run all models with full training configurations:

```bash
python scripts/run_all_models_imported.py
```

This will:
- Train each model with optimal hyperparameters
- Run both PAN and no-PAN configurations
- Generate comprehensive results

### Individual Model Training

```python
from training.pannet_trainer import run_pannet
import torch

config = {
    'epochs': 40,
    'lr': 1e-4,
    'filters': 64,
    'batch_size': 16,
    'use_pan': True
}

result, trainer = run_pannet(
    data_root="processed_data",
    train_locations=["KANSAS", "PARIS", "PNG"],
    test_locations=["SYDNEY"],
    samples_per_location=6000,
    device=torch.device('cuda'),
    config=config
)
```

## Evaluation Metrics

The models are evaluated using four metrics:

- **PSNR** (Peak Signal-to-Noise Ratio): Higher is better
- **SSIM** (Structural Similarity Index): Range [0, 1], higher is better
- **ERGAS** (Erreur Relative Globale Adimensionnelle de Synthèse): Lower is better
- **SAM** (Spectral Angle Mapper): Lower is better (degrees)

## Model Details

### PanNet
- **Architecture:** 7 residual blocks with skip connection
- **Input:** 7 channels (6 MS + 1 PAN) or 6 channels (MS only)
- **Output:** 6 channels (MS)
- **Loss:** MSE
- **Optimizer:** AdamW

### SRCNN
- **Architecture:** 3 convolutional layers
- **Input:** 7 channels (6 MS + 1 PAN) or 6 channels (MS only)
- **Output:** 6 channels (MS)
- **Loss:** MSE
- **Optimizer:** Adam

### VDSR
- **Architecture:** 20 convolutional layers (very deep)
- **Input:** 7 channels (6 MS + 1 PAN) or 6 channels (MS only)
- **Output:** 6 channels (MS)
- **Loss:** MSE
- **Optimizer:** SGD with momentum

### SRGAN
- **Architecture:** Generator with residual blocks + Discriminator
- **Input:** 7 channels (6 MS + 1 PAN) or 6 channels (MS only)
- **Output:** 6 channels (MS)
- **Loss:** Content loss + Gradient loss + Adversarial loss
- **Optimizer:** Adam

## Configuration

Models can be configured with or without PAN band input using the `use_pan` parameter:

```python
config = {
    'use_pan': True,   # Use PAN band for guidance
    # or
    'use_pan': False,  # Use only MS bands
}
```

## Author: Kyle Soepono
[ksoe3074@uni.sydney.edu.au]

