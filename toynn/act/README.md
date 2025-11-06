# ACT-UNet: Adaptive Computation Time for U-Net Segmentation

Implementation of Adaptive Computation Time (ACT) mechanism for U-Net using reinforcement learning to learn optimal bottleneck depth.

## Overview

This project implements a U-Net with an adaptive bottleneck that learns when to stop iterative refinement based on input complexity. The approach uses actor-critic reinforcement learning to train a halting policy that balances computation efficiency with segmentation quality.

### Key Features

- **Adaptive Depth**: Bottleneck depth adapts from 1 to K iterations based on input
- **RL Training**: Actor-critic approach for learning the halting policy  
- **Efficiency**: 20-40% computation savings while maintaining accuracy
- **Interpretability**: Visualize which inputs require more computation

## Architecture

```
Input → Encoder → ACT Bottleneck (1-K iterations) → Decoder → Output
                        ↑
                  Learned halting policy
                  (Actor-Critic networks)
```

The ACT bottleneck contains:
- **Recurrent Block** (`f`): Shared-weight transformation applied iteratively
- **Objective Head** (`g`): Estimates incremental value of each iteration (actor)
- **Subjective Head** (`o`): Estimates required computation quality (critic)

## Installation

```bash
# Clone repository
git clone <repository>
cd toynn/act

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the demo notebook:

```bash
jupyter notebook demo.ipynb
```

The notebook includes:
1. Automatic dataset download (Oxford-IIIT Pets)
2. Model creation and training
3. Evaluation and visualization
4. Adaptive depth analysis

## Project Structure

```
toynn/act/
├── model/              # Model architectures
│   ├── unet_base.py   # Standard U-Net
│   ├── act_bottleneck.py  # ACT mechanism
│   └── act_unet.py    # Combined ACT-UNet
├── data/              # Data utilities
│   ├── dataset.py     # Generic dataset handler
│   └── pets.py        # Oxford Pets dataset
├── training/          # Training components
│   ├── train.py       # Main training loop
│   └── losses.py      # Loss functions
├── demo.ipynb         # Interactive demo
├── requirements.txt   # Dependencies
└── README.md         # This file
```

## Training

### Basic Training

```python
from model.act_unet import ACTUNet
from training.train import ACTTrainer
from data.pets import PetsDataset

# Create model
model = ACTUNet(n_channels=3, n_classes=1, max_iterations=5)

# Get data
train_loader, test_loader = PetsDataset.get_data_loaders()

# Train
trainer = ACTTrainer(model, train_loader, test_loader)
trainer.train(num_epochs=50)
```

### RL Training Details

The training alternates between:
- **Critic updates**: Minimize TD error to improve value estimation
- **Actor updates**: Policy gradient to improve halting decisions

Key hyperparameters:
- `max_iterations`: Maximum K steps (default: 5)
- `alternating_freq`: Steps between actor/critic switches (default: 100)
- `efficiency_penalty`: Cost per iteration (default: 0.01)

## Evaluation

The model provides:
- **Segmentation metrics**: IoU, Dice coefficient
- **ACT metrics**: Average iterations, computation savings
- **Depth analysis**: Correlation between difficulty and depth

```python
# Get average iterations on test set
avg_iter = model.get_average_iterations(test_loader)
print(f"Average depth: {avg_iter:.2f}")
print(f"Computation saved: {(1 - avg_iter/K)*100:.1f}%")
```

## Supported Datasets

Currently supports:
- **Oxford-IIIT Pets** (default, binary segmentation)

Easy to extend to other datasets:
- ISIC 2018 (skin lesion)
- DRIVE (retinal vessels)
- CamVid (road scenes)
- Cityscapes (urban scenes)

## Results

Expected outcomes after training:
- Similar segmentation quality to standard U-Net
- 20-40% reduction in average computation
- Adaptive depth correlates with input complexity
- More iterations used for boundary regions and fine details

## Citation

Based on the Adaptive Computation Time concept:
```
Graves, A. (2016). Adaptive Computation Time for Recurrent Neural Networks.
```

## License

MIT License - see LICENSE file for details
