# Simple ACT Test Implementation

This folder contains a simplified implementation of Adaptive Computation Time (ACT) for testing on MNIST before integrating into complex architectures like UNet.

## Structure

```
act_simple/
├── model/
│   ├── simple_act.py    # Simple ACT model with recurrent refinement
│   └── __init__.py
├── training/
│   ├── train.py         # Simple training loop
│   └── __init__.py
├── demo_simple.ipynb    # Demo notebook
└── README.md
```

## Key Simplifications vs Full ACT-UNet

1. **Architecture**: Simple feedforward + recurrent instead of UNet
2. **Task**: MNIST classification instead of segmentation
3. **Size**: ~200K params instead of 26M params
4. **Training**: Single optimizer instead of complex alternating
5. **Loss**: Simple CrossEntropy instead of Dice+BCE

## Theory Implementation

Based on `act.md`, this implements:
- **Subjective contribution** τ: How much each iteration contributes
- **Objective threshold** o(x): When is computation sufficient
- **Halting criterion**: Stop when Στᵢ > o(x^n)
- **RL training**: TD error for critic, policy gradient for actor

## Key Fixes for Stability

1. **Bounded activations**: Sigmoid instead of Softplus (prevents explosion)
2. **Scaled outputs**: τ ∈ [0, 0.3], o ∈ [0.1, 0.6]
3. **Conservative init**: Small weights to prevent early explosions
4. **Gradient clipping**: Prevent gradient explosions

## Usage

```python
from model import SimpleACTNet
from training import SimpleACTTrainer, get_mnist_loaders

# Create model
model = SimpleACTNet(max_iterations=5)

# Get data
train_loader, test_loader = get_mnist_loaders()

# Train
trainer = SimpleACTTrainer(model)
trainer.train(train_loader, test_loader, num_epochs=5)
```

## Expected Results

- **Accuracy**: ~98% on MNIST
- **Computation saved**: 20-40% vs always using max iterations
- **Digit patterns**: Simple digits (1,7) use fewer iterations than complex (8,9)
- **Stable training**: No gradient explosions with bounded ACT

## Lessons for UNet Integration

1. **Bounded ACT outputs are critical** - Prevents gradient explosion
2. **Progressive ACT weight** - Start small (0.001) and increase
3. **Separate gradient clipping** - ACT components need stricter clipping
4. **Warmup period** - Don't use ACT loss for first N steps

Once this simple version works reliably, the same principles can be applied to the full ACT-UNet!
