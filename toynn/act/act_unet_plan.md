# Adaptive Computation Time (ACT) for U-Net Bottleneck

## Project Overview
Apply ACT mechanism to U-Net's bottleneck layer to enable adaptive computational depth, allowing the model to learn how many refinement iterations are needed based on input complexity.

## Architecture Design

### Base U-Net with ACT Bottleneck

```
Input Image
    ↓
[Encoder Path]
    ↓
Bottleneck with ACT ← Adaptive depth (1 to K iterations)
    ↓
[Decoder Path]
    ↓
Segmentation Map
```

### ACT-Enhanced Bottleneck Architecture

```python
class ACTBottleneck:
    def __init__(self):
        # Recurrent transformation module
        self.f = RecurrentBlock()  # Shared weights across iterations
        
        # Value functions for halting policy
        self.objective_head = nn.Sequential(...)  # g network: estimates τ_i
        self.subjective_head = nn.Sequential(...)  # o network: estimates threshold
        
        # Maximum iterations
        self.max_iterations = K
```

### Key Components

1. **Recurrent Processing Block** (`f`)
   - Shared-weight transformer or convolutional block
   - Applied iteratively: `x^{n+1} = f(x^{n})`
   - Maintains spatial dimensions for skip connections

2. **Objective Contribution Network** (`g`)
   - Estimates incremental value τ_i at each step
   - Input: difference between consecutive states `x^{i} - x^{i-1}`
   - Output: positive scalar contribution

3. **Subjective Threshold Network** (`o`)
   - Estimates required computation quality
   - Input: current state `x^{i}`
   - Output: dynamic stopping threshold

4. **Halting Mechanism**
   - Stop when: `Σ τ_i > o(x^{n})`
   - Forward pass uses learned policy
   - Training explores up to K iterations

## Implementation Strategy

### Phase 1: Baseline U-Net Implementation
- [ ] Standard U-Net for chosen dataset
- [ ] Establish baseline performance metrics
- [ ] Profile computational requirements

### Phase 2: ACT Integration
- [ ] Replace standard bottleneck with ACT module
- [ ] Implement recurrent processing block
- [ ] Add objective/subjective heads
- [ ] Implement differentiable halting mechanism

### Phase 3: Training Pipeline
- [ ] Exploration strategy (rollout to K steps)
- [ ] Optimal depth discovery via task loss
- [ ] Actor-critic updates for value functions
- [ ] Alternating optimization schedule

### Phase 4: Evaluation
- [ ] Adaptive depth analysis
- [ ] Performance vs. computation trade-offs
- [ ] Visualization of stopping decisions
- [ ] Ablation studies

## Accessible Segmentation Datasets

### Medical Imaging (2D)

#### 1. **ISIC 2018 (Skin Lesion)**
- **Size**: ~2,600 images
- **Task**: Melanoma segmentation
- **Format**: RGB images + binary masks
- **Access**: Free with registration
- **Pros**: Clear task, good for initial testing
```python
# Download: https://challenge.isic-archive.com/data/
```

#### 2. **DRIVE (Retinal Vessels)**
- **Size**: 40 images (20 train, 20 test)
- **Task**: Blood vessel segmentation
- **Format**: RGB fundus images + manual annotations
- **Access**: Free with registration
- **Pros**: Small dataset, quick iteration
```python
# Download: https://drive.grand-challenge.org/
```

#### 3. **Montgomery County Chest X-ray**
- **Size**: 138 images
- **Task**: Lung segmentation
- **Format**: X-ray images + lung masks
- **Access**: Public domain
- **Pros**: Simple binary segmentation

### Natural Images

#### 4. **Oxford-IIIT Pet Dataset**
- **Size**: 7,349 images
- **Task**: Pet segmentation (37 categories)
- **Format**: RGB images + trimap annotations
- **Access**: Public, freely available
- **Pros**: Good balance of size and complexity
```python
import torchvision.datasets as datasets
dataset = datasets.OxfordIIITPet(root='./data', download=True)
```

#### 5. **CamVid (Road Scene)**
- **Size**: 701 images
- **Task**: Semantic segmentation (32 classes)
- **Format**: RGB video frames + pixel labels
- **Access**: Public
- **Pros**: Real-world application, manageable size
```python
# Available through: https://github.com/alexgkendall/SegNet-Tutorial
```

#### 6. **Cityscapes (Urban Scenes)**
- **Size**: 5,000 fine annotations, 20,000 coarse
- **Task**: Urban scene understanding (30 classes)
- **Format**: High-res images + fine/coarse annotations
- **Access**: Free for research (registration required)
- **Pros**: Industry standard, extensive benchmarks
```python
# Download: https://www.cityscapes-dataset.com/
```

### Recommended Starting Dataset

**Oxford-IIIT Pet Dataset** - Best for initial experiments:
- Medium size (not too small, not too large)
- Simple binary/ternary segmentation
- Well-established baselines
- Easy to load with PyTorch
- Clear visual interpretability

## Training Configuration

### Loss Functions

```python
# Combined loss for ACT-UNet
L_total = λ_task * L_segmentation + λ_act * L_act

where:
- L_segmentation: Dice + BCE loss
- L_act: RL-based halting loss
```

### Hyperparameters

```yaml
# ACT-specific
max_iterations: 5-10  # K value
exploration_prob: 0.1  # ε-greedy exploration
actor_lr: 1e-4
critic_lr: 1e-3
alternating_freq: 100  # steps between actor/critic switches

# Standard U-Net
base_channels: 64
depth: 4  # encoder/decoder depth
batch_size: 8-16
learning_rate: 1e-3
```

## Evaluation Metrics

### Performance Metrics
- **IoU (Intersection over Union)**
- **Dice Coefficient**
- **Pixel Accuracy**
- **Boundary F1-Score**

### ACT-Specific Metrics
- **Average Stopping Depth**: Mean iterations before halting
- **Depth Variance**: Consistency of stopping decisions
- **Computation Savings**: % reduction vs. fixed K iterations
- **Depth-Complexity Correlation**: How depth correlates with input difficulty

## Experimental Questions

1. **Depth Adaptation**: Does the model use more iterations for difficult regions (boundaries, fine details)?
2. **Generalization**: Can the learned halting policy transfer to different image complexities?
3. **Efficiency**: What's the speed/accuracy trade-off compared to fixed-depth U-Net?
4. **Interpretability**: Can we visualize what triggers deeper computation?

## Code Structure

```
act_unet/
├── models/
│   ├── unet_base.py       # Standard U-Net
│   ├── act_bottleneck.py  # ACT mechanism
│   └── act_unet.py        # Combined model
├── datasets/
│   ├── pets.py
│   └── transforms.py
├── training/
│   ├── train_act.py       # RL training loop
│   ├── exploration.py     # K-step rollout
│   └── losses.py
├── evaluation/
│   ├── metrics.py
│   └── visualize.py
└── configs/
    └── act_unet_pets.yaml
```

## Timeline

### Week 1: Setup & Baseline
- Day 1-2: Dataset preparation, data loaders
- Day 3-4: Baseline U-Net implementation
- Day 5-7: Baseline training and evaluation

### Week 2: ACT Integration
- Day 1-3: ACT bottleneck module
- Day 4-5: Halting mechanism
- Day 6-7: Initial training pipeline

### Week 3: RL Training
- Day 1-2: Exploration strategy
- Day 3-4: Actor-critic updates
- Day 5-7: Full training loop

### Week 4: Evaluation & Analysis
- Day 1-3: Comprehensive evaluation
- Day 4-5: Ablation studies
- Day 6-7: Visualization and interpretation

## Expected Outcomes

1. **Adaptive Computation**: Model learns to spend more iterations on challenging image regions
2. **Improved Efficiency**: 20-40% reduction in average computation vs. always using K iterations
3. **Maintained Accuracy**: Similar or better segmentation quality with adaptive depth
4. **Interpretable Decisions**: Clear correlation between image complexity and computational depth

## Potential Challenges

1. **Training Stability**: Balancing RL and supervised objectives
2. **Skip Connections**: Maintaining feature alignment with variable bottleneck depth
3. **Gradient Flow**: Ensuring gradients propagate through variable-length computation
4. **Exploration**: Sufficient exploration of different depths during training

## Next Steps

1. Choose initial dataset (recommend Oxford Pets)
2. Implement baseline U-Net
3. Design ACT bottleneck module
4. Set up RL training infrastructure
5. Run initial experiments
6. Iterate based on results
