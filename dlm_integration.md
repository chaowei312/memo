# Diffusion Language Model with Memory

## Overview

This document outlines a minimal Diffusion Language Model (DLM) that integrates the Hierarchical Reasoning Model (HRM) architecture with long-term memory mechanisms. The model uses **masked token diffusion** (not continuous noise) following the discrete DLM approach.

Key components:
- **Masked token diffusion** for iterative text generation/refinement
- **H/L module hierarchy** for multi-scale reasoning  
- **Memory cells** with cross-attention bottleneck
- **Adaptive computation** via ACT mechanism

## Core Architecture

### 1. Masked Token Diffusion Process with HRM

The diffusion process operates over **discrete tokens** using masking/unmasking, with the HRM's L-module for token prediction and H-module for strategic guidance.

#### Forward Diffusion (Masking Process)
For text sequence $x_0$ with tokens $[w_1, w_2, ..., w_n]$:

At diffusion timestep $t$, mask tokens with probability $\gamma_t$:
$$x_t = \text{MaskTokens}(x_{t-1}, \gamma_t)$$

where $\gamma_t$ increases with $t$, progressively masking more tokens until reaching nearly full masking.

#### Reverse Diffusion (Unmasking/Denoising)
At each timestep $t$, predict and unmask tokens using HRM structure:
- **L-module**: Predicts masked token probabilities through iterative refinement
- **H-module**: Provides strategic context every T steps

### 2. L-Module (Fast Token Prediction)

The L-module performs rapid masked token prediction constrained by H-module guidance:

$$z_L^{(i)} = f_L(x_t, \text{mask}_t, z_H, \text{pos}_t; \theta_L)$$

where:
- $x_t$: Current partially masked sequence
- $\text{mask}_t$: Binary mask indicating masked positions
- $z_H$: High-level strategic state (fixed during L iterations)
- $\text{pos}_t$: Diffusion timestep encoding

The L-module iteratively refines token predictions using ACT to determine optimal computation depth.


### 3. H-Module with Memory System

The H-module maintains long-term memory across diffusion steps:

#### Memory Cells
- $\mathcal{M} = \{m_1, ..., m_N\}$: N specialized memory cells
- Each cell: $m_i = [\text{CLS}_i; \text{content}_i]$

#### Top-k Selection
Select k most relevant cells based on current state:
$$\mathcal{A} = \text{TopK}(\{\text{CLS}_i^\top W \cdot z_H\}_{i=1}^N, k)$$

### 4. Adaptive Computation Time for Token Prediction

Following the ACT framework adapted for discrete tokens:

**Objective Contribution** (per L-module iteration):
$$\tau_i = \frac{g(\text{logits}_{i+1} - \text{logits}_i)}{|\text{CLS}(z_L^i) \cdot \text{CLS}(z_L^{i+1})| + \epsilon}$$

where $\text{logits}_i$ are the token prediction logits at iteration $i$.

**Subjective Score**:
$$o(z_L) = \text{SwiGLU}(\text{LN}(z_L)) \rightarrow [0, 1]$$

**Termination**: Stop refining when $\sum_i \tau_i > o(z_L^{\text{current}})$

### 5. Token Unmasking Strategy

At each reverse diffusion step, we unmask tokens based on:
1. **Confidence-based selection**: Unmask tokens with highest prediction confidence
2. **Strategic importance**: H-module influences which tokens to prioritize
3. **Progressive revelation**: Gradually reduce masking rate following schedule $\gamma_{T-t}$

### 6. Training Objective

The model is trained with a composite loss:

$$\mathcal{L} = \mathcal{L}_{\text{MLM}} + \lambda_1 \mathcal{L}_{\text{memory}} + \lambda_2 \mathcal{L}_{\text{ACT}}$$

where:
- $\mathcal{L}_{\text{MLM}}$: Masked language modeling loss for predicting masked tokens
- $\mathcal{L}_{\text{memory}}$: Load balancing for memory cells to ensure diverse specialization
- $\mathcal{L}_{\text{ACT}}$: Alignment loss for subjective scorer (computed after k iterations)

## Key Design Principles

### Why Masked Token Diffusion (Not Continuous)

1. **Discrete Nature of Language**: Tokens are discrete entities; continuous noise doesn't preserve semantic meaning
2. **Controllable Corruption**: Masking provides interpretable corruption vs random Gaussian noise
3. **Bidirectional Context**: Unlike autoregressive models, can leverage full context for prediction
4. **Progressive Refinement**: Start with fully masked sequence, progressively reveal/refine tokens

### HRM Integration Benefits

1. **Temporal Hierarchy**: L-module refines predictions within each diffusion step; H-module maintains strategy
2. **Memory Persistence**: Long-term memory cells maintain context across entire generation process
3. **Adaptive Depth**: ACT prevents unnecessary iterations for simple token predictions
4. **Strategic Unmasking**: H-module can influence which tokens to prioritize during unmasking

