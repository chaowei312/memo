# Hierarchical Reasoning Model (HRM) - Architecture Review

## Summary

The Hierarchical Reasoning Model (HRM) introduces a novel recurrent architecture that achieves exceptional reasoning capabilities through hierarchical, multi-timescale processing inspired by human cognition. With only 27M parameters, HRM solves complex reasoning tasks in a single forward pass without requiring Chain-of-Thought (CoT) supervision or pre-training.

### Architectural Components

#### **Attention Mechanism** (`models/layers.py`, lines 98-136)
- Custom implementation supporting both RoPE and learned positional encodings
- Non-causal attention for bidirectional reasoning
- 8 attention heads with dimension 64 each

#### **SwiGLU Activation** (`models/layers.py`, lines 139-159)
- Gated linear unit with Swish activation
- Expansion factor of 4 for hidden dimensions
- Superior gradient flow compared to ReLU

#### **RMS Normalization**
- Post-normalization after each sub-layer
- More stable training than LayerNorm
- Epsilon value: 1e-5

#### **Sparse Puzzle Embeddings** (`models/sparse_embedding.py`)
- Learnable embeddings for puzzle identities
- Zero-initialized to prevent overfitting
- Separate learning rate for fine-tuning

## Core Architectural Innovation

### 1. Two-Level Hierarchical Design

The HRM architecture (`HRM/models/hrm/hrm_act_v1.py`) fundamentally reimagines how neural networks approach reasoning through a dual-module system:

#### **H-Level (High-Level Module)**
- **Purpose**: Abstract planning and strategic reasoning
- **Characteristics**: 
  - Slower update cycles (2 total updates: 1 no-grad + 1 grad)
  - 4 transformer layers (`H_layers: 4`)
  - Maintains global context and high-level state
  - Hidden size: 512 dimensions

#### **L-Level (Low-Level Module)**  
- **Purpose**: Rapid, detailed computations and execution
- **Characteristics**:
  - More frequent updates (4 total updates: 3 no-grad + 1 grad)
  - 4 transformer layers (`L_layers: 4`)
  - Processes fine-grained details
  - Receives input injection from H-level

```python
# Actual hierarchical iteration pattern (lines 192-204)
with torch.no_grad():
    for _H_step in range(H_cycles):  # 2 cycles
        for _L_step in range(L_cycles):  # 2 cycles each
            # Skip the LAST L-level update (reserved for gradient computation)
            if not ((_H_step == H_cycles - 1) and (_L_step == L_cycles - 1)):
                z_L = L_level(z_L, z_H + input_embeddings)
        
        # Skip the LAST H-level update (reserved for gradient computation)
        if not (_H_step == H_cycles - 1):
            z_H = H_level(z_H, z_L)

# Final updates WITH gradients
z_L = L_level(z_L, z_H + input_embeddings)  # Final L-level with gradients
z_H = H_level(z_H, z_L)                     # Final H-level with gradients
```

#### **Input Injection Pattern**

**Input Embeddings Structure**:
```python
# input_embeddings composition (line 146-166)
input_embeddings = [puzzle_emb | token_emb] + position_emb
# where:
#   - puzzle_emb: Zero-initialized task identifier (e.g., Sudoku=0, Maze=1)
#   - token_emb: Actual problem tokens (input sequence)
#   - position_emb: RoPE or learned positional encodings
#   - All scaled by sqrt(hidden_size) for variance control
```

**L-Level (Low-Level Module) Injection**:
```python
# External call
z_L = L_level(z_L, z_H + input_embeddings)

# Inside L_level.forward():
def forward(self, hidden_states, input_injection):
    hidden_states = hidden_states + input_injection  # z_L + (z_H + input_embeddings)
    # Process through layers...
```

**L-level actual computation**: Processes the sum of ALL three components:
- `hidden = z_L + z_H + input_embeddings`
- `hidden = z_L + z_H + ([puzzle_emb | token_emb] + position_emb)`

**H-Level (High-Level Module) Injection**:
```python
# External call
z_H = H_level(z_H, z_L)

# Inside H_level.forward():
def forward(self, hidden_states, input_injection):
    hidden_states = hidden_states + input_injection  # z_H + z_L
    # Process through layers...
```

**H-level actual computation**: Processes the sum of two components:
- `hidden = z_H + z_L`

**Hierarchical Information Flow**:
- **L-level receives**: Own state (z_L) + High-level guidance (z_H) + Original input (input_embeddings)
- **H-level receives**: Own state (z_H) + Low-level details (z_L)

This bidirectional injection pattern ensures:
1. **L-level triple combination**:
   - State continuity (z_L): Maintains low-level working memory
   - Hierarchical guidance (z_H): Incorporates high-level planning
   - Grounding (input_embeddings): Keeps connection to original problem

2. **H-level dual combination**:
   - Global planning (z_H): Maintains high-level strategic state
   - Detail feedback (z_L): Integrates low-level execution information

The additive combination allows gradient flow through all paths and prevents information loss across iterations.

#### State Evolution
```
H^(t+1) = H_level(H^t, L^t)
L^(t+1) = L_level(L^t, H^t + X)
```

More precisely, inside the modules:
```
H_internal^(t+1) = H^t + L^t  (then processed through H-layers)
L_internal^(t+1) = L^t + H^t + X  (then processed through L-layers)
```

Where:
- H^t: High-level state at iteration t
- L^t: Low-level state at iteration t  
- X: Input embeddings
- The modules internally add their inputs before processing

**Critical Innovation**: This design provides:
- **Deep computation**: With H_cycles=2, L_cycles=2, this gives 3 no-grad + 2 grad = 5 effective computation steps
- **Stable training** by avoiding vanishing/exploding gradients through long sequences  
- **Efficient backpropagation** through only the final L and H updates
- **Planning without gradient noise** in the no-grad iterations

**Exact computation sequence** (H_cycles=2, L_cycles=2):
```
Step 1 [no-grad]: z_L = L_level(z_L, z_H + X)    # H=0, L=0
Step 2 [no-grad]: z_L = L_level(z_L, z_H + X)    # H=0, L=1  
Step 3 [no-grad]: z_H = H_level(z_H, z_L)        # H=0 complete
Step 4 [no-grad]: z_L = L_level(z_L, z_H + X)    # H=1, L=0
                  # Skip: L_level for H=1,L=1 (saved for gradients)
                  # Skip: H_level for H=1 (saved for gradients)
Step 5 [grad]:    z_L = L_level(z_L, z_H + X)    # Final L with ∇
Step 6 [grad]:    z_H = H_level(z_H, z_L)        # Final H with ∇
```

**Total depth**: 6 sequential transformer layer applications (4 L-level + 2 H-level)
**Gradient flow**: Only through final 2 steps, preventing gradient issues

### 2. Adaptive Computation Time (ACT) with Q-Learning

The model implements a sophisticated halting mechanism using Q-learning (`lines 40-55, 240-283`):

#### **Key Components**:
- **Q-Head**: Neural network that outputs Q-values for halt/continue decisions
- **Exploration Strategy**: ε-greedy with `halt_exploration_prob: 0.1`
- **Maximum Steps**: Configurable limit (`halt_max_steps: 16`)

#### **Q-Learning Innovation**:
```python
# Q-value computation for halting decision
q_halt_logits, q_continue_logits = self.q_head(z_H[:, 0])
halted = q_halt_logits > q_continue_logits

# Target Q-value (no replay buffer needed)
target_q = torch.sigmoid(max(next_q_halt, next_q_continue))
```

**Significance**: Unlike traditional ACT, HRM learns when to halt through reinforcement learning without explicit supervision, allowing the model to adaptively allocate computation based on problem difficulty.

#### **Position 0 Transformation for Q-Head**

The Q-head makes halt/continue decisions based on position 0 of z_H, which undergoes a crucial transformation:

```python
# Line 186: Start with raw embeddings
input_embeddings = self._input_embeddings(...)  # [puzzle_emb | token_emb] + position_emb
# Position 0 specifically contains the zero-initialized puzzle embedding (task conditioning)

# Lines 195, 198: Multiple cycles WITHOUT gradients
z_L = self.L_level(z_L, z_H + input_embeddings)
z_H = self.H_level(z_H, z_L)
# Position 0 accumulates reasoning through self-attention

# Lines 203-204: Final cycle WITH gradients  
z_L = self.L_level(z_L, z_H + input_embeddings)
z_H = self.H_level(z_H, z_L)
# Position 0 receives final refinement with gradient flow

# Line 211: Q-head uses TRANSFORMED position 0
q_logits = self.q_head(z_H[:, 0])
# Position 0 now contains global reasoning summary
```

**Position 0 Evolution**:
1. **Initial State**: Zero-initialized puzzle embedding (`init_std=0`, line 119-120)
   - Acts as learnable task identifier (e.g., Sudoku vs Maze)
   - Provides constant conditioning for the specific puzzle type

2. **Transformation Process**: Through multiple H/L cycles
   - Accumulates information from all positions via self-attention
   - Integrates both high-level planning (H) and low-level execution (L)
   - Gradient-free cycles allow exploration without commitment

3. **Final State**: Global reasoning token
   - Contains model's understanding of problem completion
   - Optimal representation for halt/continue decision
   - Similar to BERT's [CLS] but with initial task conditioning

This design cleverly repurposes the puzzle embedding position as a decision token, ensuring the halt mechanism is both task-aware and reasoning-informed.

### Q-Learning Objective

The Q-head training is a **key innovation** that integrates reinforcement learning with language modeling in a single gradient step:

#### **Simultaneous Training Mechanism**
```python
# 1. Gradient-free exploration (lines 189-199)
with torch.no_grad():
    for _H_step in range(H_cycles - 1):  # No gradients for planning
        for _L_step in range(L_cycles):
            z_L = L_level(z_L, z_H + input_embeddings)
        z_H = H_level(z_H, z_L)

# 2. Final gradient step (lines 202-204)
z_L = L_level(z_L, z_H + input_embeddings)  # Gradients flow
z_H = H_level(z_H, z_L)                     # Gradients flow

# 3. Q-head prediction from final state
q_halt_logits, q_continue_logits = q_head(z_H[:, 0])
```

#### **Joint Loss Function (ACT Loss)**
```python
# Three losses computed simultaneously (losses.py lines 83-95):

# 1. Language modeling loss - standard next-token prediction
lm_loss = cross_entropy(logits, labels)  # Line 83

# 2. Q-halt loss - learns when to stop based on correctness
seq_is_correct = (num_correct_predictions == num_valid_tokens)  # Lines 66-67
q_halt_loss = BCE(q_halt_logits, seq_is_correct)  # Line 84
# Directly supervised: halt if problem solved correctly

# 3. Q-continue loss - temporal difference (TD) learning with bootstrapped Q-values
# KEY: Uses future Q-values, NOT final correctness!
target_q_continue = sigmoid(max(next_q_halt, next_q_continue))  # Line 279
q_continue_loss = BCE(q_continue_logits, target_q_continue)  # Line 94
# Temporal difference learning: learns from future state Q-values (enables learning from intermediate improvements)

# Combined ACT loss (line 101)
L_total = lm_loss + 0.5 * (q_halt_loss + q_continue_loss)
```

**Key Innovation**: The 0.5 weighting balances task performance (lm_loss) with computational efficiency (Q-losses). This joint optimization ensures the model learns both:
- **What to compute** (language modeling)
- **When to stop computing** (adaptive halting)

The `seq_is_correct` signal provides direct supervision for halting - the model learns to stop when it has solved the problem correctly, not just when it's confident.

#### **Critical Q-Head Initialization** 
```python
# Special initialization for stable learning (lines 142-144)
self.q_head.weight.zero_()    # Start with zero weights  
self.q_head.bias.fill_(-5)    # Strong bias towards continuing
```
This initialization prevents premature halting and allows the model to learn appropriate stopping criteria.

#### **Temporal Difference Learning with Bootstrapping**
```python
# Innovative target computation (line 279) - uses TD learning!
next_q_halt, next_q_continue = self.inner(new_inner_carry, new_current_data)[-1]
target_q_continue = sigmoid(
    where(is_last_step, next_q_halt, maximum(next_q_halt, next_q_continue))
)
```

**Key insight: Q-continue learns from intermediate improvements through TD learning**
- **Not dependent on sparse completion signals**: Q-continue uses future Q-values (bootstrapping), not waiting for final `seq_is_correct`
- **Propagates value information**: Even if puzzle not fully solved, if next step yields better Q-values, current step learns to continue
- **Handles sparse rewards**: Through bootstrapping, value information propagates backward from future steps, overcoming sparsity of puzzle completion signals

Key differences from standard Q-learning:
- **Simultaneous updates**: Q-head trained with same gradients as language model
- **No replay buffer**: Uses parallel environments (large batch size)
- **Dual supervision**:
  - Q-halt: Directly supervised by task correctness (sparse signal)
  - Q-continue: Learns through TD learning from future Q-values (dense signal)
- **Binary targets**: Uses BCE loss instead of MSE for Q-values
- **Shared representations**: Q-head learns from same z_H as language output

## Critical Analysis

![HRM Architecture](resources/hrm.png)

### Strengths
1. **Parameter Efficiency**: 27M parameters outperform models 100× larger
2. **Sample Efficiency**: Learns from only 1000 examples
3. **Generalization**: Strong performance on unseen puzzle types
4. **Interpretability**: Clear separation of planning vs. execution
5. **Information Preservation**: Triple input combination (z_L + z_H + X) ensures no information bottlenecks
6. **Novel Q-Learning Integration**: First architecture to successfully combine ACT with RL for reasoning

### Limitations

- **Performance gaps in reproduction**: Community attempts to replicate HRM results often achieve **10–15% lower accuracy** than the paper's reported numbers (e.g., Sudoku, ARC-AGI), suggesting sensitivity to setup and compute budgets.  

- **Overfitting and memorization concerns**: Analyses (e.g., ARC Prize) highlight that part of HRM's performance may come from *task similarity* or hidden training tricks, rather than true generalization to unseen tasks.  

- **Ambiguity in outer-loop refinement**: A significant portion of HRM's gains appear linked to its **refinement loop**, which is not fully detailed in the original release, making exact reproduction harder.  

- **Checkpoint and hidden evaluation issues**: Official checkpoints rely on **semi-private hold-out datasets**, limiting independent verification and complicating community benchmarking.  

- **Hardware and compute dependence**: Reported results may rely on substantial GPU resources and exact environment tuning; community reproductions on different hardware (e.g., consumer GPUs like RTX 5090) often see degraded performance.  

- **Transparency gaps**: Missing details such as seeds, hyperparameters, and containerized environments reduce reproducibility and make it difficult for others to fully validate claims.

### Computational Efficiency
1. **Inference Speed**: Single forward pass vs. autoregressive generation (10-100× faster)
2. **Memory Efficiency**: Dual state requires ~2× memory but enables much deeper computation
3. **Training Efficiency**: Joint Q-learning + LM training more efficient than separate training
4. **Adaptive Depth**: Model uses only needed computation (early halting for easy problems)

### Future Directions
1. **Specialized H/L Architectures**: Use different architectural compositions for H/L modules (e.g., CNN-based H for spatial planning, Transformer-based L for feature processing, or Graph-based H for relational reasoning etc.)

2. **Spatio-Temporal Hierarchical Processing**: Apply dual-hierarchy reasoning where H-module tracks temporal events and trajectories while L-module handles spatial feature extraction, enabling adaptive temporal resolution for video understanding and embodied AI applications

3. **Adaptive Resolution Vision Tasks**: Leverage H-module for strategic spatial planning (where to apply high/low resolution) and L-module for local feature extraction with selected CNN kernels, achieving efficient processing across 2D image manifolds with adaptive computational budgets for any vision task (classification, detection, or segmentation)

4. **Multi-Modal Applications**: Deploy H-module as cross-modal coordinator maintaining semantic understanding across modalities (vision, language, audio) while L-modules specialize in modality-specific processing, enabling dynamic context-aware fusion for multimodal reasoning

5. **Hierarchical Retrieval-Augmented Systems**: Use H-module as strategic query selector for vector databases/knowledge graphs while L-module performs detailed feature extraction on retrieved items, creating efficient RAG systems for CAT, art valuation, legal analysis, and scientific literature review

#### **Spatio-Temporal Hierarchical Processing**
Unifying temporal event tracking with spatial feature extraction for **real-time video understanding and embodied AI**:

**Benchmark Datasets** (for evaluation):
- **Kinetics-400** ([DeepMind](https://www.deepmind.com/open-source/kinetics)): 306K video clips, 400 action classes, ~450GB (requires YouTube download)
- **UCF101** ([CRCV](https://www.crcv.ucf.edu/data/UCF101.php)): 13K videos, 101 action classes, 6.5GB (registration required)
- **HMDB51** ([Brown University](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)): 6.8K clips, 51 action classes, 2GB (form required)

**Core Concept**: Employ the **H-module for temporal event reasoning** (tracking object trajectories, understanding causal relationships, predicting future states) while the **L-module handles spatial feature extraction** (processing individual frames, detecting objects, analyzing scenes). The architecture creates a **bidirectional information flow** where L-module's spatial features become new tokens for H-module's temporal reasoning, while H-module's temporal context guides L-module's spatial attention.

**Applications**: Autonomous driving (trajectory prediction with scene understanding), robotic manipulation (action planning with visual feedback), sports analytics (tracking player strategies over time), surveillance systems (anomaly detection with context awareness), and video analysis (surgery monitoring with temporal-spatial precision).

#### **Adaptive Resolution Vision Tasks**
Dynamically allocating computational resources across images for **efficient multi-scale visual processing**:

**Benchmark Datasets** (for evaluation):
- **ImageNet** ([ILSVRC](https://www.image-net.org/)): 1.28M images, 1000 classes, ~150GB (academic credentials required)
- **MS COCO** ([COCO](https://cocodataset.org/)): 330K images, 80 object classes, 25GB (freely downloadable)
- **Cityscapes** ([Cityscapes](https://www.cityscapes-dataset.com/)): 5K images, 2048x1024 resolution, 60GB (registration required)
- **ADE20K** ([MIT CSAIL](https://groups.csail.mit.edu/vision/datasets/ADE20K/)): 25K images, 150 scene categories, 3.8GB (freely downloadable)

**Core Concept**: With RL methods, use the **H-module** for strategic spatial planning (deciding where to apply high-resolution vs. low-resolution processing) and the **L-module** for local feature extraction with selected CNN kernels. The model learns to allocate computational resources adaptively across the 2D image manifold for any computer vision task - classification, detection, or segmentation.

**Applications**: Medical imaging (focus on pathological regions), autonomous driving (detailed road analysis), mobile devices (efficient on-device processing), and satellite imagery (multi-scale analysis from overview to inspection).

#### **Multi-Modal Applications**
Coordinating cross-modal attention and fusion strategies for **unified multimodal understanding** across vision, language, and audio:

**Core Concept**: The **H-module serves as a cross-modal coordinator** that maintains high-level semantic understanding across modalities (vision, language, audio), while **L-modules specialize in modality-specific processing**. The H-module decides which modality to attend to and how to combine information, while L-modules extract detailed features from their respective inputs.

**Applications**: Visual question answering (coordinate image regions with text queries), video captioning (align visual events with language generation), multimodal reasoning (solve problems requiring both visual and textual understanding), assistive AI (process speech, vision, and text for accessibility), and medical diagnosis (combine imaging, lab results, and clinical notes).

#### **Hierarchical Retrieval-Augmented Systems**
Separating strategic query formulation from detailed feature extraction for **intelligent retrieval from massive knowledge bases**:

**Core Concept**: The **H-module acts as a strategic query selector** that searches vector databases or traverses knowledge graphs to retrieve relevant information, while the **L-module performs detailed feature extraction and value refinement** on retrieved items. This creates a **hierarchical retrieval pipeline** where high-level planning guides low-level processing.

**Example Applications**:

1. **Computer Adaptive Testing (CAT)**: H-module selects appropriate test problems from a question bank based on student ability estimates, while L-module analyzes student responses to update ability assessments and provide feedback.

2. **Art Valuation Systems**: H-module retrieves similar artworks and their features from a database to establish context, while L-module processes the target artwork's specific characteristics conditioned on the retrieved references to predict market value.

3. **Legal Document Analysis**: H-module identifies relevant case precedents and statutes, L-module performs detailed analysis of applicability to current case.

4. **Scientific Literature Review**: H-module navigates citation graphs to find relevant papers, L-module extracts and synthesizes key findings.
