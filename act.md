## Adaptive Computation Time (ACT) as Reinforcement Learning: Learning When to Stop Thinking

### Overview
We formulate Adaptive Computation Time (ACT) as a **reinforcement learning problem** where a neural network must learn an optimal halting policy for recursive computation. Unlike traditional fixed-depth architectures, our approach treats computational depth as a sequential decision-making problem: at each iteration, the network must decide whether to continue processing or halt computation.

The system learns two complementary value functions:
- **Objective contribution** ($\tau$): Estimates the incremental value added by each computational step
- **Subjective threshold** ($o$): A dynamic value function estimating the sufficient quality level for the current state

This creates a learned halting policy where computation stops when accumulated contributions exceed the current quality threshold, fundamentally solving an exploration-exploitation tradeoff between computational efficiency and task performance.

### Core Framework

#### Recursive Process

Consider a recursive process with a neural network $f$ such that:

$$x^{n+1} = f(x^{n})$$

where:
- $f$ is a neural network transformation
- $x^{n}$ is the state at iteration $n$
- The process iterates until convergence or a maximum number of steps

#### Step-wise Transformation

At each iteration $i$, starting from initial input $x^{0}$:
$$x^{i+1} = f(x^{i})$$

where $x^{i}$ represents the state after $i$ iterations of applying $f$.

### Reinforcement Learning Formulation

We cast the adaptive computation problem as a Markov Decision Process (MDP):

#### MDP Components
- **State space** $\mathcal{S}$: The intermediate representations $\{x^{i}\}$ at each recursive step
- **Action space** $\mathcal{A}$: Binary actions at each state - {continue, halt}
- **Transition dynamics**: Deterministic transition $x^{i+1} = f(x^{i})$ when action = continue
- **Policy** $\pi(a|x^{i})$: The halting policy learned through the objective-subjective mechanism
- **Reward function** $r(x^{i}, a)$: Implicit reward balancing:
  - **Efficiency penalty**: $-\lambda$ for each additional computation step (continue action)
  - **Performance reward**: Task accuracy/loss improvement at termination

#### Dual Value Functions for Halting Decisions
Our system employs **two distinct value functions** with competing objectives:
- The accumulated subjective contribution $S_n = \sum_{i=1}^{n} \tau_i$ measures the cumulative progress made through iterations
- The objective threshold $o(x^{n})$ estimates the total contribution required for the current state

The halting criterion $S_n > o(x^{n})$ creates an **emergent policy** from comparing these two value estimates - neither function alone determines the action.

### Transformer-Based ACT Bottleneck

- **Conv encoder front-end**: Spatial features from the U-Net encoder remain convolutional, preserving the pyramidal hierarchy before ACT processing.
- **Recurrent transformer core**: At the bottleneck we tokenize the feature map via stride-$p$ convolutional patch embeddings, append a learnable CLS token, and run a weight-shared transformer layer at every ACT iteration to produce a refined spatial state plus the CLS summary.
- **CLS-driven threshold**: The CLS embedding is routed through a learnable function to estimate the objective threshold $o(x^n)$, one of the two value functions.
- **State-and-difference contribution**: A learnable function takes the current state and state difference to produce the subjective contribution score $\tau_i = g(x^i, \Delta x^i)$, the other value function.
- **Shared halting policy**: The recurrent transformer, contribution function, and threshold function together implement the continue/halt policy while keeping the encoder/decoder unchanged.

### Contribution Measures

#### Subjective Contribution ($\tau_i$)

The contribution $\tau_i$ of the $i$-th step is defined as a **strictly positive value**:

$$\tau_i = g(x^{i}, \Delta x^{i}) \in \mathbb{R}^{+}$$

where:
- $x^{i}$ is the current state at iteration $i$
- $\Delta x^{i} = x^{i} - x^{i-1}$ is the change between consecutive states
- $g$ is a learnable function with positive output
- The function learns to measure meaningful progress from both the absolute state and relative change

#### Objective Threshold

The objective threshold $o(x^{i})$ provides a **positive, non-accumulative** score that changes at each iteration:

$$o: \mathcal{X} \rightarrow \mathbb{R}^{+}$$

where:
- $x^{i}$ is the current state at iteration $i$
- $o$ is a learnable function that maps the state to a positive scalar
- The function learns to estimate how much total contribution is needed for the current state

Key properties:
- **Positive valued**: Always returns positive numbers
- **Non-accumulative**: Unlike subjective contributions, objective threshold doesn't accumulate
- **State-dependent jumps**: Can change discontinuously between iterations based on current state quality
- **Dynamic threshold**: Represents the "satisfaction level" for the current state

The objective threshold $o(x^{i})$ evaluates the quality or completeness of the current output state $x^{i}$ at iteration $i$.

### ACT Termination Criterion

The adaptive computation terminates when the cumulative subjective contribution exceeds the current objective threshold:

$$\text{Stop when: } S_n = \sum_{i=1}^{n} \tau_i > o(x^{n})$$

where:
- $S_n$ is the accumulated subjective contribution (monotonically increasing)
- $o(x^{n})$ is the objective threshold at iteration $n$ (can jump/change at each step)

This criterion balances:
- **Subjective progress**: Accumulated contributions from each transformation step (always growing)
- **Objective quality**: Current output quality assessment (dynamically changing)

### Reinforcement Learning Training Workflow

#### Exploration Phase
- **Trajectory rollout**: Execute the current policy by unrolling the recurrent network $f$ for up to $K$ steps (exploration horizon), collecting the trajectory $\{x^{1}, ..., x^{K}\}$ with corresponding value estimates
- **Optimal action discovery**: Use the environment feedback (task loss) to identify the optimal halting point $k^{*}$ - this serves as the **true reward signal** revealing where the policy should have stopped
- **Policy evaluation**: Compare the current policy's decision (where $S_n > o(x^{n})$ first occurs) against the optimal action $k^{*}$

#### Policy Improvement Phase
- **Credit assignment**: When the policy's decision differs from $k^{*}$, propagate error signals to update both value functions
- **Dual value learning**: Use bootstrapped updates between the two value estimators:
  - Update $o$ (threshold estimator) using $S_n$ as the target value at $k^{*}$
  - Update $g$ (contribution estimator) using policy gradients when decisions are incorrect
- **Alternating optimization**: Alternate between updating the threshold function $o$ and the contribution function $g$ to prevent collapse and maintain balanced learning

### Dual Value Network Training Strategy

The objective and subjective heads form a **dual value network system** where the policy emerges from their comparison. This differs from standard actor-critic architectures:

- **Standard Actor-Critic**: Separate policy network (actor) and value network (critic), with actor outputting action probabilities
- **ACT Dual Value System**: Two value networks with different objectives; policy emerges deterministically from their comparison $S_n > o(x^n)$
- **Neither g nor o is a policy network** - both output continuous values, and the decision emerges from their relative magnitudes

#### Competing Value Functions with Bootstrapped Learning

- **Threshold calibration**: After discovering the optimal stopping point $k^{*}$, align the threshold estimator:

  $$\mathcal{L}_{threshold} = \text{MSE}(o(x^{k^*}), S_{k^*}) \quad \text{at optimal } k^{*}$$
  
- **Dual value updates**: The two value functions learn complementary objectives:
  - **Threshold update**: Train $o$ to predict the correct cumulative contribution needed at optimal states
  - **Contribution update**: Train $g$ to produce contributions that lead to correct stopping decisions via policy gradients
  
- **Conditional learning**: Updates occur only when the emergent policy (from $S_n > o(x^n)$) disagrees with the optimal action $k^{*}$, focusing learning on decision boundaries

### Key Distinction: Emergent Policy from Dual Values

While ACT is fundamentally reinforcement learning (learning through trial and error without action supervision), it employs a unique architecture:

- **Not traditional actor-critic**: No explicit policy network outputting action probabilities
- **Not standard Q-learning**: No single value function determining actions
- **ACT's innovation**: Two competing value functions ($g$ and $o$) whose comparison yields an emergent, deterministic policy

This design maintains all core RL properties (exploration, credit assignment, reward-based learning) while introducing a novel mechanism for action selection through dual value comparison.

