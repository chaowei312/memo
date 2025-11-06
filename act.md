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

#### Value Functions as Contribution Measures
Our contribution measures are essentially **learned value functions**:
- The accumulated objective contribution $S_n = \sum_{i=1}^{n} \tau_i$ estimates the cumulative value of computation performed
- The subjective threshold $o(x^{n})$ acts as a state-value function $V(x^{n})$ estimating the required computation quality

The halting criterion $S_n > o(x^{n})$ is thus a **learned policy** comparing two value estimates.

### Contribution Measures

#### Subjective Contribution ($\tau_i$)

The contribution $\tau_i$ of the $i$-th step is defined as a **strictly positive value**:

$$\tau_i = \frac{\|x^{i} - x^{i-1}\|_2}{|c_{i-1} \cdot c_i| + \epsilon} \cdot g(x^{i} - x^{i-1}) \in \mathbb{R}^{+}$$

where:
- $c_i = \text{CLS}(x^{i})$ denotes the class token of $x^{i}$
- $\|x^{i} - x^{i-1}\|_2$ is the Euclidean distance between consecutive states
- $g$ is a neural network with positive output activation (e.g., softplus, ReLU, or exponential) that learns to evaluate the difference between consecutive states
- $\epsilon$ is a small constant for numerical stability
- The absolute value ensures the denominator is positive

#### Objective Difficulty

The subjective contribution $o(x^{i})$ provides a **positive, non-accumulative** score that changes at each iteration:

$$o: \mathcal{X} \rightarrow \mathbb{R}^{+} \quad \text{where } o(x^{i}) > 0$$

Key properties:
- **Positive valued**: Always returns positive numbers
- **Non-accumulative**: Unlike objective contributions, subjective score doesn't accumulate
- **State-dependent jumps**: Can change discontinuously between iterations based on current state quality
- **Dynamic threshold**: Represents the "satisfaction level" for the current state

The subjective score $o(x^{i})$ evaluates the quality or completeness of the current output state $x^{i}$ at iteration $i$.

### ACT Termination Criterion

The adaptive computation terminates when the cumulative objective contribution exceeds the current subjective threshold:

$$\text{Stop when: } S_n = \sum_{i=1}^{n} \tau_i > o(x^{n})$$

where:
- $S_n$ is the accumulated objective contribution (monotonically increasing)
- $o(x^{n})$ is the subjective threshold at iteration $n$ (can jump/change at each step)

This criterion balances:
- **Objective progress**: Accumulated contributions from each transformation step (always growing)
- **Subjective quality**: Current output quality assessment (dynamically changing)

### Reinforcement Learning Training Workflow

#### Exploration Phase
- **Trajectory rollout**: Execute the current policy by unrolling the recurrent network $f$ for up to $K$ steps (exploration horizon), collecting the trajectory $\{x^{1}, ..., x^{K}\}$ with corresponding value estimates
- **Optimal action discovery**: Use the environment feedback (task loss) to identify the optimal halting point $k^{*}$ - this serves as the **true reward signal** revealing where the policy should have stopped
- **Policy evaluation**: Compare the current policy's decision (where $S_n > o(x^{n})$ first occurs) against the optimal action $k^{*}$

#### Policy Improvement Phase
- **Credit assignment**: When the policy's decision differs from $k^{*}$, propagate error signals to update the value functions
- **Temporal difference learning**: Use bootstrapped updates between objective and subjective heads:
  - Update $o$ (critic) using $S_n$ as the target value estimate
  - Update $g$ (actor) using $o(x^{n})$ as the value baseline
- **Alternating optimization**: Similar to actor-critic methods, alternate between updating the value estimator $o$ and the policy component $g$ to prevent collapse and maintain diverse exploration

#### Key RL Mechanisms
- **Exploration-exploitation**: The system explores different stopping points during training while exploiting learned value estimates for efficient inference
- **Off-policy learning**: Learn from the optimal trajectory $k^{*}$ even when the current policy would have stopped elsewhere
- **Bootstrapping**: Value functions learn from each other's estimates rather than requiring explicit reward supervision

### Actor-Critic Training Strategy

The objective and subjective heads form an **actor-critic architecture** for the halting policy:

#### Temporal Difference Learning with Delayed Bootstrapping

- **Value target alignment**: After discovering the optimal stopping point $k^{*}$, compute the TD error:
  $$\mathcal{L}_{\text{TD}} = \text{MSE}(o(x^{n}), S_n) \quad \text{for } n \geq k^{*}$$
  
- **Actor-Critic updates**: Following standard actor-critic methodology:
  - **Critic update**: Minimize TD error for value function $o$ while treating $S_n$ as the target (detached)
  - **Actor update**: Update policy parameters in $g$ using the critic's value estimate as baseline (detach $o(x^{n})$)
  
- **Advantage-based learning**: Updates occur only when the policy error is non-zero (i.e., when current policy disagrees with optimal action $k^{*}$), focusing learning on meaningful decision boundaries

#### Connection to Classical RL Algorithms
This approach combines elements from:
- **Double Q-learning**: Alternating value updates prevent overestimation bias
- **A3C/PPO**: Parallel trajectory collection with asynchronous updates
- **TD($\lambda$)**: Bootstrapping from intermediate value estimates
- **REINFORCE with baseline**: Using value estimates to reduce variance in policy gradients

### Why This Is Fundamentally Reinforcement Learning

Despite using supervised task labels, this is **not supervised learning** but rather **reinforcement learning with learned rewards**:

1. **No fixed supervision for actions**: We don't have labeled examples of "stop at step 3" or "continue at step 5". The optimal stopping point $k^{*}$ is discovered through exploration and environmental feedback (task performance).

2. **Sequential decision making**: The core problem is learning a policy for a sequence of decisions (continue/halt), not learning a fixed input-output mapping.

3. **Exploration is essential**: Without exploring different stopping points up to $K$ steps, we cannot discover the optimal policy. This exploration-exploitation tradeoff is fundamental to RL.

4. **Value function learning**: The objective ($S_n$) and subjective ($o$) heads are learning value functions that estimate future returns, not supervised predictors.

5. **Credit assignment problem**: We must assign credit across a trajectory of decisions to determine which stopping point yields the best reward (task performance vs. computation cost).

6. **Policy gradient structure**: The alternating updates between actor (g) and critic (o) follow the mathematical structure of policy gradient methods, not supervised gradient descent.

The supervised task loss merely provides the **reward signal** that guides policy learning - it tells us how good our halting decision was, but doesn't directly supervise the halting mechanism itself. This is analogous to how game scores guide RL agents without providing explicit action labels.
