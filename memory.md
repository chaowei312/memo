

## Probabilistic View of Conditional Reasoning

Motivated by the intuition that multiple reasoning trajectories can exist for the same conclusion, for the implicit reasoning module, we adopt the Markov assumption, where the next state depends only on the current state. This enables the use of efficient one-step gradient estimates for optimization.

### Reasoning Modeling
We define a reasoning trajectory as a discrete-time dynamical system represented by the tuple $(\mathcal{S}, \mathcal{F} , \mathcal{C}, \sigma)$, where the dynamics $(\mathcal{S}, \mathcal{F})$ are conditioned on the constraints $(\mathcal{C}, \sigma)$:
- $\mathcal{S} = \{s_t\}_{t=0}^T \subset \mathbb{R}^n$ is a sequence of state vectors.
- $\mathcal{F}$ is a family of differentiable transition functions such that for some $f_t \in \mathcal{F}$ and $c \in \mathcal{C}$, the state evolves as $s_{t+1} = f_t(s_t, c)$.
- $\mathcal{C}$ is a set of constraints.
- $\sigma: \mathbb{R}^n \times \mathcal{C} \to \{0, 1\}$ is a judgment function that validates if a state $s_t$ satisfies a constraint $c$.

The probability of a trajectory $\mathcal{S}$ conditioned on the set of constraints $\mathcal{C}$ is:

$$
P(\mathcal{S} | \mathcal{C}, \sigma) = P(s_0 | \mathcal{C}, \sigma) \prod_{t=0}^{T-1} P(s_{t+1} | s_t, \mathcal{C}, \sigma)
$$

Where:
- **$P(s_0 | \mathcal{C}, \sigma)$** is the **initial state distribution**.
- **$P(s_{t+1} | s_t, \mathcal{C}, \sigma)$** is the **probabilistic transition function**. The judgment function $\sigma$ can influence this by assigning zero probability to invalid transitions.
