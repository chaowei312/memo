## Adaptive Computation Time (ACT) with Objective-Subjective Contributions

### Overview
We propose an adaptive computation time mechanism based on recursive neural network processing with objective and subjective contribution measures for determining computational depth. The subjective contribution score is a positive value based on output features at each step, providing a non-biased task difficulty measure. The objective contribution score is a positive estimate of the contribution by the network between each step, estimating how much the network at the current iteration has contributed to the output feature. The subjective contribution network is trained using the accumulated objective score after k iterations.

### Core Framework

#### Recursive Process

Consider a recursive process with a neural network $f$ such that:

$$f^n(x^*) = x^* \quad \text{(converges with } n \text{ bounded)}$$

where:
- $f$ is a neural network transformation
- $x^*$ is a fixed point of the transformation
- $n$ is the number of recursive applications (bounded)

#### Step-wise Transformation

At each iteration $i$, we have:
$$f^i(x) = y$$

where $y$ is the output after $i$ applications of $f$ to input $x$.

### Contribution Measures

#### Objective Contribution ($\tau_i$)

The contribution $\tau_i$ of the $i$-th step is defined as a **strictly positive value**:

$$\tau_i = \frac{r}{|C_x \cdot C_y| + \epsilon} \in \mathbb{R}^+$$

where:
- $C_x = \text{CLS}(x)$ denotes the class token of $x$
- $C_y = \text{CLS}(y)$ denotes the class token of $y$
- $r$ is a relevance measure (positive) computed as:

$$r = g(y - x) \in \mathbb{R}^+$$

where $g$ is a neural network with positive output activation (e.g., softplus, ReLU, or exponential) that learns to evaluate the difference between consecutive states.
- $\epsilon$ is a small constant for numerical stability
- The absolute value ensures the denominator is positive

#### Subjective Contribution

The subjective contribution $o(y_i)$ provides a **positive, non-accumulative** score that changes at each iteration:

$$o: \mathcal{Y} \rightarrow \mathbb{R}^+ \quad \text{where } o(y_i) > 0$$

Key properties:
- **Positive valued**: Always returns positive numbers
- **Non-accumulative**: Unlike objective contributions, subjective score doesn't accumulate
- **State-dependent jumps**: Can change discontinuously between iterations based on current state quality
- **Dynamic threshold**: Represents the "satisfaction level" for the current state

The subjective score $o(y_i)$ evaluates the quality or completeness of the current output state $y_i$ at iteration $i$.

### ACT Termination Criterion

The adaptive computation terminates when the cumulative objective contribution exceeds the current subjective threshold:

$$\text{Stop when: } S_n = \sum_{i=1}^{n} \tau_i > o(y_n)$$

where:
- $S_n$ is the accumulated objective contribution (monotonically increasing)
- $o(y_n)$ is the subjective threshold at iteration $n$ (can jump/change at each step)

This criterion balances:
- **Objective progress**: Accumulated contributions from each transformation step (always growing)
- **Subjective quality**: Current output quality assessment (dynamically changing)

### Training Strategy: Objective-Guided Subjective Learning

A key insight is to **use objective contributions to train the subjective head**:

#### Training Approach with Delayed Alignment

**Delayed Objective-Teacher Alignment**: Wait until iteration $k$ (where features begin converging) before computing alignment loss:
   $$\mathcal{L}_{\text{align}} = \text{MSE}(o(y_n), S_n) \quad \text{for } n \geq k$$
   
   Only compute alignment loss after $k$ iterations using stabilized cumulative objective $S_n$ to supervise subjective scorer.
