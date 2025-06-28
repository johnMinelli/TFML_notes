# Neural Networks and Stochastic Gradient Descent

## Table of Contents
1. [From Linear Models to Neural Networks](#from-linear-models-to-neural-networks)
2. [Shallow Neural Networks](#shallow-neural-networks)
3. [Activation Functions](#activation-functions)
4. [Deep Neural Networks](#deep-neural-networks)
5. [Network Architecture and Terminology](#network-architecture-and-terminology)
6. [Regularization in Neural Networks](#regularization-in-neural-networks)
7. [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
8. [SGD Convergence Theory](#sgd-convergence-theory)
9. [Advanced SGD Variants](#advanced-sgd-variants)
10. [Backpropagation](#backpropagation)
11. [Theory vs Practice in Deep Learning](#theory-vs-practice-in-deep-learning)

---

## From Linear Models to Neural Networks

### The Limitation of Linear Models

**Classical approach**: Linear models with feature engineering
```
f(x) = w^T φ(x)
```

**Two-stage procedure**:
1. **Feature design**: Expert designs φ(x) (manual, domain-specific)
2. **Optimization**: Learn weights w (automatic, well-understood)

**Problems**:
- Requires domain expertise for feature design
- Features are fixed and not adaptive
- Limited expressiveness for complex patterns

### Beyond Linear Models: Adding Nonlinearity

**Goal**: Make function nonlinear in parameters, not just in features

**Approach 1**: Single nonlinearity
```
f(x) = σ(w^T x)
```
**Limitation**: Very restricted function class

**Approach 2**: Linear combination of nonlinearities
```
f(x) = Σⱼ₌₁ᴹ vⱼ σ(wⱼ^T x + bⱼ)
```

**Key insight**: This creates a **parameterized function class** where we learn both:
- The "features" wⱼ^T x + bⱼ (feature extraction)
- The "weights" vⱼ (feature combination)

---

## Shallow Neural Networks

### Mathematical Definition

A **shallow neural network** (single hidden layer) computes:
```
f(x) = Σⱼ₌₁ᴹ vⱼ σ(wⱼ^T x + bⱼ)
```

**Parameters**:
- **Weights**: wⱼ ∈ ℝᵈ (input-to-hidden connections)
- **Biases**: bⱼ ∈ ℝ (hidden unit thresholds)  
- **Output weights**: vⱼ ∈ ℝ (hidden-to-output connections)
- **Width**: M (number of hidden units)

### Expressiveness Examples

**Example 1**: Absolute value function
```
|x| = ReLU(x) + ReLU(-x)
```
- w₁ = 1, b₁ = 0, v₁ = 1
- w₂ = -1, b₂ = 0, v₂ = 1

**Example 2**: Piecewise linear functions
- Can approximate any continuous function arbitrarily well
- Number of pieces grows with M

### Biological Motivation

**Artificial neuron model** (McCulloch-Pitts, 1943):
1. **Inputs**: x₁, ..., xᵈ (dendrites)
2. **Weighted sum**: Σᵢ wᵢxᵢ + b (cell body)
3. **Activation**: σ(·) (action potential)
4. **Output**: Activity level (axon)

**Biological interpretation**:
- wᵢ: Synaptic strengths
- b: Activation threshold
- σ: Firing behavior (active/inactive)

---

## Activation Functions

### Rectified Linear Unit (ReLU)

**Definition**: σ(z) = max(0, z)

**Properties**:
- **Simple**: Cheap to compute
- **Non-differentiable**: At z = 0
- **Sparse**: Can produce exactly zero outputs

**Dead neuron problem**:
```
If b < 0 and wᵀx + b < 0 for all training data,
then ∂f/∂w = 0 and ∂f/∂b = 0
```
**Consequence**: Neuron never updates (gradient is zero)

**Practical solution**: Use subdifferentials (treat ∂ReLU(0) as 0 or 1)

### Sigmoid Function

**Definition**: σ(z) = 1/(1 + e^(-z))

**Properties**:
- **Smooth**: Differentiable everywhere
- **Bounded**: Output ∈ (0,1)
- **Probabilistic interpretation**: Can represent probabilities
- **Vanishing gradients**: σ'(z) ≈ 0 for |z| large

### Hyperbolic Tangent

**Definition**: σ(z) = tanh(z) = (e^z - e^(-z))/(e^z + e^(-z))

**Properties**:
- **Zero-centered**: Output ∈ (-1,1)
- **Smooth**: Differentiable everywhere
- **Vanishing gradients**: Similar to sigmoid

### Leaky ReLU

**Definition**: σ(z) = max(αz, z) where α ∈ (0,1)

**Properties**:
- **Solves dead neurons**: Always has non-zero gradient
- **Simple**: Easy to compute
- **Parameter**: α typically 0.01

### Softplus (Smooth ReLU)

**Definition**: σ(z) = log(1 + e^z)

**Properties**:
- **Smooth approximation** of ReLU
- **Always positive**: σ(z) > 0
- **Approaches ReLU**: As temperature → 0

---

## Deep Neural Networks

### Recursive Definition

A **deep neural network** with L layers computes:
```
H₀ = x
Hₗ = σ(WₗHₗ₋₁ + bₗ)  for ℓ = 1, ..., L
f(x) = H_L
```

**Dimensions**:
- H₀ ∈ ℝᵈ⁰ (input dimension)
- Hₗ ∈ ℝᵈˡ (hidden layer dimensions)
- Wₗ ∈ ℝᵈˡ ˣ ᵈˡ⁻¹ (weight matrices)
- bₗ ∈ ℝᵈˡ (bias vectors)

### Compositional View

**Function composition**:
```
f(x) = f_L ∘ f_{L-1} ∘ ... ∘ f₁(x)
```
where fₗ(h) = σ(Wₗh + bₗ)

### Why Deep Networks?

**Compositionality hypothesis**: Many natural functions have compositional structure
- Functions depend on hierarchical features
- Each layer extracts features at different abstraction levels
- Depth enables efficient representation of complex dependencies

**Example**: Image recognition
- Layer 1: Edges and textures
- Layer 2: Object parts
- Layer 3: Objects
- Layer 4: Scenes

**Theoretical advantage**: 
- **Exponential expressiveness**: Deep networks can represent certain functions exponentially more efficiently than shallow ones
- **Curse of dimensionality mitigation**: Focus on relevant low-dimensional structure

---

## Network Architecture and Terminology

### Layer Types

1. **Input layer**: H₀ (data)
2. **Hidden layers**: H₁, ..., H_{L-1} (intermediate representations)
3. **Output layer**: H_L (predictions)

### Network Classifications

**By depth**:
- **Shallow**: L = 1 (single hidden layer)
- **Deep**: L > 1 (multiple hidden layers)

**By width**: 
- **Width**: max{d₁, ..., d_L} (maximum layer size)
- **Narrow**: Small hidden dimensions
- **Wide**: Large hidden dimensions

### Weight Normalization

**Standard parameterization**:
```
f(x) = Σⱼ vⱼ σ(wⱼᵀx + bⱼ)
```

**Weight normalized parameterization**:
```
f(x) = Σⱼ vⱼ σ(ρⱼ (dⱼᵀx) + bⱼ)
```
where ‖dⱼ‖ = 1 and ρⱼ ≥ 0

**Properties**:
- **Same function class**: Different parameterization
- **Different optimization**: Changes gradient flow
- **Separates magnitude and direction**: ρⱼ controls magnitude, dⱼ controls direction

---

## Regularization in Neural Networks

### Weight Decay

**Standard L₂ regularization**:
```
R(θ) = λ Σⱼ (‖wⱼ‖² + bⱼ²)
```

**Balanced regularization** (accounting for input norms):
```
R(θ) = λ Σⱼ (‖wⱼ‖²/κ² + bⱼ²)
```
where κ² = E[‖x‖²]

### Positive Homogeneity and Implicit Regularization

**Key property**: ReLU is positively homogeneous:
```
σ(αz) = ασ(z) for α ≥ 0
```

**Scaling invariance**: For α = (α₁, ..., αₘ) with αⱼ > 0:
```
f(x; θ) = f(x; θ_α)
```
where θ_α = (α₁w₁, ..., αₘwₘ, v₁/α₁, ..., vₘ/αₘ, b₁/α₁, ..., bₘ/αₘ)

**Optimal scaling**: Given parameters θ, find α minimizing L₂ penalty:
```
min_α Σⱼ (αⱼ²‖wⱼ‖² + (vⱼ/αⱼ)² + (bⱼ/αⱼ)²)
```

**Solution**: αⱼ* = (vⱼ² + bⱼ²/κ²)^(1/4) / ‖wⱼ‖^(1/2)

**Implied regularization**: After optimal rescaling:
```
R_eff(θ) = Σⱼ 2‖wⱼ‖ √(vⱼ² + bⱼ²/κ²)
```

**Key insight**: L₂ regularization + positive homogeneity → **L₁-type regularization**
- Promotes sparsity (some neurons become inactive)
- More robust than pure L₂ penalty

---

## Stochastic Gradient Descent (SGD)

### Motivation

**Problem with full gradient**:
```
∇L(θ) = (1/n) Σᵢ₌₁ⁿ ∇ℓᵢ(θ)
```
- **Expensive**: Need to process all n examples per step
- **Redundant**: Similar examples give similar gradients
- **Large datasets**: n can be millions or billions

**SGD solution**: Use single example per step
```
θₖ₊₁ = θₖ - γₖ ∇ℓᵢₖ(θₖ)
```

### Data Selection Strategies

**1. Cyclic sampling**:
```
i₁ = 1, i₂ = 2, ..., iₙ = n, iₙ₊₁ = 1, ...
```
- **Deterministic**: Predictable order
- **Analysis**: Harder to analyze theoretically

**2. Uniform random sampling**:
```
iₖ ~ Uniform{1, ..., n} (with replacement)
```
- **Stochastic**: Random selection each step
- **Analysis**: Easier convergence proofs
- **Practice**: May see same example multiple times

**3. Random reshuffling**:
```
Each epoch: Random permutation of {1, ..., n}
```
- **Balanced**: See each example exactly once per epoch
- **Practice**: Often works best
- **Analysis**: Most complex to analyze

### SGD as Stochastic Optimization

**Two perspectives**:

**1. Finite sum optimization**:
```
min_θ (1/n) Σᵢ₌₁ⁿ ℓᵢ(θ)
```
Multiple passes through finite dataset

**2. Stochastic optimization**:
```
min_θ E[ℓ(θ; ξ)]
```
Single pass through infinite stream

**Connection**: As n → ∞, finite sum approaches expectation

### Stochastic Gradient Property

**Key property**: Random gradient is unbiased:
```
E[∇ℓᵢₖ(θₖ)] = ∇L(θₖ)
```

**Variance**: 
```
Var[∇ℓᵢₖ(θₖ)] = E[‖∇ℓᵢₖ(θₖ) - ∇L(θₖ)‖²]
```

**Trade-off**: 
- **Low cost per iteration**: O(1) vs O(n)
- **High variance**: Noisy gradient estimates
- **Many iterations**: Need more steps to converge

---

## SGD Convergence Theory

### Assumptions

**Function properties**:
1. **Convexity**: Each ℓᵢ is convex
2. **Smoothness**: Each ℓᵢ is L-Lipschitz smooth
3. **Existence**: Minimizer θ* exists

**Gradient properties**:
4. **Bounded variance**: 
   ```
   E[‖∇ℓᵢₖ(θₖ) - ∇L(θₖ)‖²] ≤ σ²
   ```

### Main Convergence Theorem

**Theorem**: Under above assumptions with step sizes γₖ satisfying:
```
Σₖ γₖ = ∞  and  Σₖ γₖ² < ∞
```

we have:
```
E[L(θ̄ₖ) - L*] ≤ ‖θ₀ - θ*‖²/(2 Σᵢ₌₁ᵏ γᵢ) + σ² Σᵢ₌₁ᵏ γᵢ²/(2 Σᵢ₌₁ᵏ γᵢ)
```

where θ̄ₖ = (Σᵢ₌₁ᵏ γᵢθᵢ)/(Σᵢ₌₁ᵏ γᵢ) is the **weighted average**.

### Robbins-Monro Conditions

**Step size requirements**:
1. **Σₖ γₖ = ∞**: Steps large enough to reach optimum
2. **Σₖ γₖ² < ∞**: Steps decay fast enough to converge

**Common choice**: γₖ = γ₀/√k

**Convergence rate**: O(1/√k) (sublinear)

### Error Decomposition

**Two error sources**:

1. **Optimization error**: ‖θ₀ - θ*‖²/(2 Σᵢ γᵢ)
   - Depends on initialization
   - Decreases with more iterations
   - Vanishes as k → ∞

2. **Stochastic error**: σ² Σᵢ γᵢ²/(2 Σᵢ γᵢ)
   - Due to gradient noise
   - Controlled by step size decay
   - Never fully eliminated

### Strong Convexity

**Additional assumption**: Each ℓᵢ is μ-strongly convex

**Improved rate**: O(log k / k) instead of O(1/√k)

**Key insight**: Variance still prevents linear convergence
- **Deterministic GD**: O(ρᵏ) linear convergence
- **SGD**: O(log k / k) due to gradient noise

---

## Advanced SGD Variants

### Mini-batch SGD

**Idea**: Use multiple examples per step
```
θₖ₊₁ = θₖ - γₖ (1/B) Σⱼ₌₁ᴮ ∇ℓᵢₖⱼ(θₖ)
```

**Benefits**:
- **Reduced variance**: Var[gradient] = σ²/B
- **Parallelization**: Can compute gradients in parallel
- **Hardware efficiency**: Better GPU utilization

**Trade-offs**:
- **More computation per step**: B times more work
- **Same convergence rate**: Still O(1/√k)
- **Practical speedup**: If B can be parallelized

### Variance Reduction Methods

**Problem**: SGD variance doesn't decrease over time

**Solution**: Methods like SVRG, SAGA, SAG
- **Idea**: Use stored gradients to reduce variance
- **Memory**: O(n) storage for previous gradients
- **Convergence**: O(1/k) rate even without strong convexity
- **Linear convergence**: With strong convexity

**SVRG example**:
```
Option I: θₖ₊₁ = θₖ - γ[∇ℓᵢₖ(θₖ) - ∇ℓᵢₖ(θ̃) + ∇L(θ̃)]
Option II: θₖ₊₁ = θₖ - γ∇L(θₖ)
```
Choose Option I with probability p, Option II with probability 1-p

### Momentum Methods

**Heavy ball method**:
```
θₖ₊₁ = θₖ - γ∇L(θₖ) + β(θₖ - θₖ₋₁)
```

**Physical interpretation**:
- Particle with mass moving in potential ∇L
- Friction coefficient decreases over time
- Momentum helps escape shallow minima

**Nesterov acceleration**:
```
θₖ₊₁ = θₖ - γ∇L(θₖ + β(θₖ - θₖ₋₁)) + β(θₖ - θₖ₋₁)
```

**Convergence**: O(1/k²) for smooth convex functions

**Momentum coefficient**: β = (√L - √μ)/(√L + √μ) where μ ≤ L

### Adaptive Methods (AdaGrad family)

**AdaGrad**:
```
Gₖ₊₁ = Gₖ + ∇L(θₖ)∇L(θₖ)ᵀ
θₖ₊₁ = θₖ - γ/√(diag(Gₖ₊₁) + ε) ⊙ ∇L(θₖ)
```

**Intuition**:
- **Adaptive step sizes**: Different rate for each coordinate
- **Frequent directions**: Get smaller steps (accumulated in Gₖ)
- **Rare directions**: Get larger steps
- **Automatic preconditioning**: Adapts to problem geometry

**RMSprop**:
```
Gₖ₊₁ = αGₖ + (1-α)∇L(θₖ)∇L(θₖ)ᵀ
```
**Exponential moving average** instead of full sum

**Adam** (Adaptive Moment Estimation):
```
mₖ₊₁ = β₁mₖ + (1-β₁)∇L(θₖ)           # First moment
vₖ₊₁ = β₂vₖ + (1-β₂)∇L(θₖ)²          # Second moment
m̂ₖ₊₁ = mₖ₊₁/(1-β₁ᵏ⁺¹)               # Bias correction
v̂ₖ₊₁ = vₖ₊₁/(1-β₂ᵏ⁺¹)               # Bias correction
θₖ₊₁ = θₖ - γm̂ₖ₊₁/(√v̂ₖ₊₁ + ε)
```

**Combines**:
- **Momentum**: First moment mₖ
- **Adaptive learning rates**: Second moment vₖ
- **Bias correction**: Accounts for initialization bias

---

## Backpropagation

### The Gradient Computation Problem

**Challenge**: Neural network is a composition:
```
f(x) = f_L ∘ f_{L-1} ∘ ... ∘ f₁(x)
```

**Need**: Gradient ∇_θ ℓ(f(x), y) where θ = (W₁, b₁, ..., W_L, b_L)

**Naive approach**: Apply chain rule directly → **computationally expensive**

### Forward Pass

**Compute intermediate values**:
```
H⁽⁰⁾ = x
Z⁽ˡ⁾ = W^(ℓ)H⁽ˡ⁻¹⁾ + b⁽ˡ⁾
H⁽ˡ⁾ = σ(Z⁽ˡ⁾)
```

**Store**: All H⁽ˡ⁾ and Z⁽ˡ⁾ values needed for backward pass

### Backward Pass

**Define**: δ⁽ˡ⁾ = ∂ℓ/∂Z⁽ˡ⁾

**Recursive computation**:
```
δ⁽ᴸ⁾ = ∂ℓ/∂H⁽ᴸ⁾           # Output layer
δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀδ⁽ˡ⁺¹⁾ ⊙ σ'(Z⁽ˡ⁾)   # Hidden layers
```

**Weight gradients**:
```
∂ℓ/∂W⁽ˡ⁾ = δ⁽ˡ⁾(H⁽ˡ⁻¹⁾)ᵀ
∂ℓ/∂b⁽ˡ⁾ = δ⁽ˡ⁾
```

### Algorithm Summary

**Backpropagation algorithm**:
1. **Forward pass**: Compute H⁽ˡ⁾, Z⁽ˡ⁾ for ℓ = 1, ..., L
2. **Backward pass**: Compute δ⁽ˡ⁾ for ℓ = L, ..., 1
3. **Gradients**: Compute ∂ℓ/∂W⁽ˡ⁾, ∂ℓ/∂b⁽ˡ⁾

**Complexity**: 
- **Time**: O(# parameters) per example
- **Space**: O(# neurons) to store activations

### Automatic Differentiation

**Key insight**: Backpropagation is **reverse-mode automatic differentiation**

**Chain rule**: For composition g(h(x)):
```
(g ∘ h)'(x) = g'(h(x)) · h'(x)
```

**Computational graph**: 
- **Nodes**: Variables and operations
- **Edges**: Dependencies
- **Forward**: Compute function values
- **Backward**: Compute derivatives via chain rule

**Modern frameworks**: TensorFlow, PyTorch implement automatic differentiation
- **Define-by-run**: Build graph dynamically
- **Gradient computation**: Automatic via backpropagation

---

## Theory vs Practice in Deep Learning

### Theoretical Gaps

**What we know**:
- **Convex case**: Strong convergence guarantees for SGD
- **Shallow networks**: Some approximation theory
- **Linear networks**: Dynamics understood

**What we don't know**:
- **Non-convex convergence**: Why SGD works for neural networks
- **Generalization**: Why networks don't overfit with many parameters
- **Architecture choice**: How to design optimal architectures

### Current Research Directions

**1. Stochastic dynamics**: 
- SGD as continuous-time stochastic process
- Escaping saddle points and local minima
- Noise helps optimization

**2. Infinite width analysis**:
- Neural Tangent Kernel (NTK) theory
- Gradient flow in function space
- Connection to kernel methods

**3. Geometric properties**:
- Loss landscape structure
- Implicit regularization of SGD
- Path-connectedness of minima

### Practical Insights

**Why deep learning works** (empirical observations):
1. **Overparameterization helps**: More parameters → better optimization
2. **SGD finds good solutions**: Despite non-convexity
3. **Implicit regularization**: SGD biases toward generalizable solutions
4. **Compositional structure**: Deep networks match natural hierarchies

**Key practical principles**:
- **Initialization matters**: Proper weight initialization crucial
- **Learning rate schedules**: Decay learning rate over time
- **Batch normalization**: Normalize layer inputs
- **Residual connections**: Skip connections help deep networks
- **Dropout**: Random deactivation prevents overfitting

---

## Summary and Key Takeaways

### Neural Networks

1. **Universal approximators**: Can represent complex functions
2. **Hierarchical representations**: Learn features at multiple levels
3. **End-to-end learning**: No manual feature engineering
4. **Compositional structure**: Efficient for hierarchical data

### Stochastic Gradient Descent

1. **Scalable optimization**: Handles large datasets efficiently
2. **Noise helps**: Stochasticity aids escaping local minima
3. **Simple and effective**: Minimal hyperparameter tuning
4. **Foundation**: Basis for most deep learning optimizers

### Theory vs Practice

1. **Convex theory**: Well-understood but limited applicability
2. **Non-convex practice**: Works amazingly well but poorly understood
3. **Active research**: Major theoretical gaps being addressed
4. **Empirical guidance**: Practice often ahead of theory

### Modern Developments

1. **Architecture innovation**: Transformers, ResNets, attention mechanisms
2. **Optimization advances**: Adam, learning rate schedules, normalization
3. **Scaling laws**: Bigger models and datasets → better performance
4. **Transfer learning**: Pre-trained models for new tasks

The field continues to evolve rapidly, with new architectures, optimization methods, and theoretical insights emerging regularly. While our theoretical understanding lags behind practical successes, both continue to advance in tandem.