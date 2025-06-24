# Convex Optimization for Machine Learning

## Table of Contents
1. [Introduction to Optimization in Machine Learning](#introduction-to-optimization-in-machine-learning)
2. [Linear vs Non-Linear Parameterization](#linear-vs-non-linear-parameterization)
3. [First-Order Methods](#first-order-methods)
4. [Oracle Complexity and Algorithm Analysis](#oracle-complexity-and-algorithm-analysis)
5. [Convex Functions](#convex-functions)
6. [Strongly Convex Functions](#strongly-convex-functions)
7. [Smooth Functions and Lipschitz Continuity](#smooth-functions-and-lipschitz-continuity)
8. [Gradient Descent Algorithm](#gradient-descent-algorithm)
9. [Convergence Analysis](#convergence-analysis)
10. [Applications](#applications)

---

## Introduction to Optimization in Machine Learning

Machine learning problems fundamentally involve optimization. We seek to minimize either:
- **Expected Risk**: `E[ℓ(f(X), Y)]` - theoretical objective
- **Empirical Risk**: `(1/n) Σᵢ ℓ(f(xᵢ), yᵢ)` - practical objective

The optimization landscape differs dramatically based on how we parameterize our function class.

---

## Linear vs Non-Linear Parameterization

### Linear Parameterization

**Definition**: A function class has linear parameterization if:
```
f(x; w) = ⟨w, φ(x)⟩
```
where the function depends **linearly** on the parameter w.

**Examples**:
- Linear regression: `f(x; w) = w^T x`
- Kernel methods: `f(x; w) = Σᵢ wᵢ K(x, xᵢ)`
- Feature maps: `f(x; w) = w^T φ(x)`

**Key Property**: The empirical risk becomes **convex** in w when the loss function is convex.

### Non-Linear Parameterization

**Definition**: The function depends non-linearly on parameters.

**Example**: Neural networks where parameters appear in compositions:
```
f(x; W₁, W₂, ...) = σ(W_L σ(W_{L-1} ... σ(W₁x)))
```

**Challenge**: Even with convex losses (like squared loss), the optimization problem becomes **non-convex**.

### Why This Distinction Matters

| Aspect | Linear Parameterization | Non-Linear Parameterization |
|--------|------------------------|----------------------------|
| **Optimization** | Convex (when loss is convex) | Non-convex |
| **Theory** | Well-established | Limited guarantees |
| **Algorithms** | Global optimum guaranteed | Local optima, saddle points |
| **Software** | Mature, reliable | Specialized, heuristic |

**Key Insight**: Convex optimization is "as easy as linear algebra" - we have complete theoretical understanding and efficient algorithms.

---

## First-Order Methods

### Motivation for First-Order Methods

Modern ML optimization problems have two key characteristics:

1. **High Dimensionality**: Parameter spaces with millions to billions of dimensions
   - Linear models: d can be very large
   - Neural networks: Billions of parameters

2. **Large-Scale Data**: Training sets with millions of examples
   - Sum structure: `f(w) = (1/n) Σᵢ fᵢ(w)`
   - Expectation structure: `f(w) = E[f(w; ξ)]`

### Why Not Second-Order Methods?

**Second-order methods** (Newton's method, interior point methods):
- **Pros**: Very fast convergence (quadratic)
- **Cons**: Each iteration requires:
  - Computing Hessian: O(d²) memory
  - Inverting Hessian: O(d³) operations
  - Infeasible for large d

**First-order methods**:
- **Pros**: Each iteration only needs gradients
  - O(d) memory and computation per iteration
  - Matrix-vector operations only
- **Cons**: Slower convergence rate
- **Trade-off**: In ML, we don't need high precision solutions

### Additional Challenges

1. **Non-smooth Functions**: 
   - Hinge loss: `ℓ(y, ŷ) = max(0, 1 - yŷ)` (not differentiable at boundary)
   - L1 regularization: `‖w‖₁` (not differentiable at zero)

2. **Splitting Algorithms**: 
   - Treat smooth and non-smooth parts separately
   - Example: `f(w) = smooth_loss(w) + λ‖w‖₁`

---

## Oracle Complexity and Algorithm Analysis

### Oracle Model

An **oracle** provides local information about the function:
- **First-order oracle**: Returns `f(w)` and `∇f(w)` at point w
- **Cost**: One oracle call per iteration
- **Assumption**: All algorithms use the same oracle type

### Complexity Analysis Framework

**Given**:
- Function class `ℱ` (e.g., convex, L-smooth functions)
- Algorithm `A` that uses first-order oracle
- Accuracy `ε > 0`

**Goal**: Find the **iteration complexity**:
```
K(ε) = min{k : sup_{f∈ℱ} f(w_k) - f* ≤ ε}
```

This is the **worst-case number of iterations** needed to achieve ε-accuracy.

### Convergence Analysis Types

#### 1. Function Value Convergence
Study the sequence: `{f(w_k) - f*}_{k≥0}`

**Common rates**:
- **Sublinear**: `f(w_k) - f* ≤ C/k` 
- **Linear**: `f(w_k) - f* ≤ C ρᵏ` where `0 < ρ < 1`

#### 2. Iterate Convergence  
Study the sequence: `{‖w_k - w*‖}_{k≥0}`

**Importance**: Relevant for sparsity properties (e.g., LASSO solutions)

---

## Convex Functions

### Definition

A function `f: ℝᵈ → ℝ ∪ {+∞}` is **convex** if:
```
f(λw + (1-λ)u) ≤ λf(w) + (1-λ)f(u)
```
for all `w, u ∈ ℝᵈ` and `λ ∈ [0,1]`.

**Geometric Interpretation**: The function lies below any line segment connecting two points on its graph.

### Effective Domain

Since we allow `f(w) = +∞`, we define:
```
dom(f) = {w ∈ ℝᵈ : f(w) < +∞}
```

### Indicator Functions

A powerful tool for handling constraints:
```
I_C(w) = {0 if w ∈ C, +∞ if w ∉ C}
```

**Key Property**: 
```
min_{w∈C} f(w) = min_{w∈ℝᵈ} f(w) + I_C(w)
```

This allows treating constrained optimization as unconstrained.

### Operations Preserving Convexity

1. **Non-negative combinations**: If `f, g` convex, then `αf + βg` convex for `α, β ≥ 0`
2. **Pointwise maximum**: `max{f₁, f₂, ...}` is convex if each `fᵢ` is convex
3. **Composition with affine functions**: If `f` convex and `A` linear, then `f(Aw + b)` is convex

### First-Order Characterization

**Theorem**: If `f` is differentiable, then `f` is convex if and only if:
```
f(u) ≥ f(w) + ⟨∇f(w), u - w⟩ ∀w, u
```

**Geometric Interpretation**: The function lies above all its tangent planes.

### Critical Points Are Global Minima

**Fundamental Theorem**: For convex functions, every critical point is a global minimum:
```
∇f(w*) = 0 ⟹ w* ∈ argmin f
```

This is why convex optimization is "easy" - finding critical points solves the global optimization problem.

---

## Strongly Convex Functions

### Definition

A function `f` is **μ-strongly convex** if:
```
f(λw + (1-λ)u) ≤ λf(w) + (1-λ)f(u) - (μ/2)λ(1-λ)‖w - u‖²
```

**Alternative Definition**: `f(w) - (μ/2)‖w‖²` is convex.

### Properties

1. **Unique Global Minimum**: If `f` is μ-strongly convex, then it has a unique global minimum
2. **Quadratic Lower Bound**: At any point w:
   ```
   f(u) ≥ f(w) + ⟨∇f(w), u - w⟩ + (μ/2)‖u - w‖²
   ```

### Second-Order Characterization

If `f` is twice differentiable, then `f` is μ-strongly convex if and only if:
```
∇²f(w) ⪰ μI ∀w
```

The smallest eigenvalue of the Hessian is at least μ everywhere.

### Examples

1. **Quadratic**: `f(w) = (1/2)w^T Qw` is μ-strongly convex with `μ = λ_min(Q)`
2. **Ridge Regression**: `(1/2)‖Xw - y‖² + (λ/2)‖w‖²` is λ-strongly convex
3. **Regularized Logistic**: `Σᵢ log(1 + exp(-yᵢw^T xᵢ)) + (λ/2)‖w‖²` is λ-strongly convex

---

## Smooth Functions and Lipschitz Continuity

### L-Smooth Functions

A differentiable function `f` is **L-smooth** if its gradient is Lipschitz continuous:
```
‖∇f(w) - ∇f(u)‖ ≤ L‖w - u‖ ∀w, u
```

### Descent Lemma

**Key Theorem**: If `f` is convex and L-smooth, then:
```
f(u) ≤ f(w) + ⟨∇f(w), u - w⟩ + (L/2)‖u - w‖²
```

**Geometric Interpretation**: The function lies below a quadratic upper bound at every point.

### Second-Order Characterization

If `f` is twice differentiable, then `f` is L-smooth if and only if:
```
∇²f(w) ⪯ LI ∀w
```

### Condition Number

For functions that are both μ-strongly convex and L-smooth:
```
κ = L/μ
```

**Interpretation**:
- `κ ≈ 1`: Well-conditioned (easy to optimize)
- `κ >> 1`: Ill-conditioned (hard to optimize)

The condition number determines convergence rates of optimization algorithms.

---

## Gradient Descent Algorithm

### Algorithm Definition

**Input**: Starting point `w₀ ∈ ℝᵈ`, step size `γ > 0`

**Iteration**: For `k = 0, 1, 2, ...`:
```
w_{k+1} = w_k - γ∇f(w_k)
```

### Key Properties

1. **Only one parameter**: Step size γ (learning rate)
2. **Any starting point**: Can initialize anywhere for convex functions
3. **Simple update**: Just matrix-vector multiplication

### Step Size Selection

The step size must satisfy:
```
0 < γ < 2/L
```
where L is the Lipschitz constant of the gradient.

**Intuition**: 
- `γ` too large → Divergence (overshooting)
- `γ` too small → Very slow convergence
- `γ = 1/L` → Common practical choice

### Geometric Interpretation

At each iteration:
1. Compute gradient `∇f(w_k)` (steepest ascent direction)
2. Move in opposite direction: `-∇f(w_k)` (steepest descent)
3. Step size `γ` controls how far to move

---

## Convergence Analysis

### Convex and L-Smooth Functions

**Theorem**: Let `f` be convex and L-smooth. Choose `γ ≤ 1/L`. Then:
```
f(w_k) - f* ≤ (L‖w₀ - w*‖²)/(2k)
```

**Properties**:
1. **Sublinear convergence**: `O(1/k)` rate
2. **Iteration complexity**: `K(ε) = O(1/ε)` to achieve `f(w_k) - f* ≤ ε`
3. **Dependence on initialization**: Constant depends on `‖w₀ - w*‖²`

### Strongly Convex and L-Smooth Functions

**Theorem**: Let `f` be μ-strongly convex and L-smooth. Choose `γ ≤ 2/(μ + L)`. Then:
```
f(w_k) - f* ≤ C(1 - 2μγ)^k
```

where `C` depends on the initialization.

**Properties**:
1. **Linear convergence**: Exponential decay
2. **Convergence rate**: `ρ = 1 - 2μγ < 1`
3. **Optimal step size**: `γ* = 2/(μ + L)`

### Optimal Convergence Rate

With optimal step size `γ* = 2/(μ + L)`:
```
f(w_k) - f* ≤ C((κ-1)/(κ+1))^k
```

where `κ = L/μ` is the condition number.

**Key Insights**:
- Better conditioning (`κ ≈ 1`) → Faster convergence
- Poor conditioning (`κ >> 1`) → Slow convergence
- Rate `(κ-1)/(κ+1) ≈ 1 - 2/κ` for large κ

### Comparison of Convergence Rates

| Function Class | Rate | To achieve `ε` accuracy |
|----------------|------|------------------------|
| Convex + L-smooth | `O(1/k)` | `O(1/ε)` iterations |
| μ-strongly convex + L-smooth | `O(((κ-1)/(κ+1))^k)` | `O(κ log(1/ε))` iterations |

---

## Applications

### Ridge Regression

**Objective**: 
```
f(w) = (1/2n)‖Xw - y‖² + (λ/2)‖w‖²
```

**Properties**:
- **Convex**: Quadratic function
- **λ-strongly convex**: Due to regularization term
- **L-smooth**: `L = λ_max(X^T X)/n + λ`

**Gradient**:
```
∇f(w) = (1/n)X^T(Xw - y) + λw
```

### Logistic Regression

**Objective**:
```
f(w) = (1/n)Σᵢ log(1 + exp(-yᵢw^T xᵢ)) + (λ/2)‖w‖²
```

**Properties**:
- **Convex**: Logistic loss is convex
- **λ-strongly convex**: With regularization
- **L-smooth**: Can compute L explicitly

**Advantage**: No closed-form solution, but gradient descent converges globally.

### Why Regularization Helps Optimization

1. **Statistical**: Better generalization, prevents overfitting
2. **Numerical**: 
   - Makes problem strongly convex
   - Improves condition number
   - Faster convergence

This is a **win-win**: Better statistical properties AND easier optimization.

---

## Summary and Key Takeaways

### Function Classes Hierarchy

```
Convex ⊃ Strongly Convex ⊃ Quadratic
   ↓         ↓                ↓
O(1/k)   O(ρ^k)         Exact solution
```

### Algorithm Performance

| Property | Convex | Strongly Convex | Quadratic |
|----------|--------|-----------------|-----------|
| **Convergence** | Sublinear | Linear | Finite |
| **Rate** | `1/k` | `((κ-1)/(κ+1))^k` | 1 step |
| **Dependence** | Problem dimension | Condition number | Matrix structure |

### Practical Guidelines

1. **Always use regularization** when possible:
   - Improves generalization
   - Makes optimization easier
   - Provides strong convexity

2. **Step size selection**:
   - `γ = 1/L` is often a good choice
   - Can estimate L from largest eigenvalue
   - Adaptive methods (next lecture) can help

3. **When to use gradient descent**:
   - Large-scale problems
   - When second-order methods are too expensive
   - Non-smooth problems (with modifications)

### Coming Next

- **Acceleration**: Nesterov's method, momentum
- **Adaptive methods**: AdaGrad, Adam
- **Stochastic methods**: SGD, mini-batch methods
- **Non-smooth optimization**: Subgradients, proximal methods