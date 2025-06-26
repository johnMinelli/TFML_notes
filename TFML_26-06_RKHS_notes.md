# Reproducing Kernel Hilbert Spaces: Theory and Applications

## Table of Contents
1. [Introduction and Motivation](#introduction-and-motivation)
2. [Mathematical Prerequisites](#mathematical-prerequisites)
3. [Hilbert Spaces: Fundamental Properties](#hilbert-spaces-fundamental-properties)
4. [Reproducing Kernel Hilbert Spaces (RKHS)](#reproducing-kernel-hilbert-spaces-rkhs)
5. [The Reproducing Property](#the-reproducing-property)
6. [Examples of RKHS](#examples-of-rkhs)
7. [Kernel Construction and Characterization](#kernel-construction-and-characterization)
8. [Properties of Kernels](#properties-of-kernels)
9. [Feature Maps and the Kernel Trick](#feature-maps-and-the-kernel-trick)
10. [Moore-Aronszajn Theorem](#moore-aronszajn-theorem)
11. [Regularization and Kernel Methods](#regularization-and-kernel-methods)
12. [The Representer Theorem](#the-representer-theorem)
13. [Kernel Ridge Regression](#kernel-ridge-regression)
14. [Advanced Topics](#advanced-topics)

---

## Introduction and Motivation

### Machine Learning Context

**Goal of Machine Learning**: Given a dataset of feature-label pairs `{(x_i, y_i)}_{i=1}^n`, find a function `f: X → Y` that generalizes well to new data.

**Function Space Approach**: Instead of parametric models, search in a function space `H` of candidate functions.

**Key Requirements for H**:
1. **Vector Space Structure**: Allow linear combinations of functions
2. **Topology/Norm**: Measure "size" and "closeness" of functions  
3. **Completeness**: Limit points of convergent sequences exist in `H`
4. **Evaluation Property**: Control pointwise evaluation `f(x)` using the norm

**Why RKHS?**: Reproducing Kernel Hilbert Spaces satisfy all these requirements with additional nice properties that make them computationally tractable.

---

## Mathematical Prerequisites

### Topology and Convergence

**Topological Space**: Pair `(X, τ)` where `τ` is a collection of "open sets" satisfying:
- `∅, X ∈ τ`
- Arbitrary unions of open sets are open
- Finite intersections of open sets are open

**Convergence**: Sequence `{x_n}` converges to `x` if every open neighborhood of `x` contains all but finitely many `x_n`.

**Continuity**: Function `f: X → Y` is continuous if preimage of every open set is open.

### Normed Vector Spaces

**Vector Space**: Set `V` with operations:
- **Addition**: `u + v ∈ V` for `u, v ∈ V`
- **Scalar Multiplication**: `αu ∈ V` for `α ∈ ℝ, u ∈ V`
- **Axioms**: Associativity, commutativity, distributivity, identity elements

**Norm**: Function `‖·‖: V → ℝ₊` satisfying:
- `‖v‖ = 0 ⟺ v = 0` (definiteness)
- `‖αv‖ = |α|‖v‖` (homogeneity)  
- `‖u + v‖ ≤ ‖u‖ + ‖v‖` (triangle inequality)

**Induced Topology**: Norm induces metric `d(u,v) = ‖u - v‖`, which induces topology.

### Measure Theory Basics

**Measure Space**: Triple `(Ω, Σ, μ)` where:
- `Ω` is a set
- `Σ` is a σ-algebra of subsets of `Ω`
- `μ: Σ → [0,∞]` is a measure

**L² Space**: For measure space `(X, μ)`:
```
L²(X, μ) = {f: X → ℝ : ∫_X |f(x)|² μ(dx) < ∞}
```
modulo functions equal almost everywhere.

**Inner Product in L²**:
```
⟨f, g⟩_{L²} = ∫_X f(x)g(x) μ(dx)
```

---

## Hilbert Spaces: Fundamental Properties

### Definition and Structure

**Inner Product Space**: Vector space `H` with inner product `⟨·,·⟩: H × H → ℝ` satisfying:
- **Linearity**: `⟨αu + βv, w⟩ = α⟨u,w⟩ + β⟨v,w⟩`
- **Symmetry**: `⟨u,v⟩ = ⟨v,u⟩`
- **Positive Definiteness**: `⟨u,u⟩ ≥ 0`, with equality iff `u = 0`

**Induced Norm**: `‖u‖ = √⟨u,u⟩`

**Hilbert Space**: Complete inner product space (every Cauchy sequence converges).

### Key Properties of Hilbert Spaces

#### 1. Orthonormal Bases

**Orthogonal**: `u ⊥ v` if `⟨u,v⟩ = 0`

**Orthonormal System**: Collection `{e_i}` where:
- `⟨e_i, e_j⟩ = δ_{ij}` (Kronecker delta)

**Orthonormal Basis**: Maximal orthonormal system. Every `f ∈ H` can be written as:
```
f = Σᵢ ⟨f, e_i⟩ e_i
```
with `‖f‖² = Σᵢ |⟨f, e_i⟩|²` (Parseval's identity).

#### 2. Orthogonal Complements and Projection

**Orthogonal Complement**: For subspace `M ⊆ H`:
```
M⊥ = {f ∈ H : ⟨f,g⟩ = 0 for all g ∈ M}
```

**Orthogonal Decomposition**: `H = M ⊕ M⊥` for closed subspace `M`.

**Projection Theorem**: For closed subspace `M` and `f ∈ H`, there exists unique `P_M f ∈ M` such that:
```
‖f - P_M f‖ = min_{g∈M} ‖f - g‖
```

**Pythagorean Theorem**: If `u ⊥ v`, then `‖u + v‖² = ‖u‖² + ‖v‖²`.

#### 3. Riesz Representation Theorem

**Linear Functional**: Map `φ: H → ℝ` that is linear.

**Bounded Linear Functional**: Linear functional `φ` with `|φ(f)| ≤ C‖f‖` for some `C > 0`.

**Riesz Representation**: For every bounded linear functional `φ`, there exists unique `g ∈ H` such that:
```
φ(f) = ⟨f, g⟩ for all f ∈ H
```

**Operator Norm**: `‖φ‖ = sup_{‖f‖≤1} |φ(f)| = ‖g‖`.

#### 4. Weak Topology and Compactness

**Weak Convergence**: `f_n ⇀ f` if `⟨f_n, g⟩ → ⟨f, g⟩` for all `g ∈ H`.

**Strong Convergence**: `f_n → f` if `‖f_n - f‖ → 0`.

**Key Fact**: Strong convergence implies weak convergence.

**Banach-Alaoglu Theorem**: Closed unit ball is compact in weak topology.

**Practical Importance**: Bounded sequences have weakly convergent subsequences.

---

## Reproducing Kernel Hilbert Spaces (RKHS)

### Definition

**RKHS**: Hilbert space `H` of functions `f: X → ℝ` such that:

1. **Function Space**: Elements of `H` are real-valued functions on set `X`
2. **Hilbert Space Structure**: `H` is complete inner product space
3. **Reproducing Property**: For each `x ∈ X`, evaluation functional `δ_x: H → ℝ` defined by `δ_x(f) = f(x)` is continuous

### The Reproducing Property

**Equivalent Formulations**:

**Form 1** (Point Evaluation): For each `x ∈ X`, there exists `C_x > 0` such that:
```
|f(x)| ≤ C_x ‖f‖_H for all f ∈ H
```

**Form 2** (Riesz Representation): For each `x ∈ X`, there exists unique `K_x ∈ H` such that:
```
f(x) = ⟨f, K_x⟩_H for all f ∈ H
```

**Form 3** (Sequential): If `f_n → f` in `H`, then `f_n(x) → f(x)` for all `x ∈ X`.

### The Reproducing Kernel

**Definition**: Function `K: X × X → ℝ` defined by:
```
K(x, x') = ⟨K_x, K_{x'}⟩_H = K_x(x')
```

**Key Property**: 
```
f(x) = ⟨f, K_x⟩_H = ⟨f, K(·,x)⟩_H
```

**Reproducing Property**: 
```
K(x, x') = ⟨K(·,x), K(·,x')⟩_H
```

---

## Examples of RKHS

### Example 1: Sobolev Spaces

**Definition**: For `s > 0`, the Sobolev space `H^s(ℝ^d)` consists of functions `f ∈ L²(ℝ^d)` such that:
```
‖f‖²_{H^s} = ∫_{ℝ^d} (1 + |ω|²)^s |ℱf(ω)|² dω < ∞
```
where `ℱf` is the Fourier transform of `f`.

**Fourier Transform**: 
```
ℱf(ω) = ∫_{ℝ^d} f(x) e^{-i⟨ω,x⟩} dx
```

**Inner Product**:
```
⟨f,g⟩_{H^s} = ∫_{ℝ^d} (1 + |ω|²)^s ℱf(ω) overline{ℱg(ω)} dω
```

**Embedding Theorem**: If `s > d/2`, then `H^s(ℝ^d)` embeds continuously into space of continuous bounded functions.

**Reproducing Kernel**: For `s > d/2`:
```
K(x,x') = ∫_{ℝ^d} (1 + |ω|²)^{-s} e^{i⟨ω,x-x'⟩} dω
```

**Special Case**: When `s = d/2 + 1/2`, kernel has explicit form involving Bessel functions.

### Example 2: Gaussian Kernel RKHS

**Gaussian Kernel**:
```
K(x,x') = exp(-‖x-x'‖²/(2σ²))
```

**Properties**:
- **Smoothness**: Infinitely differentiable
- **Universal**: Dense in space of continuous functions on compact sets
- **Translation Invariant**: `K(x,x') = k(x-x')` for `k(t) = exp(-‖t‖²/(2σ²))`

**Fourier Characterization**: Fourier transform of Gaussian is Gaussian:
```
ℱk(ω) = (2πσ²)^{d/2} exp(-σ²‖ω‖²/2)
```

**RKHS Structure**: Functions in RKHS have Fourier transforms decaying faster than any polynomial.

### Example 3: Finite-Dimensional Feature Spaces

**Setup**: Let `φ₁, ..., φₙ: X → ℝ` be basis functions.

**Function Space**: 
```
H = span{φ₁, ..., φₙ} = {f(x) = Σᵢ wᵢ φᵢ(x) : w ∈ ℝⁿ}
```

**Inner Product**: For `f = Σᵢ wᵢ φᵢ` and `g = Σᵢ vᵢ φᵢ`:
```
⟨f,g⟩_H = Σᵢ,ⱼ wᵢ vⱼ Gᵢⱼ
```
where `G` is Gram matrix: `Gᵢⱼ = ⟨φᵢ, φⱼ⟩`.

**Reproducing Kernel**:
```
K(x,x') = Σᵢ,ⱼ φᵢ(x) G⁻¹ᵢⱼ φⱼ(x')
```

**Linear Independence**: If `{φᵢ}` linearly independent, then `G` is invertible and `{φᵢ}` forms orthogonal basis after Gram-Schmidt.

---

## Kernel Construction and Characterization

### Kernel Properties

**Definition**: Function `K: X × X → ℝ` is a **kernel** (positive definite) if:

1. **Symmetry**: `K(x,x') = K(x',x)` for all `x,x' ∈ X`

2. **Positive Semi-Definiteness**: For any finite set `{x₁, ..., xₙ} ⊆ X` and coefficients `c₁, ..., cₙ ∈ ℝ`:
```
Σᵢ,ⱼ cᵢ cⱼ K(xᵢ, xⱼ) ≥ 0
```

**Gram Matrix**: For points `x₁, ..., xₙ`, the `n × n` matrix `K` with entries `Kᵢⱼ = K(xᵢ, xⱼ)` is positive semi-definite.

### Kernel Operations

**Kernel Algebra**: If `K₁, K₂` are kernels, then so are:
- **Linear Combination**: `αK₁ + βK₂` for `α, β ≥ 0`
- **Pointwise Product**: `(K₁ · K₂)(x,x') = K₁(x,x') K₂(x,x')`
- **Tensor Product**: `(K₁ ⊗ K₂)((x₁,x₂), (x'₁,x'₂)) = K₁(x₁,x'₁) K₂(x₂,x'₂)`

**Composition with Functions**: If `φ: Y → X` and `K` is kernel on `X`, then `K'(y,y') = K(φ(y), φ(y'))` is kernel on `Y`.

### Feature Map Perspective

**Feature Map**: Function `Φ: X → F` into inner product space `F`.

**Induced Kernel**: 
```
K(x,x') = ⟨Φ(x), Φ(x')⟩_F
```

**Key Insight**: Every kernel arises this way (Moore-Aronszajn theorem).

**Kernel Trick**: Instead of working with explicit feature map `Φ`, work directly with kernel `K`.

---

## Properties of Kernels

### Continuity

**Pointwise Continuity**: Kernel `K` is continuous if `K(·,x)` is continuous for each fixed `x ∈ X`.

**Uniform Continuity**: On compact sets, pointwise continuity implies uniform continuity.

**RKHS Continuity**: If `K` is continuous, then every `f ∈ H` is continuous.

**Characterization**: `H ⊆ C(X)` (continuous functions) iff `K` is continuous.

### Separability

**Separable Hilbert Space**: Contains countable dense subset.

**Characterization**: If `X` is separable topological space and `K` is continuous, then RKHS `H` is separable.

**Practical Importance**: Separable spaces allow countable approximations.

### Integrability

**Setup**: Let `μ` be probability measure on `X`.

**Square Integrability**: `H ⊆ L²(X,μ)` iff
```
∫_X K(x,x) μ(dx) < ∞
```

**Integral Operator**: Define `T_K: L²(X,μ) → L²(X,μ)` by:
```
(T_K f)(x) = ∫_X K(x,x') f(x') μ(dx')
```

**Hilbert-Schmidt**: If `∫∫ K(x,x')² μ(dx)μ(dx') < ∞`, then `T_K` is Hilbert-Schmidt operator.

**Mercer's Theorem**: If `K` is continuous and `T_K` is trace-class, then:
```
K(x,x') = Σᵢ λᵢ φᵢ(x) φᵢ(x')
```
where `{λᵢ, φᵢ}` are eigenvalues/eigenfunctions of `T_K`.

---

## Feature Maps and the Kernel Trick

### Canonical Feature Map

**Construction**: For RKHS `H` with kernel `K`, define canonical feature map:
```
Φ: X → H
x ↦ K(·,x)
```

**Verification**: 
```
⟨Φ(x), Φ(x')⟩_H = ⟨K(·,x), K(·,x')⟩_H = K(x,x')
```

**Universal Property**: Any other feature map `Ψ: X → F` with `K(x,x') = ⟨Ψ(x), Ψ(x')⟩_F` factors through `Φ`.

### Non-Uniqueness of Feature Maps

**Isometric Feature Maps**: If `U: H → F` is isometry, then `Ψ(x) = U(Φ(x))` gives same kernel.

**Characterization**: Two feature maps `Φ₁: X → F₁` and `Φ₂: X → F₂` induce same kernel iff there exists isometry `U: span{Φ₁(X)} → span{Φ₂(X)}` with `Φ₂ = U ∘ Φ₁`.

### The Kernel Trick

**Problem**: Feature space `F` may be high-dimensional or infinite-dimensional.

**Solution**: Work directly with kernel `K(x,x')` instead of explicit features `Φ(x)`.

**Applications**:
- **SVM**: Decision function `f(x) = Σᵢ αᵢ yᵢ K(xᵢ, x) + b`
- **Kernel Ridge Regression**: Solution `f(x) = Σᵢ αᵢ K(xᵢ, x)`
- **Kernel PCA**: Eigendecomposition of kernel matrix

**Computational Advantage**: Avoid explicit computation in high-dimensional space.

---

## Moore-Aronszajn Theorem

### Statement

**Theorem**: There is bijective correspondence between:
1. Reproducing kernel Hilbert spaces of functions on `X`
2. Positive definite kernels on `X × X`

**Construction Direction**: Given kernel `K`, construct RKHS `H_K`.

**Characterization Direction**: Given RKHS `H`, its reproducing kernel is unique.

### Construction of RKHS from Kernel

**Step 1**: Define pre-Hilbert space
```
H₀ = span{K(·,x) : x ∈ X}
```

**Step 2**: Elements of `H₀` have form `f = Σᵢ cᵢ K(·,xᵢ)`

**Step 3**: Define inner product on `H₀`:
```
⟨Σᵢ cᵢ K(·,xᵢ), Σⱼ dⱼ K(·,yⱼ)⟩ = Σᵢ,ⱼ cᵢ dⱼ K(xᵢ, yⱼ)
```

**Step 4**: Verify well-definedness using positive definiteness of `K`

**Step 5**: Complete `H₀` to obtain Hilbert space `H_K`

**Step 6**: Show reproducing property holds in completion

### Uniqueness

**Theorem**: If `H` is RKHS on `X`, its reproducing kernel is unique.

**Proof**: If `K₁, K₂` both reproduce `H`, then for any `f ∈ H`:
```
f(x) = ⟨f, K₁(·,x)⟩ = ⟨f, K₂(·,x)⟩
```
implies `K₁(·,x) = K₂(·,x)` for all `x`.

---

## Regularization and Kernel Methods

### Regularized Risk Minimization

**Setup**: Given training data `{(xᵢ, yᵢ)}ᵢ₌₁ⁿ`, loss function `L`, and RKHS `H`.

**Regularized Problem**:
```
min_{f∈H} [1/n Σᵢ L(yᵢ, f(xᵢ)) + λ‖f‖²_H]
```

**Components**:
- **Empirical Risk**: `1/n Σᵢ L(yᵢ, f(xᵢ))`
- **Regularization**: `λ‖f‖²_H` with `λ > 0`

**Benefits**:
- **Existence**: Regularization ensures solution exists
- **Uniqueness**: Strict convexity gives uniqueness  
- **Finite Representation**: Solution has finite representation

### Convex Loss Functions

**Quadratic Loss**: `L(y, f(x)) = (y - f(x))²`
- **Application**: Regression
- **Properties**: Smooth, strongly convex

**Hinge Loss**: `L(y, f(x)) = max(0, 1 - yf(x))`
- **Application**: SVM classification
- **Properties**: Convex, not differentiable

**Logistic Loss**: `L(y, f(x)) = log(1 + exp(-yf(x)))`
- **Application**: Logistic regression
- **Properties**: Smooth, convex

---

## The Representer Theorem

### Statement

**Theorem**: Consider regularized problem:
```
min_{f∈H} [Σᵢ L(yᵢ, f(xᵢ)) + λ‖f‖²_H]
```
where `L` is convex in second argument and `λ > 0`.

Then minimizer has form:
```
f*(x) = Σᵢ αᵢ K(x, xᵢ)
```
for some coefficients `α₁, ..., αₙ ∈ ℝ`.

### Proof Strategy

**Step 1**: Decompose `f ∈ H` as `f = f_∥ + f_⊥` where:
- `f_∥ ∈ span{K(·,x₁), ..., K(·,xₙ)}`
- `f_⊥ ⊥ span{K(·,x₁), ..., K(·,xₙ)}`

**Step 2**: Show `f(xᵢ) = f_∥(xᵢ)` for all `i` (orthogonal part doesn't affect evaluation)

**Step 3**: Note `‖f‖² = ‖f_∥‖² + ‖f_⊥‖²` (Pythagorean theorem)

**Step 4**: Conclude objective is minimized when `f_⊥ = 0`

**Step 5**: Therefore `f* ∈ span{K(·,x₁), ..., K(·,xₙ)}`

### Computational Implications

**Finite-Dimensional Problem**: Instead of optimizing over infinite-dimensional `H`, optimize over `ℝⁿ`.

**Kernel Matrix**: Let `K ∈ ℝⁿˣⁿ` with `Kᵢⱼ = K(xᵢ, xⱼ)`.

**Reduced Problem**:
```
min_{α∈ℝⁿ} [Σᵢ L(yᵢ, Σⱼ αⱼ K(xᵢ, xⱼ)) + λα^T K α]
```

**Matrix Form**:
```
min_{α∈ℝⁿ} [L(y, Kα) + λα^T K α]
```

---

## Kernel Ridge Regression

### Problem Setup

**Regression**: Predict `y ∈ ℝ` from `x ∈ X`

**Quadratic Loss**: `L(y, f(x)) = (y - f(x))²`

**Regularized Problem**:
```
min_{f∈H} [1/n Σᵢ (yᵢ - f(xᵢ))² + λ‖f‖²_H]
```

### Solution via Representer Theorem

**Finite Representation**: `f*(x) = Σⱼ αⱼ K(x, xⱼ)`

**Objective in Terms of α**:
```
1/n ‖y - Kα‖² + λα^T K α
```

**Optimality Condition**: Setting gradient to zero:
```
∂/∂α [1/n ‖y - Kα‖² + λα^T K α] = 0
```

**Solution**:
```
-2/n K(y - Kα) + 2λKα = 0
⟹ K(y - Kα) = nλKα  
⟹ Ky = K(K + nλI)α
⟹ α = (K + nλI)⁻¹y
```

### Final Predictor

**Explicit Form**:
```
f*(x) = Σᵢ αᵢ K(x, xᵢ) = y^T (K + nλI)⁻¹ K(x,·)
```
where `K(x,·) = [K(x,x₁), ..., K(x,xₙ)]^T`.

**Matrix Interpretation**: 
```
f*(x) = k(x)^T (K + nλI)⁻¹ y
```

### Properties

**Well-Posedness**: Matrix `K + nλI` is always invertible for `λ > 0` since `K` is positive semi-definite.

**Limiting Cases**:
- **λ → 0**: Approaches interpolation (if `K` invertible)
- **λ → ∞**: Approaches zero function

**Computational Complexity**: `O(n³)` for matrix inversion.

---

## Advanced Topics

### Kernel Learning

**Problem**: How to choose appropriate kernel `K`?

**Approaches**:
1. **Multiple Kernel Learning**: Combine kernels `K = Σₖ βₖ Kₖ`
2. **Parametric Kernels**: Learn parameters in kernel family
3. **Deep Kernels**: Use neural networks to learn kernel representations

### Infinite-Dimensional Feature Spaces

**Random Features**: Approximate kernel using random projections
```
K(x,x') ≈ 1/m Σᵢ φ(x; ωᵢ) φ(x'; ωᵢ)
```

**Nyström Method**: Low-rank approximation using subset of data points

### Computational Challenges

**Large-Scale Problems**: Kernel matrix `K ∈ ℝⁿˣⁿ` becomes prohibitive for large `n`.

**Solutions**:
- **Sparse Methods**: Select subset of basis functions
- **Low-Rank Approximations**: Approximate kernel matrix
- **Stochastic Methods**: Sample-based optimization

### Connections to Neural Networks

**Neural Tangent Kernel**: Infinite-width neural networks correspond to specific RKHS.

**Feature Learning vs. Kernel Methods**:
- **RKHS**: Fixed feature space, convex optimization
- **Neural Networks**: Learned features, non-convex optimization

### Theoretical Guarantees

**Approximation Theory**: Universal kernels are dense in continuous functions on compact sets.

**Learning Theory**: 
- **Consistency**: Regularized solutions converge to optimal predictor
- **Rate of Convergence**: Depends on regularity of target function and kernel

**Generalization Bounds**: Control difference between training and test error.

---

## Summary and Key Insights

### Main Theoretical Results

1. **Moore-Aronszajn Theorem**: Bijection between RKHS and positive definite kernels

2. **Representer Theorem**: Solutions to regularized problems have finite representation

3. **Riesz Representation**: Point evaluation functionals have inner product representation

4. **Kernel Characterization**: Positive definiteness characterizes valid kernels

### Practical Benefits

**Computational**: Transform infinite-dimensional optimization to finite-dimensional problem.

**Modularity**: Kernel choice separates domain knowledge from optimization algorithm.

**Flexibility**: Wide variety of kernels for different data types and applications.

**Theory**: Strong mathematical foundation with guarantees.

### Limitations and Extensions

**Scalability**: Cubic complexity in training set size.

**Kernel Choice**: No universal method for selecting optimal kernel.

**Feature Learning**: Fixed feature space vs. learned representations.

**Modern Extensions**: 
- Deep learning connections (Neural Tangent Kernels)
- Gaussian processes (Bayesian perspective)
- Kernel methods for structured data (graphs, sequences, etc.)

This framework provides the mathematical foundation for understanding how kernel methods work and why they are effective for machine learning, bridging functional analysis with practical algorithms.