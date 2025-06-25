# Online Learning and Regularization: From Follow-the-Leader to Mirror Descent

## Table of Contents
1. [Online Learning Framework](#online-learning-framework)
2. [Regret Analysis and Sequential Risk](#regret-analysis-and-sequential-risk)
3. [Follow-the-Leader Algorithm](#follow-the-leader-algorithm)
4. [Strong Convexity and Stability](#strong-convexity-and-stability)
5. [The Instability Problem](#the-instability-problem)
6. [Regularization Strategy](#regularization-strategy)
7. [Follow-the-Regularized-Leader (FTRL)](#follow-the-regularized-leader-ftrl)
8. [Mirror Descent Framework](#mirror-descent-framework)
9. [Bregman Divergences](#bregman-divergences)
10. [Convex Conjugates and Duality](#convex-conjugates-and-duality)
11. [Specific Algorithms](#specific-algorithms)
12. [Regret Analysis for FTRL](#regret-analysis-for-ftrl)
13. [Geometry and Regularizer Choice](#geometry-and-regularizer-choice)
14. [Lower Bounds and Optimality](#lower-bounds-and-optimality)

---

## Online Learning Framework

### The Sequential Decision Problem

**Setting**: At each time step `t = 1, 2, ..., T`:
1. Algorithm chooses predictor `h_t` from hypothesis class `H`
2. Environment reveals data point `(x_t, y_t)`
3. Algorithm suffers loss `ℓ_t(h_t) = ℓ(y_t, h_t(x_t))`

**Key Difference from Statistical Learning**:
- **No probabilistic assumptions** on data generation
- **Adversarial setting**: Environment can be worst-case
- **Sequential**: Decisions made online without seeing future data

### Regret Definition

**Sequential Risk** of algorithm after `T` rounds:
```
R_T^{alg} = Σ_{t=1}^T ℓ_t(h_t)
```

**Sequential Risk** of any fixed predictor `h ∈ H`:
```
R_T^h = Σ_{t=1}^T ℓ_t(h)
```

**Regret** (fundamental performance measure):
```
Regret_T = R_T^{alg} - min_{h∈H} R_T^h = Σ_{t=1}^T ℓ_t(h_t) - min_{h∈H} Σ_{t=1}^T ℓ_t(h)
```

### Learning Goal

**Sublinear Regret**: `Regret_T = o(T)`

**Interpretation**: Average regret `Regret_T/T → 0` as `T → ∞`

This means the algorithm's average performance converges to that of the best fixed predictor in hindsight.

---

## Regret Analysis and Sequential Risk

### Why Regret Makes Sense

**Comparison with Statistical Learning**:
- Statistical: Compare expected risk of learned model vs. best model
- Online: Compare sequential risk of algorithm vs. best fixed model

**No-Regret Property**: If `Regret_T = o(T)`, then:
```
(1/T) Σ_{t=1}^T ℓ_t(h_t) → (1/T) min_{h∈H} Σ_{t=1}^T ℓ_t(h)
```

### Fundamental Requirements

1. **Bounded losses**: `|ℓ_t(h)| ≤ L` for all `t, h`
   - Without this, regret could grow arbitrarily fast
   
2. **Convex loss functions**: Enable gradient-based analysis
   - Most results require convexity for tractable analysis

---

## Follow-the-Leader Algorithm

### Algorithm Description

**Parameter Space**: `Θ ⊆ ℝ^d` (bounded, closed, non-empty)

**Losses**: `ℓ_t: Θ → ℝ` (convex functions)

**Update Rule**:
```
w_{t+1} = argmin_{w∈Θ} Σ_{s=1}^t ℓ_s(w)
```

**Intuition**: At each step, choose the parameter that minimizes cumulative loss on all past data.

### Regret Decomposition Lemma

**Key Identity**:
```
Regret_T = Σ_{t=1}^T [L_{t-1}(w_t) - L_{t-1}(u)]
```

where:
- `L_t(w) = Σ_{s=1}^t ℓ_s(w)` (cumulative loss)
- `u ∈ Θ` is any comparator
- `w_t` minimizes `L_{t-1}`

**Proof Sketch**:
```
Regret_T = Σ_{t=1}^T ℓ_t(w_t) - min_u Σ_{t=1}^T ℓ_t(u)
        = Σ_{t=1}^T [ℓ_t(w_t) - ℓ_t(u)]
        = Σ_{t=1}^T [L_t(w_t) - L_{t-1}(w_t) - L_t(u) + L_{t-1}(u)]
        = Σ_{t=1}^T [L_{t-1}(w_t) - L_{t-1}(u)]  [using w_{t+1} minimizes L_t]
```

---

## Strong Convexity and Stability

### Strong Convexity Definition

**Definition**: Function `f: Θ → ℝ` is `μ`-strongly convex w.r.t. norm `‖·‖` if:
```
f(y) ≥ f(x) + ⟨∇f(x), y-x⟩ + (μ/2)‖y-x‖²
```

**Equivalent Characterizations** (when twice differentiable):
- Hessian condition: `∇²f(x) ⪰ μI` for all `x`
- Second-order: `f(x) - (μ/2)‖x‖²` is convex

### First-Order Optimality Conditions

**Unconstrained**: If `x*` minimizes `f`, then `∇f(x*) = 0`

**Constrained**: If `x*` minimizes `f` over convex set `C`, then:
```
⟨∇f(x*), x - x*⟩ ≥ 0  for all x ∈ C
```

**Geometric Interpretation**: No descent direction exists within the feasible region.

### Regret Bound for Strongly Convex Losses

**Theorem**: If each `ℓ_t` is `μ`-strongly convex and `G`-Lipschitz, then:
```
Regret_T ≤ (G²/μ) Σ_{t=1}^T (1/t) = O(G²log(T)/μ)
```

**Proof Strategy**:
1. Use strong convexity: `L_t(u) ≥ L_t(w_{t+1}) + ⟨∇L_t(w_{t+1}), u - w_{t+1}⟩ + (μt/2)‖u - w_{t+1}‖²`
2. Apply first-order optimality: `⟨∇L_t(w_{t+1}), u - w_{t+1}⟩ ≥ 0`
3. Use Lipschitz property to bound `‖∇L_t(w_{t+1}) - ∇L_{t-1}(w_{t+1})‖ ≤ G`
4. Combine with regret decomposition lemma

---

## The Instability Problem

### Example: Linear Losses on `[-1,1]`

**Setup**:
- Parameter space: `Θ = [-1, 1]`
- Losses: `ℓ_t(w) = a_t · w` (linear functions)
- Adversarial sequence: `a_1 = -1, a_2 = +1, a_3 = -1, a_4 = +1, ...`

**Algorithm Behavior**:
- `t=1`: `w_1 = 0` (initial), loss = 0
- `t=2`: `w_2 = argmin_{w∈[-1,1]} (-w) = +1`, loss = +1
- `t=3`: `w_3 = argmin_{w∈[-1,1]} (-w + w) = +1`, loss = -1
- `t=4`: `w_4 = argmin_{w∈[-1,1]} (-w + w - w) = +1`, loss = +1
- Pattern continues...

**Result**: 
- Algorithm suffers loss ≈ 1 at each step
- Best fixed predictor: `w* = 0` suffers loss = 0 at each step
- **Linear regret**: `Regret_T ≈ T`

### Root Cause: Lack of Curvature

**Problem**: Linear functions have no curvature (Hessian = 0)
- Small changes in data cause large changes in optimal solution
- Algorithm oscillates between extremes
- **Instability** leads to poor performance

---

## Regularization Strategy

### The Regularization Idea

**Motivation**: Add curvature to stabilize the algorithm

**Regularized Objective**:
```
w_{t+1} = argmin_{w∈Θ} [Σ_{s=1}^t ℓ_s(w) + R(w)]
```

where `R(w)` is a **regularizer** (strongly convex function).

### Properties of Good Regularizers

1. **Strongly convex**: Provides curvature for stability
2. **Computationally tractable**: Enables efficient optimization
3. **Geometry-aware**: Matches the structure of parameter space `Θ`

### Follow-the-Regularized-Leader (FTRL)

**Algorithm**:
```
w_{t+1} = argmin_{w∈Θ} [Σ_{s=1}^t ℓ_s(w) + (1/η)R(w)]
```

where `η > 0` is a **learning rate** parameter.

**Interpretation**: 
- `R(w)` acts like a "prior loss" (original sin)
- Balances fitting past data vs. staying close to regularizer minimum
- `1/η` controls regularization strength

---

## Follow-the-Regularized-Leader (FTRL)

### Linearization Trick

**Challenge**: Original losses `ℓ_t` might be complex, making optimization hard.

**Solution**: Replace losses with linear approximations using gradients:
```
ℓ̃_t(w) = ⟨∇ℓ_t(w_t), w⟩
```

**Justification**: For convex functions and any `u ∈ Θ`:
```
ℓ_t(w_t) - ℓ_t(u) ≤ ⟨∇ℓ_t(w_t), w_t - u⟩ = ℓ̃_t(w_t) - ℓ̃_t(u)
```

**Regret Bound**: If we bound regret for linearized losses, we bound regret for original losses.

### Linearized FTRL Algorithm

**Update Rule**:
```
w_{t+1} = argmin_{w∈Θ} [⟨Σ_{s=1}^t ∇ℓ_s(w_s), w⟩ + (1/η)R(w)]
```

**Compact Form**: Let `g_t = ∇ℓ_t(w_t)` and `G_t = Σ_{s=1}^t g_s`, then:
```
w_{t+1} = argmin_{w∈Θ} [⟨G_t, w⟩ + (1/η)R(w)]
```

---

## Mirror Descent Framework

### Constrained Optimization Decomposition

**Key Insight**: Constrained optimization = Unconstrained optimization + Projection

**Theorem**: For strictly convex, differentiable `f` and convex set `C`:
```
argmin_{x∈C} f(x) = Π_C^{Ψ}(argmin_{x∈ℝ^d} f(x))
```

where `Π_C^{Ψ}` is the **Bregman projection** induced by strongly convex function `Ψ`.

### Unconstrained Optimization

**Problem**: `min_{w∈ℝ^d} [⟨G_t, w⟩ + (1/η)R(w)]`

**First-order condition**: `G_t + (1/η)∇R(w) = 0`

**Solution**: `w = -η∇R^*(η G_t)`

where `R^*` is the **convex conjugate** of `R`.

---

## Bregman Divergences

### Definition

**Bregman Divergence** induced by strictly convex, differentiable `Ψ`:
```
D_Ψ(x, y) = Ψ(x) - Ψ(y) - ⟨∇Ψ(y), x - y⟩
```

**Interpretation**: 
- Error in first-order Taylor approximation of `Ψ` around `y`
- Generalized "distance" (not symmetric!)

### Properties

1. **Non-negativity**: `D_Ψ(x, y) ≥ 0` with equality iff `x = y`
2. **Strong convexity relation**: If `Ψ` is `μ`-strongly convex:
   ```
   D_Ψ(x, y) ≥ (μ/2)‖x - y‖²
   ```
3. **Linearity invariance**: `D_{Ψ+L}(x, y) = D_Ψ(x, y)` for any linear `L`

### Examples

1. **Euclidean**: `Ψ(x) = (1/2)‖x‖²` → `D_Ψ(x, y) = (1/2)‖x - y‖²`
2. **Entropic**: `Ψ(x) = Σᵢ xᵢ log xᵢ` → `D_Ψ(x, y) = KL(x, y)` (KL divergence)

### Bregman Projection

**Definition**:
```
Π_C^Ψ(x) = argmin_{y∈C} D_Ψ(y, x)
```

**Properties**:
- Always exists and unique (when `Ψ` strictly convex)
- Generalizes Euclidean projection

---

## Convex Conjugates and Duality

### Convex Conjugate Definition

**Definition**: For convex function `f`, its **convex conjugate** is:
```
f^*(θ) = sup_{x∈dom(f)} [⟨θ, x⟩ - f(x)]
```

**Properties**:
1. `f^*` is always convex (even if `f` isn't)
2. **Fenchel-Moreau**: If `f` is closed convex, then `f^{**} = f`
3. **Young's inequality**: `⟨x, θ⟩ ≤ f(x) + f^*(θ)`

### Key Theorem for FTRL

**Theorem**: If `R` is strictly convex, then:
```
argmin_{w∈ℝ^d} [⟨G, w⟩ + (1/η)R(w)] = ∇R^*(η G)
```

**Proof**: 
- First-order condition: `G + (1/η)∇R(w) = 0`
- Rearrange: `∇R(w) = -η G`
- Since `R` strictly convex: `w = (∇R)^{-1}(-η G) = ∇R^*(η G)`

### Mirror Descent Algorithm

**Complete Algorithm**:
1. **Gradient accumulation**: `G_t = Σ_{s=1}^t g_s`
2. **Mirror mapping**: `w'_{t+1} = ∇R^*(-η G_t)`
3. **Bregman projection**: `w_{t+1} = Π_Θ^R(w'_{t+1})`

**Key Insight**: Algorithm only needs to maintain `G_t` (single vector), not entire loss history!

---

## Specific Algorithms

### Online Gradient Descent (OGD)

**Setup**:
- Domain: `Θ ⊆ ℝ^d` (bounded convex set)
- Regularizer: `R(w) = (1/2)‖w‖²` (Euclidean)
- Learning rate: `η`

**Convex conjugate**: `R^*(θ) = (1/2)‖θ‖²`

**Gradient**: `∇R^*(θ) = θ`

**Algorithm**:
```
G_t = Σ_{s=1}^t g_s
w'_{t+1} = -η G_t
w_{t+1} = Π_Θ(w'_{t+1})  [Euclidean projection]
```

**Incremental form**:
```
w_{t+1} = Π_Θ(w_t - η g_t)
```

### Exponentiated Gradient (EG)

**Setup**:
- Domain: `Θ = Δ_d = {x ∈ ℝ^d : xᵢ ≥ 0, Σᵢ xᵢ = 1}` (probability simplex)
- Regularizer: `R(w) = Σᵢ wᵢ log wᵢ` (negative entropy)
- Learning rate: `η`

**Convex conjugate**: `R^*(θ) = log(Σᵢ exp(θᵢ))`

**Gradient**: `∇R^*(θ) = exp(θ) / Σⱼ exp(θⱼ)` (softmax)

**Algorithm**:
```
G_t = Σ_{s=1}^t g_s
w_{t+1} = exp(-η G_t) / Σⱼ exp(-η G_{t,j})
```

**Key Property**: Automatic projection! The softmax output is already in the simplex.

**Incremental form**:
```
w_{t+1,i} = w_{t,i} exp(-η g_{t,i}) / Z_t
```
where `Z_t` is the normalization constant.

---

## Regret Analysis for FTRL

### Regret Decomposition

**Theorem**: For FTRL with regularizer `R`, the regret satisfies:
```
Regret_T ≤ (1/η)[R(u) - R(w_1)] + η Σ_{t=1}^T D_R(w_{t+1}, w_t)
```

for any comparator `u ∈ Θ`.

### Proof Strategy

**Step 1**: Inductive regret bound
```
Σ_{t=1}^T ⟨g_t, w_t - u⟩ ≤ (1/η)[R(u) - R(w_1)] + (1/η)Σ_{t=1}^T [R(w_t) - R(w_{t+1})]
```

**Step 2**: Use first-order optimality conditions and strong convexity of regularizer

**Step 3**: Control stability term `Σ_{t=1}^T D_R(w_{t+1}, w_t)`

### Stability Analysis

**Key Lemma**: If `R` is `μ`-strongly convex and losses are `G`-Lipschitz:
```
D_R(w_{t+1}, w_t) ≤ η²G²/(2μ)
```

**Intuition**: Strong convexity bounds how much parameters can change between steps.

### Final Bound

**Theorem**: Under above conditions:
```
Regret_T ≤ (1/η)[R(u) - R(w_1)] + (η T G²)/(2μ)
```

**Optimal learning rate**: `η* = √(2μ[R(u) - R(w_1)])/(G√T)`

**Optimal regret**: `Regret_T ≤ G√(2T[R(u) - R(w_1)]/μ)`

**Rate**: `O(√T)` regret (vs. `O(log T)` for strongly convex losses)

---

## Geometry and Regularizer Choice

### The Geometry Principle

**Key Insight**: Regularizer should match the natural geometry of the domain.

**Diameter Concept**: For regularizer `R` and domain `Θ`:
```
D_R(Θ) = max_{x,y∈Θ} D_R(x, y)
```

This measures the "size" of `Θ` according to `R`.

### Dual Norms and Lipschitz Constants

**Theorem**: If `ℓ` is differentiable convex with `‖∇ℓ(w)‖_* ≤ G` (dual norm), then `ℓ` is `G`-Lipschitz w.r.t. primal norm `‖·‖`.

**Application**: For bounded gradients `‖∇ℓ_t(w)‖_∞ ≤ G`:
- `ℓ_t` is `G√d`-Lipschitz w.r.t. `‖·‖_2` (Euclidean)
- `ℓ_t` is `G`-Lipschitz w.r.t. `‖·‖_1`

### OGD on Euclidean Ball

**Domain**: `Θ = {w : ‖w‖_2 ≤ D}`
**Regularizer**: `R(w) = (1/2)‖w‖²` (1-strongly convex w.r.t. `‖·‖_2`)
**Diameter**: `D_R(Θ) = D²`
**Lipschitz**: `G√d` (if gradient coordinates bounded by `G`)

**Regret bound**:
```
Regret_T ≤ D G√d √T
```

**Dimension dependence**: `√d`

### EG on Probability Simplex  

**Domain**: `Θ = Δ_d` (probability simplex)
**Regularizer**: `R(w) = Σᵢ wᵢ log wᵢ` (1-strongly convex w.r.t. `‖·‖_1`)
**Diameter**: `D_R(Θ) = log d`
**Lipschitz**: `G` (w.r.t. `‖·‖_1`)

**Regret bound**:
```
Regret_T ≤ G√(log d) √T  
```

**Dimension dependence**: `√log d`

### Comparison: Why Geometry Matters

**Using OGD on simplex**:
- Euclidean diameter of simplex: `O(1)`
- But Lipschitz constant w.r.t. `‖·‖_2`: `O(√d)`
- **Regret**: `O(√(d T))`

**Using EG on simplex**:
- Entropic diameter: `O(log d)`  
- Lipschitz constant w.r.t. `‖·‖_1`: `O(1)`
- **Regret**: `O(√(log d · T))`

**Exponential improvement**: From `√d` to `√log d` dependence!

---

## Lower Bounds and Optimality

### General Lower Bound

**Theorem**: For any online algorithm on domain `Θ` with Lipschitz losses:
```
max_{ℓ₁,...,ℓₜ} Regret_T ≥ Ω(√T)
```

**Proof idea**: Adversary can force any algorithm to have `Ω(√T)` regret by constructing worst-case loss sequence.

### Geometric Lower Bounds

**Euclidean domains**: 
```
Regret_T ≥ Ω(D G √T)
```
where `D` is Euclidean diameter and `G` is Lipschitz constant.

**Simplex domains**:
```
Regret_T ≥ Ω(G √(log d · T))
```

### Optimality

**Consequence**: Our algorithms achieve optimal rates:
- **OGD**: Optimal for Euclidean domains
- **EG**: Optimal for simplex domains

**General principle**: Matching regularizer to domain geometry yields optimal algorithms.

---

## Summary and Key Insights

### Main Results

1. **Follow-the-Leader**: `O(log T)` regret for strongly convex losses, but unstable for general convex losses

2. **Follow-the-Regularized-Leader**: `O(√T)` regret for general convex losses via regularization

3. **Mirror Descent**: Unified framework encompassing many algorithms

4. **Geometry matters**: Proper regularizer choice crucial for optimal performance

### Key Techniques

1. **Regret decomposition**: Breaking regret into manageable pieces
2. **Linearization**: Replacing complex losses with linear approximations  
3. **Regularization**: Adding curvature for stability
4. **Bregman divergences**: Generalized distance measures
5. **Convex duality**: Enabling closed-form solutions

### Practical Guidelines

1. **Euclidean domains**: Use Online Gradient Descent
2. **Probability simplex**: Use Exponentiated Gradient
3. **General principle**: Match regularizer to domain geometry
4. **Learning rate**: `η = O(1/√T)` typically optimal

### Extensions and Open Questions

1. **Adaptive methods**: Adjusting to problem difficulty
2. **Non-convex losses**: Extending beyond convex optimization
3. **High-dimensional settings**: Exploiting sparsity and structure
4. **Computational efficiency**: Scaling to massive problems

This framework provides the foundation for modern online learning, with applications ranging from online advertising to robotics to game playing.