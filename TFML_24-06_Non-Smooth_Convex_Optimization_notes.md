# Non-Smooth Convex Optimization for Machine Learning

## Table of Contents
1. [Motivation: Non-Differentiable Loss Functions](#motivation-non-differentiable-loss-functions)
2. [The Class of Proper Convex Functions](#the-class-of-proper-convex-functions)
3. [Subdifferential Theory](#subdifferential-theory)
4. [Subdifferential Calculus Rules](#subdifferential-calculus-rules)
5. [Computing Subdifferentials](#computing-subdifferentials)
6. [Subgradient Descent Algorithm](#subgradient-descent-algorithm)
7. [Convergence Analysis of Subgradient Methods](#convergence-analysis-of-subgradient-methods)
8. [Proximal Gradient Algorithms](#proximal-gradient-algorithms)
9. [Proximal Operators](#proximal-operators)
10. [Applications and Examples](#applications-and-examples)

---

## Motivation: Non-Differentiable Loss Functions

### The Limitation of Gradient Descent

Yesterday we saw that gradient descent works well for:
- **Squared loss**: `ℓ(y, ŷ) = (1/2)(y - ŷ)²` - differentiable everywhere
- **Logistic loss**: `ℓ(y, ŷ) = log(1 + exp(-yŷ))` - differentiable everywhere

However, many important loss functions are **non-differentiable**:

### Hinge Loss (SVM Loss)

**Definition**: 
```
ℓ(y, ŷ) = max(0, 1 - yŷ)
```

**Properties**:
- Used in Support Vector Machines (SVM)
- **Convex** but **non-differentiable** at `yŷ = 1`
- For classification: `y ∈ {-1, +1}`, prediction `ŷ = w^T x`

**Geometric Interpretation**: 
- No penalty when margin `yŷ ≥ 1` (correct classification with confidence)
- Linear penalty when margin `yŷ < 1` (misclassification or low confidence)

### Empirical Risk with Hinge Loss

For linear classifiers `f(x; w) = w^T x`:
```
R_emp(w) = (1/n) Σᵢ max(0, 1 - yᵢw^T xᵢ)
```

**Key Observation**: Each term `max(0, 1 - yᵢw^T xᵢ)` is convex in w (composition of convex function with linear operation), so the sum is convex.

**Problem**: Cannot apply gradient descent because the function is not differentiable everywhere.

### L1 Regularization (LASSO)

Another common non-smooth scenario:
```
f(w) = (1/2n)‖Xw - y‖² + λ‖w‖₁
```

**Structure**: 
- **Smooth part**: `(1/2n)‖Xw - y‖²` (differentiable)
- **Non-smooth part**: `λ‖w‖₁` (not differentiable at w_i = 0)

This motivates **splitting algorithms** that treat the two parts differently.

---

## The Class of Proper Convex Functions

### Definition

A function `F: ℝᵈ → ℝ ∪ {+∞}` belongs to the class of **proper convex lower semicontinuous** functions if:

1. **Convex**: Standard convexity definition
2. **Proper**: `dom(F) = {w : F(w) < +∞} ≠ ∅` (at least one finite point)
3. **Lower Semicontinuous (LSC)**: The epigraph is closed

### Lower Semicontinuity

**Epigraph**: `epi(F) = {(w, t) ∈ ℝᵈ × ℝ : F(w) ≤ t}`

**Definition**: F is lower semicontinuous if `epi(F)` is a closed set.

**Geometric Interpretation**: 
- Points "above" the graph form a closed set
- Allows for "jumps" in function values from below
- Essential for existence of minimizers

**Example**: 
```
F(w) = {0 if w = 0, 1 if w ≠ 0}
```
This is LSC but not continuous.

### Why This Class?

1. **Theoretical**: Most subdifferential theorems work in this setting
2. **Algorithmic**: Ensures subdifferentials are non-empty in the domain
3. **Practical**: Covers most ML applications (convex losses, regularizers, constraints)

**Important Property**: If F is convex and LSC, then F has **continuity on the interior** of its domain (when finite-dimensional).

---

## Subdifferential Theory

### Historical Context

Developed by **Jean-Jacques Moreau** (French mathematician, 1960s-70s) and **R. Tyrrell Rockafellar** (made algorithmic in the US). Sometimes called the **Moreau-Rockafellar subdifferential**.

### Definition of Subdifferential

For `F: ℝᵈ → ℝ ∪ {+∞}` convex, proper, LSC, the **subdifferential** of F at w is:

```
∂F(w) = {g ∈ ℝᵈ : F(u) ≥ F(w) + ⟨g, u - w⟩ ∀u ∈ ℝᵈ}
```

if `w ∈ dom(F)`, and `∂F(w) = ∅` otherwise.

**Elements**: Each `g ∈ ∂F(w)` is called a **subgradient** of F at w.

### Geometric Interpretation

A vector g is a subgradient at w if the **hyperplane**:
```
H(u) = F(w) + ⟨g, u - w⟩
```
lies **below** the graph of F everywhere.

**In 1D**: Subgradients correspond to slopes of lines that "support" the function from below.

### Example: Absolute Value Function

For `F(w) = |w|`:

**At w > 0**: `∂F(w) = {1}` (unique gradient)
**At w < 0**: `∂F(w) = {-1}` (unique gradient)  
**At w = 0**: `∂F(0) = [-1, 1]` (interval of subgradients)

**Key Insight**: At non-differentiable points, the subdifferential contains multiple subgradients representing all possible "slopes" that support the function.

### Fundamental Properties

#### 1. Generalization of Gradient

**Theorem**: If F is differentiable at w (in the interior of dom(F)), then:
```
∂F(w) = {∇F(w)}
```

The subdifferential reduces to the singleton containing the gradient.

#### 2. Optimality Condition

**Theorem**: For convex functions, w* is a global minimizer if and only if:
```
0 ∈ ∂F(w*)
```

**Significance**: This generalizes the first-order optimality condition `∇F(w*) = 0` to non-smooth functions.

#### 3. Set Properties

- `∂F(w)` is always a **closed convex set**
- May be empty (outside domain), singleton (differentiable point), or multi-valued

---

## Subdifferential Calculus Rules

### Sum Rule

**Theorem**: For convex functions F, G:
```
∂(F + G)(w) ⊇ ∂F(w) + ∂G(w)
```

**Equality** holds if there exists a point in the **relative interior** of `dom(F) ∩ dom(G)` (qualification condition).

**Practical Case**: If one function is finite everywhere (like smooth functions), then equality holds.

### Scalar Multiplication

For `α > 0`:
```
∂(αF)(w) = α∂F(w)
```

### Chain Rule

For linear operator `A: ℝᵈ → ℝⁿ` and convex `F: ℝⁿ → ℝ`:
```
∂(F ∘ A)(w) ⊇ A^T ∂F(Aw)
```

**Equality** holds under qualification conditions (e.g., Aw in relative interior of dom(F)).

**Application**: Essential for computing subdifferentials of empirical risk functions.

### Qualification Conditions

These are technical conditions ensuring that subdifferential calculus rules hold with equality rather than just inclusion. They prevent pathological cases where domains have insufficient overlap.

---

## Computing Subdifferentials

### 1D Functions

**General Formula**: For convex function `f: ℝ → ℝ` at point w:
```
∂f(w) = [f'₋(w), f'₊(w)]
```
where `f'₋(w)` and `f'₊(w)` are left and right derivatives.

**Properties**:
- Always a closed interval (possibly singleton)
- If `f'₋(w) = f'₊(w)`, then f is differentiable at w
- If `f'₋(w) < f'₊(w)`, then w is a "corner" point

### Separable Functions

For functions of the form:
```
F(w) = Σᵢ fᵢ(wᵢ)
```

The subdifferential has **product structure**:
```
∂F(w) = ∂f₁(w₁) × ∂f₂(w₂) × ... × ∂fₙ(wₙ)
```

### L1 Norm

Using separability of `‖w‖₁ = Σᵢ |wᵢ|`:
```
∂‖w‖₁ = ∂|w₁| × ∂|w₂| × ... × ∂|wₙ|
```

**Component-wise**:
```
(∂‖w‖₁)ᵢ = {
  1        if wᵢ > 0
  -1       if wᵢ < 0  
  [-1,1]   if wᵢ = 0
}
```

**Example**: `∂‖(1,0,-2)‖₁ = {1} × [-1,1] × {-1}`

Any subgradient has the form `(1, α, -1)` where `α ∈ [-1,1]`.

---

## Subgradient Descent Algorithm

### Algorithm Definition

**Input**: Starting point `w₀ ∈ dom(F)`, step sizes `{γₖ}ₖ≥₀`

**Iteration**: For k = 0, 1, 2, ...:
1. Choose any `gₖ ∈ ∂F(wₖ)`
2. Update: `wₖ₊₁ = wₖ - γₖgₖ`

### Key Properties

1. **Flexibility in subgradient choice**: Any gₖ ∈ ∂F(wₖ) works
2. **Arbitrary initialization**: Can start anywhere in dom(F)  
3. **Variable step sizes**: γₖ can depend on iteration

### Critical Limitation: Not a Descent Method

**Warning**: Subgradient descent is **NOT** a descent algorithm!

#### Example 1: Non-descent behavior

Consider minimizing a 2D function like `F(w₁,w₂) = |w₁| + 2|w₂|`.

At point `(1,0)`: `∂F(1,0) = {1} × [-2,2]`

Choosing subgradient `(1,2)` gives direction `-(1,2)`, which **moves away** from the optimal level set, **increasing** the function value.

#### Example 2: No convergence with constant step size

For `F(w) = |w|` starting at any `w₀ ≠ 0`:
- Subgradient is always `±1` (constant magnitude)
- With constant γ: `|wₖ₊₁ - wₖ| = γ` (constant distance between iterates)
- **No convergence** unless γ → 0

### Comparison with Gradient Descent

| Property | Gradient Descent | Subgradient Descent |
|----------|------------------|-------------------|
| **Descent property** | ✓ Always decreases | ✗ May increase |
| **Step size** | Can be constant | Must decrease to 0 |
| **Gradient behavior** | ∇f → 0 near optimum | Subgradient magnitude constant |
| **Convergence** | To exact optimum | Need averaging/best iterate |

---

## Convergence Analysis of Subgradient Methods

### Assumptions

1. **F convex, real-valued** (F: ℝᵈ → ℝ)
2. **Subdifferentials uniformly bounded**: `‖g‖ ≤ B` for all `g ∈ ∂F(w)`, all w
3. **Optimal set non-empty**: `argmin F ≠ ∅`

**Note**: Assumption 2 is equivalent to F being **Lipschitz continuous** with constant B.

### Convergence Types

Since subgradient methods don't have descent property, we analyze:

#### 1. Best Iterate Convergence
```
min_{i≤k} F(wᵢ) - F*
```

#### 2. Average Iterate Convergence  
Define averaged sequence:
```
w̄ₖ = (Σᵢ₌₀ᵏ γᵢwᵢ)/(Σᵢ₌₀ᵏ γᵢ)
```
Study: `F(w̄ₖ) - F*`

### Main Convergence Theorem

**Theorem**: Under the above assumptions, for any step size sequence `{γₖ}`:
```
F(w̄ₖ) - F* ≤ (‖w₀ - w*‖² + B² Σᵢ₌₀ᵏ γᵢ²)/(2 Σᵢ₌₀ᵏ γᵢ)
```

### Step Size Analysis

#### Constant Step Size: γₖ = γ

```
F(w̄ₖ) - F* ≤ ‖w₀ - w*‖²/(2γ(k+1)) + Bγ/2
```

**Interpretation**:
- **First term**: Decreases to 0 (like gradient descent)
- **Second term**: Constant error floor
- **No convergence** to exact optimum with constant γ
- Can achieve **ε-suboptimality** by choosing γ = O(ε)

#### Decreasing Step Size: γₖ = C/√(k+1)

**Optimal choice** satisfying:
- `Σ γₖ = ∞` (not summable - ensures progress)
- `Σ γₖ² < ∞` (summable - controls noise accumulation)

**Result**: 
```
F(w̄ₖ) - F* = O(1/√k)
```

This is the **optimal rate** for subgradient methods on non-smooth convex functions.

### Robbins-Monro Conditions

**General conditions** for step sizes:
1. `Σₖ γₖ = ∞` (ensures sufficient progress)
2. `Σₖ γₖ² < ∞` (ensures convergence)

**Examples**:
- `γₖ = 1/√k` ✓
- `γₖ = 1/k` ✓  
- `γₖ = 1/log(k)` ✗ (first condition fails)
- `γₖ = 1/√k` ✗ (second condition fails)

### Practical Considerations

1. **Step size tuning**: Major practical challenge - often requires trial and error
2. **Slow convergence**: O(1/√k) vs O(1/k) for smooth case
3. **No descent property**: Makes monitoring progress difficult
4. **Implementation**: Often use piecewise constant step sizes with manual reduction

---

## Proximal Gradient Algorithms

### Motivation: Composite Functions

Many ML problems have **composite structure**:
```
min_w F(w) + R(w)
```

where:
- **F**: Smooth convex function (e.g., squared loss)
- **R**: Non-smooth convex function (e.g., L1 norm, constraints)

**Goal**: Exploit smoothness of F while handling non-smoothness of R.

### Forward-Backward Splitting

**Idea**: 
1. **Forward step**: Use gradient of smooth part F
2. **Backward step**: Use proximal operator for non-smooth part R

### Proximal Gradient Algorithm

**Input**: Starting point w₀, step size γ > 0

**Iteration**: For k = 0, 1, 2, ...:
```
wₖ₊₁ = prox_γR(wₖ - γ∇F(wₖ))
```

**Alternative names**: 
- Forward-backward algorithm
- Proximal gradient method
- Iterative shrinkage-thresholding algorithm (ISTA)

---

## Proximal Operators

### Definition

For convex function R and γ > 0, the **proximal operator** is:
```
prox_γR(v) = argmin_w {R(w) + (1/2γ)‖w - v‖²}
```

### Properties

1. **Well-defined**: The objective is strongly convex (R + strongly convex), so has unique minimizer
2. **Geometric interpretation**: Balances minimizing R with staying close to v
3. **Generalization**: Proximal operator generalizes projection operators

### Key Examples

#### 1. Indicator Function (Projection)

For `R(w) = I_C(w)` (indicator of convex set C):
```
prox_γI_C(v) = argmin_w {I_C(w) + (1/2γ)‖w - v‖²} = proj_C(v)
```

**Result**: Proximal gradient reduces to **projected gradient descent**.

#### 2. L1 Norm (Soft Thresholding)

For `R(w) = ‖w‖₁`:
```
prox_γ‖·‖₁(v) = Sγ(v)
```

where **Sγ** is the **soft thresholding operator**:
```
[Sγ(v)]ᵢ = {
  vᵢ - γ     if vᵢ > γ
  0          if |vᵢ| ≤ γ  
  vᵢ + γ     if vᵢ < -γ
}
```

**Properties**:
- **Componentwise**: Can compute each component independently
- **Sparsity-inducing**: Sets small components to exactly zero
- **Continuous**: Unlike hard thresholding at ±γ

### Why "Soft" Thresholding?

**Comparison with hard thresholding**:
- **Hard**: `H_γ(v)ᵢ = vᵢ` if `|vᵢ| > γ`, 0 otherwise
- **Soft**: Additionally shrinks large components by γ

**Advantage**: Soft thresholding is **continuous**, leading to better algorithmic properties.

### Automatic Sparsity

**Key insight**: If `|[wₖ - γ∇F(wₖ)]ᵢ| ≤ γ`, then `[wₖ₊₁]ᵢ = 0`.

**Practical benefit**: Algorithm automatically produces sparse iterates without manually setting thresholds - crucial for feature selection and compressed sensing.

---

## Applications and Examples

### LASSO (L1-Regularized Least Squares)

**Problem**: 
```
min_w (1/2n)‖Xw - y‖² + λ‖w‖₁
```

**Proximal gradient algorithm**:
```
wₖ₊₁ = S_λγ(wₖ - γX^T(Xwₖ - y)/n)
```

**Step size**: `γ < 2n/‖X^TX‖` (based on Lipschitz constant of smooth part)

### Support Vector Machines

**Problem**: 
```
min_w Σᵢ max(0, 1 - yᵢw^T xᵢ) + λ‖w‖²
```

Can use subgradient descent, but proximal methods may be more efficient for structured versions.

### Constrained Optimization

**Problem**: 
```
min_w F(w) subject to w ∈ C
```

**Reformulation**: `min_w F(w) + I_C(w)`

**Algorithm**: `wₖ₊₁ = proj_C(wₖ - γ∇F(wₖ))`

### Matrix Problems

Proximal operators exist for many matrix functions:
- **Nuclear norm**: `‖X‖_* = Σᵢ σᵢ(X)` (sum of singular values)
- **Frobenius norm**: `‖X‖_F`
- **Spectral norm**: `‖X‖₂`

**Applications**: Matrix completion, low-rank optimization, robust PCA.

---

## Convergence Properties of Proximal Methods

### Assumptions

1. **F convex and L-smooth**
2. **R convex** (possibly non-smooth)
3. **Step size**: `γ ≤ 1/L`

### Main Result

**Theorem**: Under the above assumptions:
```
(F + R)(wₖ) - (F + R)*  ≤  O(1/k)
```

**Key points**:
1. **Same rate as gradient descent** on smooth problems
2. **No penalty for non-smoothness** of R (unlike subgradient methods)
3. **Constant step size** allowed
4. **Sequence convergence**: `wₖ → w*` also holds

### Comparison Summary

| Method | Function Class | Rate | Step Size | Practical |
|--------|----------------|------|-----------|-----------|
| **Gradient Descent** | Smooth | O(1/k) | Constant | ✓ |
| **Subgradient** | Non-smooth | O(1/√k) | Decreasing | Limited |
| **Proximal Gradient** | Composite | O(1/k) | Constant | ✓ |

### Strong Convexity

If F or R (or their sum) is μ-strongly convex, then:
```
‖wₖ - w*‖² ≤ C(1 - γμ)ᵏ
```

**Linear convergence** is recovered, same as smooth strongly convex case.

---

## Advanced Topics and Extensions

### Accelerated Proximal Methods

**Nesterov acceleration** can be applied to proximal gradient:
- **Rate improvement**: O(1/k²) instead of O(1/k)
- **FISTA**: Fast Iterative Shrinkage-Thresholding Algorithm

### Inexact Proximal Operators

When `prox_γR` cannot be computed exactly:
- **Deterministic errors**: Convergence with accumulated error terms
- **Stochastic errors**: More challenging, limited theory

### Stochastic Extensions

- **Stochastic proximal gradient**: When F is an expectation
- **Challenges**: Proximal operators don't naturally handle stochasticity
- **Solutions**: Variance reduction techniques

### Splitting Methods

For problems with multiple non-smooth terms:
```
min_w F(w) + R₁(w) + R₂(w)
```

**Algorithms**:
- Douglas-Rachford splitting
- Alternating Direction Method of Multipliers (ADMM)
- Three-operator splitting

---

## Summary and Key Takeaways

### When to Use Each Method

1. **Gradient Descent**: Smooth convex functions
   - Simple, reliable, well-understood
   - Constant step size OK

2. **Subgradient Methods**: Non-smooth functions, simple structure
   - Slow convergence, challenging step size selection  
   - Historical importance, limited modern use

3. **Proximal Gradient**: Composite smooth + non-smooth
   - Best of both worlds
   - Most important for modern ML applications

### Theoretical Hierarchy

```
Smooth Functions → Composite Functions → General Non-smooth
     ↓                      ↓                    ↓
Gradient Descent → Proximal Gradient → Subgradient Methods
     ↓                      ↓                    ↓
    O(1/k)              O(1/k)              O(1/√k)
```

### Practical Guidelines

1. **Always check for composite structure** before using subgradient methods
2. **Proximal operators**: Learn to compute them for common regularizers
3. **Sparsity**: Proximal gradient with L1 automatically gives sparse solutions
4. **Step sizes**: Much easier for proximal methods than subgradient methods

### Modern Perspective

- **Subgradient methods**: Mainly theoretical interest, some niche applications
- **Proximal methods**: Core of modern non-smooth optimization
- **Extensions**: Active research in acceleration, stochastic variants, splitting methods

The transition from subgradient to proximal methods represents a major advance in optimization, enabling efficient solution of complex machine learning problems while maintaining theoretical guarantees.