# Ridge Regression: Statistical Learning Theory and Error Bounds

## Table of Contents
1. [Learning Theory Framework](#learning-theory-framework)
2. [Ridge Regression Setup](#ridge-regression-setup)
3. [Error Decomposition Strategy](#error-decomposition-strategy)
4. [Bias Analysis (Deterministic Term)](#bias-analysis-deterministic-term)
5. [Variance Analysis (Stochastic Term)](#variance-analysis-stochastic-term)
6. [Concentration Inequalities](#concentration-inequalities)
7. [Matrix Concentration Bounds](#matrix-concentration-bounds)
8. [Final Bound Assembly](#final-bound-assembly)
9. [Extension to General Loss Functions](#extension-to-general-loss-functions)
10. [Empirical Process Theory](#empirical-process-theory)
11. [Mathematical Tricks Reference](#mathematical-tricks-reference)

---

## Learning Theory Framework

### The Learning Problem

**Data**: Observations `(X₁,Y₁), ..., (Xₙ,Yₙ)` where `(X,Y) ~ P` (unknown distribution)

**Goal**: Find function `f` to minimize expected risk:
```
L(f) = E[ℓ(Y, f(X))]
```

**In Practice**: Minimize empirical risk:
```
L̂(f) = (1/n) Σᵢ ℓ(Yᵢ, f(Xᵢ))
```

### The Key Question

How close is the test error `L(f̂)` to the best possible test error?

**Test Error**: Performance on future data (what we care about)
**Training Error**: Performance on observed data (what we optimize)

### Notation Convention

- **Hat (^)**: Depends on finite data → random variable
- **No hat**: Population/ideal quantities → deterministic
- When `n → ∞`, hatted quantities approach their ideal counterparts

---

## Ridge Regression Setup

### Function Class

**Linear functions**: `f(x; w) = w^T x` where `w ∈ ℝᵈ`

**Ridge objective**:
```
min_{w∈ℝᵈ} L̂(w) + λ‖w‖²
```
where `L̂(w) = (1/n) Σᵢ (Yᵢ - w^T Xᵢ)²`

### Key Matrices and Vectors

**Empirical second moment matrix**:
```
Σ̂ = (1/n) X^T X = (1/n) Σᵢ XᵢXᵢ^T
```

**Population second moment matrix**:
```
Σ = E[XX^T]
```

**Empirical mean vector**:
```
Ĥ = (1/n) X^T Y = (1/n) Σᵢ XᵢYᵢ
```

**Population mean vector**:
```
H = E[XY]
```

### Regularization Notation

For any matrix `M`, define:
```
M^{(λ)} = M + λI
```

### Solution Formulas

**Ridge estimator**:
```
ŵ_λ = (Σ̂^{(λ)})⁻¹ Ĥ
```

**Population ridge minimizer**:
```
w_λ = (Σ^{(λ)})⁻¹ H
```

**Unregularized population minimizer**:
```
w† = Σ† H    (pseudoinverse solution)
```

### Assumptions

1. **Bounded responses**: `|Y| ≤ M` almost surely
2. **Bounded features**: `‖X‖ ≤ κ` almost surely

**Key Property**: Ridge solution always exists and is unique due to strong convexity.

---

## Error Decomposition Strategy

### The Three-Way Decomposition

**Goal**: Bound `L(ŵ_λ) - L(w†)`

**Strategy**: Introduce intermediate quantity `w_λ` (population ridge solution):
```
L(ŵ_λ) - L(w†) = [L(ŵ_λ) - L(w_λ)] + [L(w_λ) - L(w†)]
                 ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                    VARIANCE            BIAS
                   (stochastic)      (deterministic)
```

### Why This Decomposition?

1. **Bias term**: Pure regularization effect - deterministic, easy to analyze
2. **Variance term**: Finite sample effect - stochastic, requires concentration

---

## Bias Analysis (Deterministic Term)

### Key Structural Result

**Theorem**: For least squares with ridge regularization:
```
L(w) - L(w†) = ‖w - w†‖²_Σ
```
where `‖v‖²_Σ = v^T Σ v` is the Σ-weighted norm.

### Proof Technique

**Step 1**: Use least squares structure:
```
L(w) = E[(Y - w^T X)²]
```

**Step 2**: Add and subtract `X^T w†`:
```
L(w) = E[(Y - X^T w† + X^T w† - X^T w)²]
     = E[(Y - X^T w†)²] + E[(X^T(w† - w))²] + 2E[(Y - X^T w†)(X^T(w† - w))]
```

**Step 3**: Show cross-term vanishes using normal equations:
```
E[(Y - X^T w†)(X^T(w† - w))] = (H - Σw†)^T(w† - w) = 0
```
Since `w†` satisfies `Σw† = H`.

**Step 4**: Simplify:
```
L(w) - L(w†) = E[(X^T(w† - w))²] = (w - w†)^T Σ (w - w†) = ‖w - w†‖²_Σ
```

### Bias Bound

**Apply to ridge solution**:
```
L(w_λ) - L(w†) = ‖w_λ - w†‖²_Σ
```

**Key computation**:
```
w_λ - w† = (Σ^{(λ)})⁻¹H - Σ†H
         = (Σ^{(λ)})⁻¹H - (Σ^{(λ)})⁻¹Σ^{(λ)}Σ†H
         = (Σ^{(λ)})⁻¹(H - Σ^{(λ)}Σ†H)
         = (Σ^{(λ)})⁻¹(H - (Σ + λI)Σ†H)
         = -(Σ^{(λ)})⁻¹λΣ†H
         = -λ(Σ^{(λ)})⁻¹w†
```

**Therefore**:
```
‖w_λ - w†‖²_Σ = λ²‖(Σ^{(λ)})⁻¹w†‖²_Σ = λ²‖Σ^{1/2}(Σ^{(λ)})⁻¹w†‖²
```

**Matrix norm bound**: `‖Σ^{1/2}(Σ^{(λ)})⁻¹‖ ≤ 1/√λ`

**Final bias bound**:
```
L(w_λ) - L(w†) ≤ λ‖w†‖²
```

---

## Variance Analysis (Stochastic Term)

### The Challenge

Need to bound: `L(ŵ_λ) - L(w_λ)`

**Problem**: Both quantities involve matrix inverses of random matrices - highly nonlinear!

### Matrix Perturbation Technique

**Key insight**: Unravel the nonlinearity using matrix identities.

**Step 1**: Express difference:
```
ŵ_λ - w_λ = (Σ̂^{(λ)})⁻¹Ĥ - (Σ^{(λ)})⁻¹H
```

**Step 2**: Add/subtract mixed terms:
```
ŵ_λ - w_λ = (Σ̂^{(λ)})⁻¹Ĥ - (Σ̂^{(λ)})⁻¹H + (Σ̂^{(λ)})⁻¹H - (Σ^{(λ)})⁻¹H
           = (Σ̂^{(λ)})⁻¹(Ĥ - H) + [(Σ̂^{(λ)})⁻¹ - (Σ^{(λ)})⁻¹]H
```

**Step 3**: Use matrix inverse difference formula:
```
A⁻¹ - B⁻¹ = A⁻¹(B - A)B⁻¹
```

**Apply**:
```
(Σ̂^{(λ)})⁻¹ - (Σ^{(λ)})⁻¹ = (Σ̂^{(λ)})⁻¹(Σ^{(λ)} - Σ̂^{(λ)})(Σ^{(λ)})⁻¹
                             = -(Σ̂^{(λ)})⁻¹(Σ̂ - Σ)(Σ^{(λ)})⁻¹
```

**Step 4**: Combine terms:
```
ŵ_λ - w_λ = (Σ̂^{(λ)})⁻¹[(Ĥ - H) - (Σ̂ - Σ)(Σ^{(λ)})⁻¹H]
```

### Converting to Σ-norm

**Apply Σ^{1/2}**:
```
‖ŵ_λ - w_λ‖²_Σ = ‖Σ^{1/2}(Σ̂^{(λ)})⁻¹[(Ĥ - H) - (Σ̂ - Σ)w_λ]‖²
```

**Bound using triangle inequality**:
```
≤ 2‖Σ^{1/2}(Σ̂^{(λ)})⁻¹‖²[‖Ĥ - H‖² + ‖Σ̂ - Σ‖²‖w_λ‖²]
```

### Matrix Norm Control

**Key lemma**: If `‖Σ̂ - Σ‖ ≤ λ/2`, then:
```
‖Σ^{1/2}(Σ̂^{(λ)})⁻¹‖ ≤ 1/√λ
```

**Proof sketch**: Use perturbation series for matrix inverse.

### Radius Bound

**Lemma**: Ridge solutions stay bounded:
```
‖w_λ‖² ≤ ‖w†‖²
```

**Proof**: From bias analysis, `w_λ = (I + λΣ†)⁻¹w†`, and `‖(I + λΣ†)⁻¹‖ ≤ 1`.

---

## Concentration Inequalities

### Vector Concentration (Hoeffding-type)

For i.i.d. centered random vectors `Zᵢ` with `‖Zᵢ‖ ≤ C`:
```
P(‖(1/n)Σᵢ Zᵢ‖ ≥ ε) ≤ 2exp(-nε²/(2C²))
```

**High-probability bound**:
```
‖(1/n)Σᵢ Zᵢ‖ ≤ C√(2log(2/δ)/n)
```
with probability `1-δ`.

### Application to `Ĥ - H`

**Define**: `Zᵢ = XᵢYᵢ - E[XY]`
- Centered: `E[Zᵢ] = 0`
- Bounded: `‖Zᵢ‖ ≤ κM` (by assumptions)

**Result**:
```
‖Ĥ - H‖ ≤ κM√(2log(4/δ)/n)
```
with probability `1-δ/2`.

---

## Matrix Concentration Bounds

### Matrix Hoeffding Inequality

For i.i.d. centered random matrices `Zᵢ` with `‖Zᵢ‖ ≤ C` (operator norm):
```
P(‖(1/n)Σᵢ Zᵢ‖ ≥ ε) ≤ 2d·exp(-nε²/(2C²))
```

### Application to `Σ̂ - Σ`

**Define**: `Zᵢ = XᵢXᵢ^T - E[XX^T]`
- Centered: `E[Zᵢ] = 0`  
- Bounded: `‖Zᵢ‖ ≤ 2κ²` (since `‖XᵢXᵢ^T‖ = ‖Xᵢ‖² ≤ κ²`)

**Result**:
```
‖Σ̂ - Σ‖ ≤ 2κ²√(2log(4d/δ)/n)
```
with probability `1-δ/2`.

### Frobenius vs Operator Norm

**Key insight**: Frobenius norm controls operator norm:
```
‖A‖ ≤ ‖A‖_F
```

Matrix concentration often easier in Frobenius norm, then transfer to operator norm.

---

## Final Bound Assembly

### Assumption for Simplicity

**Condition**: 
```
λ ≥ 4κ²√(2log(4d/δ)/n)
```

This ensures `‖Σ̂ - Σ‖ ≤ λ/2` with high probability.

### Putting It All Together

**With probability `1-δ`**:
```
L(ŵ_λ) - L(w†) ≤ λ‖w†‖² + (C/λ)·(κ²log(d/δ)/n)
```

where `C` is a universal constant.

### Optimization

**Minimize over λ**: Setting derivative to zero gives:
```
λ* ∝ (κ²log(d/δ)/(n‖w†‖²))^{1/3}
```

**Optimal rate**:
```
L(ŵ_λ*) - L(w†) ≤ C·(κ²‖w†‖²log(d/δ)/n)^{2/3}
```

### Rate Analysis

- **Finite sample**: `O(n^{-2/3})` rate
- **Dimension dependence**: `O(log d)` - very mild!
- **Problem dependence**: Through `κ²‖w†‖²`

---

## Extension to General Loss Functions

### Loss Function Assumptions

**Assumptions**:
1. **Bounded at zero**: `ℓ(y,0) ≤ C₀` for all `y`
2. **Lipschitz in prediction**: `|ℓ(y,a) - ℓ(y,a')| ≤ C_L|a-a'|`

**Examples**:
- Logistic loss: `ℓ(y,a) = log(1 + exp(-ya))`
- Hinge loss: `ℓ(y,a) = max(0, 1-ya)`

### Key Differences from Least Squares

1. **No explicit solution** - must use algorithmic approaches
2. **No quadratic structure** - lose the `‖·‖²_Σ` decomposition  
3. **Need uniform convergence** over function class

### Error Decomposition (Still Works!)

```
L(ŵ_λ) - L(w†) = [L(ŵ_λ) - L̂(ŵ_λ)] + [L̂(ŵ_λ) - L̂(w_λ)] + [L̂(w_λ) - L(w_λ)] + [L(w_λ) - L(w†)]
                  ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯   ⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯⎯
                  Generalization    ≤ 0 (ŵ_λ optimal)  Generalization      Bias
```

### The Challenge: Uniform Convergence

Need to bound:
```
sup_{w∈B} |L(w) - L̂(w)|
```
where `B = {w: R(w) ≤ C₀/λ}` is the regularization ball.

This requires **empirical process theory**.

---

## Empirical Process Theory

### The Setup

**Goal**: Bound `sup_{w∈B} |L(w) - L̂(w)|` where each term is:
```
L(w) - L̂(w) = E[ℓ(Y,w^TX)] - (1/n)Σᵢ ℓ(Yᵢ,w^TXᵢ)
```

### Symmetrization Technique

**Step 1**: Introduce ghost sample `(X'ᵢ,Y'ᵢ)` (independent copy):
```
L(w) = E[ℓ(Y'ᵢ,w^TX'ᵢ)]
```

**Step 2**: Rewrite using symmetry:
```
L(w) - L̂(w) = E_{X'Y'}[(1/n)Σᵢ ℓ(Y'ᵢ,w^TX'ᵢ)] - (1/n)Σᵢ ℓ(Yᵢ,w^TXᵢ)
             = E_{X'Y'}[(1/n)Σᵢ(ℓ(Y'ᵢ,w^TX'ᵢ) - ℓ(Yᵢ,w^TXᵢ))]
```

**Step 3**: Use symmetry of `(ℓ(Y'ᵢ,w^TX'ᵢ) - ℓ(Yᵢ,w^TXᵢ))`:

Distribution is same as `(-1)·(ℓ(Y'ᵢ,w^TX'ᵢ) - ℓ(Yᵢ,w^TXᵢ))`.

**Step 4**: Introduce Rademacher variables `σᵢ ∈ {±1}`:
```
E[|L(w) - L̂(w)|] ≤ 2E[sup_{w∈B} |(1/n)Σᵢ σᵢℓ(Yᵢ,w^TXᵢ)|]
```

### Rademacher Complexity

**Definition**: For function class `G`:
```
R_n(G) = E[sup_{g∈G} |(1/n)Σᵢ σᵢg(Zᵢ)|]
```

**Our case**: `g(Z) = ℓ(Y,w^TX)` for `w ∈ B`.

### Contraction Principle

**Key result**: If `ℓ` is `C_L`-Lipschitz in second argument:
```
R_n({w ↦ ℓ(Y,w^TX) : w ∈ B}) ≤ C_L · R_n({w ↦ w^TX : w ∈ B})
```

**Reduces to**: Bound Rademacher complexity of linear functions!

### Linear Function Rademacher Complexity

**For ball `B = {w: ‖w‖ ≤ R}`**:
```
R_n({w ↦ w^TX : ‖w‖ ≤ R}) = (R/n)E[‖Σᵢ σᵢXᵢ‖]
```

**Using sub-Gaussian concentration**:
```
E[‖Σᵢ σᵢXᵢ‖] ≤ κ√n
```

**Final bound**:
```
R_n({w ↦ ℓ(Y,w^TX) : w ∈ B}) ≤ C_L R κ/√n
```

### Putting It Together

**With regularization ball radius `R = √(C₀/λ)`**:
```
sup_{w∈B} |L(w) - L̂(w)| ≤ C_L κ√(C₀/λ)/√n = C_L κ√(C₀/(λn))
```

**Total bound**:
```
L(ŵ_λ) - L(w†) ≤ λ‖w†‖² + C√(C₀/(λn))
```

**Optimal rate**: `O(n^{-2/3})` (same as least squares!)

---

## Mathematical Tricks Reference

### 1. Matrix Inverse Perturbation

**Identity**: `A⁻¹ - B⁻¹ = A⁻¹(B-A)B⁻¹`

**Usage**: Linearize differences of matrix inverses

### 2. Add-and-Subtract Technique

**Pattern**: `f(x) - f(y) = [f(x) - f(z)] + [f(z) - f(y)]`

**Usage**: Introduce intermediate quantities to isolate effects

### 3. Normal Equations Trick

**For least squares**: If `w†` minimizes `L(w)`, then `E[X(Y - X^Tw†)] = 0`

**Usage**: Makes cross-terms vanish in expansions

### 4. Cauchy-Schwarz for Matrices

**Pattern**: `‖AB‖ ≤ ‖A‖‖B‖`

**Usage**: Bound products of matrix/vector norms

### 5. Triangle Inequality Strategy

**Pattern**: `‖A + B‖ ≤ ‖A‖ + ‖B‖`, so `‖A + B‖² ≤ 2‖A‖² + 2‖B‖²`

**Usage**: Split complex expressions into manageable pieces

### 6. Spectral Norm Bounds

**For PSD matrix**: `‖(A + λI)⁻¹A‖ ≤ 1`

**Usage**: Control regularized inverses

### 7. Jensen's Inequality for Norms

**Pattern**: `‖E[X]‖ ≤ E[‖X‖]`

**Usage**: Move expectations outside norms (when convex)

### 8. Symmetrization

**Key insight**: Replace expectations with empirical averages using ghost samples

**Usage**: Bridge between population and empirical quantities

### 9. Union Bound

**Pattern**: `P(A₁ ∪ ... ∪ Aₖ) ≤ P(A₁) + ... + P(Aₖ)`

**Usage**: Control probability of multiple events simultaneously

### 10. Contraction Principle

**Pattern**: Lipschitz functions don't increase Rademacher complexity

**Usage**: Reduce function class complexity bounds

---

## Summary and Key Insights

### Main Results

1. **Ridge regression achieves `O(n^{-2/3})` rate** under mild assumptions
2. **Dimension dependence is logarithmic** - curse of dimensionality largely avoided
3. **Same rates extend to general Lipschitz losses** via empirical process theory

### Proof Strategy

1. **Decompose error**: Bias + Variance
2. **Bias analysis**: Use problem structure (normal equations)
3. **Variance analysis**: Matrix perturbation + concentration
4. **Uniform bounds**: Symmetrization + Rademacher complexity

### Technical Innovation

The key insight is **unraveling nonlinearity** in matrix inverses through systematic algebraic manipulation, then applying concentration to the linearized terms.

This template extends to many other regularized methods and forms the foundation of modern statistical learning theory.