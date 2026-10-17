module

public import Problems.Problem2_1
public import Verification.LeastSquaresSolution
public import Mathlib.Tactic.FieldSimp

/-!
# Problem 2.2 — A closed-form least-squares solution

Source: Understanding Deep Learning, Chapter 2, Problem 2.2.

Using the derivatives from Problem 2.1, set `∂L/∂φ₀ = 0` and `∂L/∂φ₁ = 0`,
and solve for the parameters `φ₀` and `φ₁` that minimize the loss.

Show that linear regression admits a closed-form solution. Relate this to
the use of iterative fitting methods, such as gradient descent, for more
complex models where a closed-form solution is unavailable.

Learning goal: prove existence by giving explicit parameters and showing that
they minimize the loss.

## Solution

Write `Sx = Σᵢ xᵢ`, `Sy = Σᵢ yᵢ`, `Sxx = Σᵢ xᵢ²`, and `Sxy = Σᵢ xᵢ yᵢ`.
Setting the two derivatives from Problem 2.1 to zero gives the normal equations:

`I * φ₀ + Sx * φ₁ = Sy`,
`Sx * φ₀ + Sxx * φ₁ = Sxy`.

If `D = I * Sxx - Sx²` is nonzero, solving these two linear equations gives
`φ₀ = (Sy * Sxx - Sx * Sxy) / D` and
`φ₁ = (I * Sxy - Sx * Sy) / D`.

For nonempty data, `D = I * Σᵢ (xᵢ - mean(x))²`, so the nonzero condition
means that the inputs are not all identical. This is the case with a unique
parameter pair.

Why is this a minimum? At a solution `φ`, both residual sums are zero, so
for any other parameters `ψ`, expanding the squares gives

`L[ψ] = L[φ] + Σᵢ ((ψ₀ - φ₀) + (ψ₁ - φ₁) * xᵢ)² ≥ L[φ]`.

The single theorem below supplies the explicit parameters as an existence
witness, then proves their global minimality. It uses the general minimization
facts in `Verification/LeastSquaresSolution.lean`.

Degenerate cases:
- If all inputs equal `c` and there is at least one pair, every parameter pair
  with `φ₀ + c * φ₁ = mean(y)` minimizes the loss. One choice is `(mean(y), 0)`.
- With no training pairs, the loss is zero for every parameter pair.

For more complex models, the derivative equations may not admit a closed-form
solution. This motivates iterative methods such as gradient descent; it does
not guarantee that those methods find a global minimum.
-/

@[expose] public section

namespace SupervisedLearning.Problems.Problem2_2

open LinearRegression
open scoped BigOperators

variable {I : Nat}

/-- The explicit witness used below, with a nonzero determinant assumed in the theorem. -/
noncomputable def closedFormParameters (data : TrainingDataset ℝ 1 1 I) : Parameters ℝ :=
  let Sx := ∑ i : Fin I, (data.x i)[0]
  let Sy := ∑ i : Fin I, (data.y i)[0]
  let Sxx := ∑ i : Fin I, (data.x i)[0] ^ 2
  let Sxy := ∑ i : Fin I, (data.x i)[0] * (data.y i)[0]
  let D := normalEquationDeterminant data
  { φ₀ := (Sy * Sxx - Sx * Sxy) / D
    φ₁ := ((I : ℝ) * Sxy - Sx * Sy) / D }

/-- Problem 2.2: a minimizer exists, witnessed by the closed-form parameters. -/
theorem exists_leastSquaresMinimizer (data : TrainingDataset ℝ 1 1 I)
    (hD : normalEquationDeterminant data ≠ 0) :
    ∃ φ : Parameters ℝ, IsLeastSquaresMinimizer data φ := by
  refine ⟨closedFormParameters data, ?_⟩
  -- It suffices to verify the two normal equations for this witness.
  apply isLeastSquaresMinimizer_of_stationary
  apply (isLeastSquaresStationary_iff_normalEquations data _).mpr
  dsimp [closedFormParameters]
  constructor <;> field_simp [hD] <;> dsimp [normalEquationDeterminant] <;> ring

end SupervisedLearning.Problems.Problem2_2
