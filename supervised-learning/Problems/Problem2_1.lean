module

public import Verification.GradientFormula

/-!
# Problem 2.1 — Gradients of the least-squares loss

Source: Understanding Deep Learning, Chapter 2, Problem 2.1.

For the linear regression model `f[x, φ] = φ₀ + φ₁ * x`, fix the training
dataset and consider the loss from equation (2.5):

`L[φ] = Σᵢ (φ₀ + φ₁ * xᵢ - yᵢ)²`.

Calculate expressions for the partial derivatives `∂L/∂φ₀` and `∂L/∂φ₁`.
These describe the slopes of the loss with respect to the model parameters.

Learning goal: connect the loss function to the gradient used in training.

## Solution

Write `rᵢ = φ₀ + φ₁ * xᵢ - yᵢ` for each residual. Keeping the data fixed:

- With `φ₁` fixed, `∂rᵢ/∂φ₀ = 1`, so `∂(rᵢ²)/∂φ₀ = 2 * rᵢ`.
- With `φ₀` fixed, `∂rᵢ/∂φ₁ = xᵢ`, so `∂(rᵢ²)/∂φ₁ = 2 * rᵢ * xᵢ`.

Differentiating the finite sum gives `∂L/∂φ₀ = 2 Σᵢ rᵢ` and
`∂L/∂φ₁ = 2 Σᵢ rᵢ xᵢ`. There is no division by the number of pairs:
the book uses a sum of squares, not a mean.

In Lean, `deriv (fun t => ...) a` means the derivative with respect to `t`,
evaluated at `t = a`. Varying one coordinate while fixing the other expresses
a partial derivative. The proofs reuse the differentiation rules already
applied in `Verification/GradientFormula.lean`; `.deriv` extracts the derivative
value from a `HasDerivAt` proof.
-/

@[expose] public section

namespace SupervisedLearning.Problems.Problem2_1

open LinearRegression
open scoped BigOperators

variable {I : Nat}

/-- The loss derivative with respect to the intercept, with the slope fixed. -/
theorem loss_deriv_intercept (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    let L := leastSquaresLoss data
    deriv (fun φ₀ => L ⟨φ₀, φ.φ₁⟩) φ.φ₀ =
      2 * ∑ i : Fin I, (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) := by
  simpa only [leastSquaresGradient, linearModel, Vector.getElem_singleton,
    List.sum_ofFn, Nat.cast_ofNat]
    using (leastSquaresGradient_intercept_hasDerivAt data φ).deriv

/-- The loss derivative with respect to the slope, with the intercept fixed. -/
theorem loss_deriv_slope (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    let L := leastSquaresLoss data
    deriv (fun φ₁ => L ⟨φ.φ₀, φ₁⟩) φ.φ₁ =
      2 * ∑ i : Fin I, (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) * (data.x i)[0] := by
  simpa only [leastSquaresGradient, linearModel, Vector.getElem_singleton,
    List.sum_ofFn, Nat.cast_ofNat]
    using (leastSquaresGradient_slope_hasDerivAt data φ).deriv

end SupervisedLearning.Problems.Problem2_1
