module

public import Verification.TrainingLoss
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Convert
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Step-size verification

For real-valued linear regression, a data-dependent step-size bound ensures
nonincreasing loss. This does not establish convergence or finite-step optimality.
-/

@[expose] public section

open scoped BigOperators

namespace SupervisedLearning.LinearRegression

variable {I : Nat}

/-- The data-dependent bound `B = Σᵢ (1 + xᵢ²)` for the step-size condition.
It includes the number of examples because our loss is a sum, not a mean. -/
noncomputable def leastSquaresInputBound (data : TrainingDataset ℝ 1 1 I) : ℝ :=
  ∑ i : Fin I, (1 + (data.x i)[0] ^ 2)

/-- The algebra behind the descent guarantee: expand the updated squared
residuals, then bound their quadratic term using `(g₁ - xᵢ * g₀)² ≥ 0`. -/
private theorem sumSquares_update_le (x r : Fin I → ℝ) (η : ℝ)
    (hη : 0 ≤ η) (hBound : η * (∑ i, (1 + x i ^ 2)) ≤ 1) :
    (∑ i, (r i - η * ((2 * ∑ j, r j) + (2 * ∑ j, r j * x j) * x i)) ^ 2) ≤
      ∑ i, r i ^ 2 := by
  let g₀ := 2 * ∑ i, r i
  let g₁ := 2 * ∑ i, r i * x i
  let Q := g₀ ^ 2 + g₁ ^ 2
  let C := ∑ i, (g₀ + g₁ * x i) ^ 2
  have hQ : 0 ≤ Q := add_nonneg (sq_nonneg _) (sq_nonneg _)
  have hCross : (∑ i, 2 * r i * (g₀ + g₁ * x i)) = Q := by
    calc
      _ = (2 * ∑ i, r i) * g₀ + (2 * ∑ i, r i * x i) * g₁ := by
        simp only [Finset.mul_sum, Finset.sum_mul, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = Q := by
        change g₀ * g₀ + g₁ * g₁ = g₀ ^ 2 + g₁ ^ 2
        ring
  -- Exact change in loss: L(next) = L(current) - η * Q + η² * C.
  have hExpansion : (∑ i, (r i - η * (g₀ + g₁ * x i)) ^ 2) =
      (∑ i, r i ^ 2) - η * Q + η ^ 2 * C := by
    calc
      _ = ∑ i, (r i ^ 2 - η * (2 * r i * (g₀ + g₁ * x i)) +
          η ^ 2 * (g₀ + g₁ * x i) ^ 2) := by
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = _ := by
        rw [Finset.sum_add_distrib, Finset.sum_sub_distrib,
          ← Finset.mul_sum, ← Finset.mul_sum, hCross]
  have hC : C ≤ (∑ i, (1 + x i ^ 2)) * Q := by
    calc
      C ≤ ∑ i, (1 + x i ^ 2) * Q := by
        apply Finset.sum_le_sum
        intro i _
        dsimp [Q]
        nlinarith only [sq_nonneg (g₁ - x i * g₀)]
      _ = _ := (Finset.sum_mul _ _ _).symm
  have hQuadratic := mul_le_mul_of_nonneg_left hC (sq_nonneg η)
  have hLinear := mul_le_mul_of_nonneg_right hBound (mul_nonneg hη hQ)
  change (∑ i, (r i - η * (g₀ + g₁ * x i)) ^ 2) ≤ ∑ i, r i ^ 2
  rw [hExpansion]
  nlinarith only [hQuadratic, hLinear]

/-- If `0 ≤ η` and `η * B ≤ 1`, one training update does not increase loss.
This is a sufficient condition, including for an empty dataset. It does not
assert strict decrease, convergence, or that the updated parameters minimize loss. -/
theorem trainLinearRegression_one_step_loss_le
    (data : TrainingDataset ℝ 1 1 I) (stepSize : ℝ)
    (hStepSize : 0 ≤ stepSize)
    (hBound : stepSize * leastSquaresInputBound data ≤ 1)
    (φ : Parameters ℝ) :
    leastSquaresLoss data (trainLinearRegression data stepSize φ 1) ≤
      leastSquaresLoss data φ := by
  have h := sumSquares_update_le
    (fun i => (data.x i)[0])
    (fun i => φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0])
    stepSize hStepSize (by simpa [leastSquaresInputBound] using hBound)
  simp only [leastSquaresLoss, List.sum_ofFn]
  convert h using 1 <;> apply Finset.sum_congr rfl <;> intro i _
  all_goals
    dsimp [trainLinearRegression, Optimization.gradientDescent,
      Optimization.gradientDescentStep, leastSquaresGradient, linearModel,
      HSub.hSub, Sub.sub, HSMul.hSMul, SMul.smul]
    try simp only [List.sum_ofFn]
    ring

/-- The same step-size bound guarantees that any finite training run has loss
no greater than the initial loss. The single-step premise is now proved above. -/
theorem trainLinearRegression_loss_le_initial_of_stepSize
    (data : TrainingDataset ℝ 1 1 I) (stepSize : ℝ)
    (hStepSize : 0 ≤ stepSize)
    (hBound : stepSize * leastSquaresInputBound data ≤ 1)
    (initial : Parameters ℝ) (steps : Nat) :
    leastSquaresLoss data (trainLinearRegression data stepSize initial steps) ≤
      leastSquaresLoss data initial :=
  trainLinearRegression_loss_le_initial data stepSize
    (trainLinearRegression_one_step_loss_le data stepSize hStepSize hBound) initial steps

end SupervisedLearning.LinearRegression
