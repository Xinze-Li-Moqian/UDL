module

public import LinearRegressionTraining
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Tactic.Convert
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Supervised learning verification

Proofs are grouped by the property being verified. The first sections use
abstract scalars; the gradient and step-size results specialize to real numbers.
Coverage: loss expansion, minimizer equivalence, iteration consistency,
partial derivatives, and nonincreasing loss under a sufficient step-size bound.
-/

@[expose] public section

open scoped BigOperators

namespace SupervisedLearning.LinearRegression

section AbstractScalars

variable {Scalar : Type} {I : Nat}
variable [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
variable (data : TrainingDataset Scalar 1 1 I)

section LossFormula

/-- The book's expanded objective, written once for the proofs below. -/
def expandedLeastSquaresLoss : LossFunction Scalar (Parameters Scalar) :=
  fun φ =>
    (List.ofFn fun i : Fin I =>
      let deviation := φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]
      deviation * deviation).sum

/-- Equation (2.5): the two expressions define the same loss function. -/
theorem leastSquaresLoss_eq_expanded :
    leastSquaresLoss data = expandedLeastSquaresLoss data := rfl

end LossFormula

section Optimality

/-- Equation (2.6): both expressions have the same minimizer condition.
No existence or uniqueness of a minimizer is assumed. -/
theorem isLeastSquaresMinimizer_iff_expanded [LE Scalar] (φBest : Parameters Scalar) :
    IsLeastSquaresMinimizer data φBest ↔
      Optimization.IsArgmin (expandedLeastSquaresLoss data) φBest := Iff.rfl

end Optimality

section TrainingIteration

variable [OfNat Scalar 2]

/-- Zero training steps preserve the initial parameters. -/
theorem trainLinearRegression_zero_steps (stepSize : Scalar) (initial : Parameters Scalar) :
    trainLinearRegression data stepSize initial 0 = initial := rfl

/-- With the same data and step size, training can be continued in two stages. -/
theorem trainLinearRegression_add_steps (stepSize : Scalar)
    (initial : Parameters Scalar) (m n : Nat) :
    trainLinearRegression data stepSize initial (m + n) =
      trainLinearRegression data stepSize
        (trainLinearRegression data stepSize initial m) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      exact congrArg (Optimization.gradientDescentStep (leastSquaresGradient data) stepSize) ih

end TrainingIteration

section TrainingLoss

variable [OfNat Scalar 2] [LE Scalar] [Std.IsPreorder Scalar]

/-- If every single update does not increase loss, neither does any finite run.
`hStep` is an assumption: this theorem does not establish the gradient formula,
a suitable step size, convergence, or optimality of the returned parameters. -/
theorem trainLinearRegression_loss_le_initial (stepSize : Scalar)
    (hStep : ∀ φ, leastSquaresLoss data (trainLinearRegression data stepSize φ 1) ≤
      leastSquaresLoss data φ)
    (initial : Parameters Scalar) (steps : Nat) :
    leastSquaresLoss data (trainLinearRegression data stepSize initial steps) ≤
      leastSquaresLoss data initial := by
  induction steps with
  | zero => exact Std.le_refl _
  | succ k ih =>
      exact Std.le_trans (hStep (trainLinearRegression data stepSize initial k)) ih

end TrainingLoss

end AbstractScalars

section GradientFormula

variable {I : Nat}

/-- With the slope fixed, the loss derivative with respect to the intercept
is the first coordinate of `leastSquaresGradient`: `∂L/∂φ₀ = 2 Σᵢ rᵢ`. -/
theorem leastSquaresGradient_intercept_hasDerivAt
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    HasDerivAt (fun intercept => leastSquaresLoss data ⟨intercept, φ.φ₁⟩)
      (leastSquaresGradient data φ).φ₀ φ.φ₀ := by
  have hTerm (i : Fin I) :
      HasDerivAt
        (fun intercept : ℝ =>
          (intercept + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) *
          (intercept + φ.φ₁ * (data.x i)[0] - (data.y i)[0]))
        (2 * (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0])) φ.φ₀ := by
    have h := ((hasDerivAt_id φ.φ₀).add_const (φ.φ₁ * (data.x i)[0])).sub_const
      (data.y i)[0]
    exact (h.mul h).congr_deriv (by dsimp; ring)
  simpa only [leastSquaresLoss, leastSquaresGradient, linearModel, Vector.getElem_singleton,
    Nat.cast_ofNat, List.sum_ofFn, Finset.mul_sum] using
    (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hTerm i))

/-- With the intercept fixed, the loss derivative with respect to the slope
is the second coordinate of `leastSquaresGradient`: `∂L/∂φ₁ = 2 Σᵢ rᵢ xᵢ`. -/
theorem leastSquaresGradient_slope_hasDerivAt
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    HasDerivAt (fun slope => leastSquaresLoss data ⟨φ.φ₀, slope⟩)
      (leastSquaresGradient data φ).φ₁ φ.φ₁ := by
  have hTerm (i : Fin I) :
      HasDerivAt
        (fun slope : ℝ =>
          (φ.φ₀ + slope * (data.x i)[0] - (data.y i)[0]) *
          (φ.φ₀ + slope * (data.x i)[0] - (data.y i)[0]))
        (2 * ((φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) * (data.x i)[0]))
        φ.φ₁ := by
    have h := (((hasDerivAt_id φ.φ₁).mul_const (data.x i)[0]).const_add
      φ.φ₀).sub_const (data.y i)[0]
    exact (h.mul h).congr_deriv (by dsimp; ring)
  simpa only [leastSquaresLoss, leastSquaresGradient, linearModel, Vector.getElem_singleton,
    Nat.cast_ofNat, List.sum_ofFn, Finset.mul_sum] using
    (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hTerm i))

end GradientFormula

section StepSize

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

end StepSize

end SupervisedLearning.LinearRegression
