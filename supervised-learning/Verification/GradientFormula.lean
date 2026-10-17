module

public import LinearRegressionTraining
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Tactic.Ring

/-!
# Gradient formula verification

Over the reals, the two coordinates of `leastSquaresGradient` are the corresponding
partial derivatives of `leastSquaresLoss`.
-/

@[expose] public section

namespace SupervisedLearning.LinearRegression

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

end SupervisedLearning.LinearRegression
