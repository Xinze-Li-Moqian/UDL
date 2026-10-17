module

public import Discriminative.LinearRegressionTraining
public import Generative.LinearRegressionTraining
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Tactic.Ring

/-!
# Gradient formula verification

Over the reals, the two coordinates of `leastSquaresGradient` are the corresponding
partial derivatives of `leastSquaresLoss`, in both modeling directions.

Source: Understanding Deep Learning, Chapter 2, equation (2.5) and Problem 2.1.

For the discriminative model, write `rᵢ = φ₀ + φ₁ * xᵢ - yᵢ` and keep the
training data fixed. With the slope fixed, `∂rᵢ/∂φ₀ = 1`, so the chain rule gives
`∂(rᵢ²)/∂φ₀ = 2 * rᵢ`. With the intercept fixed, `∂rᵢ/∂φ₁ = xᵢ`, giving
`∂(rᵢ²)/∂φ₁ = 2 * rᵢ * xᵢ`.

Differentiating the finite sum yields `∂L/∂φ₀ = 2 Σᵢ rᵢ` and
`∂L/∂φ₁ = 2 Σᵢ rᵢ xᵢ`. The loss is a sum of squares, so these expressions
are not divided by the number of pairs. The generative formulas exchange the
roles of `xᵢ` and `yᵢ`.

In Lean, `deriv (fun t => ...) a` is the derivative evaluated at `t = a`.
Varying one coordinate while fixing the other expresses a partial derivative.
The `HasDerivAt` proofs establish differentiability and the derivative value;
their `.deriv` projections give the explicit formulas.
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

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

end SupervisedLearning.Discriminative.LinearRegression

namespace SupervisedLearning.Generative.LinearRegression

variable {I : Nat}

/-- With the slope fixed, the generative loss has intercept derivative
`∂L/∂φ₀ = 2 Σᵢ rᵢ`, where `rᵢ = g[yᵢ, φ] - xᵢ`. -/
theorem leastSquaresGradient_intercept_hasDerivAt
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    HasDerivAt (fun intercept => leastSquaresLoss data ⟨intercept, φ.φ₁⟩)
      (leastSquaresGradient data φ).φ₀ φ.φ₀ := by
  have hTerm (i : Fin I) :
      HasDerivAt
        (fun intercept : ℝ =>
          (intercept + φ.φ₁ * (data.y i)[0] - (data.x i)[0]) *
          (intercept + φ.φ₁ * (data.y i)[0] - (data.x i)[0]))
        (2 * (φ.φ₀ + φ.φ₁ * (data.y i)[0] - (data.x i)[0])) φ.φ₀ := by
    have h := ((hasDerivAt_id φ.φ₀).add_const (φ.φ₁ * (data.y i)[0])).sub_const
      (data.x i)[0]
    exact (h.mul h).congr_deriv (by dsimp; ring)
  simpa only [leastSquaresLoss, leastSquaresGradient, linearModel, Vector.getElem_singleton,
    Nat.cast_ofNat, List.sum_ofFn, Finset.mul_sum] using
    (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hTerm i))

/-- With the intercept fixed, the generative loss has slope derivative
`∂L/∂φ₁ = 2 Σᵢ rᵢ yᵢ`, where `rᵢ = g[yᵢ, φ] - xᵢ`. -/
theorem leastSquaresGradient_slope_hasDerivAt
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    HasDerivAt (fun slope => leastSquaresLoss data ⟨φ.φ₀, slope⟩)
      (leastSquaresGradient data φ).φ₁ φ.φ₁ := by
  have hTerm (i : Fin I) :
      HasDerivAt
        (fun slope : ℝ =>
          (φ.φ₀ + slope * (data.y i)[0] - (data.x i)[0]) *
          (φ.φ₀ + slope * (data.y i)[0] - (data.x i)[0]))
        (2 * ((φ.φ₀ + φ.φ₁ * (data.y i)[0] - (data.x i)[0]) * (data.y i)[0]))
        φ.φ₁ := by
    have h := (((hasDerivAt_id φ.φ₁).mul_const (data.y i)[0]).const_add
      φ.φ₀).sub_const (data.x i)[0]
    exact (h.mul h).congr_deriv (by dsimp; ring)
  simpa only [leastSquaresLoss, leastSquaresGradient, linearModel, Vector.getElem_singleton,
    Nat.cast_ofNat, List.sum_ofFn, Finset.mul_sum] using
    (HasDerivAt.fun_sum (u := Finset.univ) (fun i _ => hTerm i))

end SupervisedLearning.Generative.LinearRegression
