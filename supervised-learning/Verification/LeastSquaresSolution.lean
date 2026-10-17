module

public import Verification.GradientFormula
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Analysis.Calculus.LocalExtr.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.LinearCombination
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Least-squares solutions

Source: Understanding Deep Learning, Chapter 2, equation (2.5) and Problem 2.2.

Write `Sx = Σᵢ xᵢ`, `Sy = Σᵢ yᵢ`, `Sxx = Σᵢ xᵢ²`, and `Sxy = Σᵢ xᵢ yᵢ`.
Setting both partial derivatives to zero gives the normal equations:

`I * φ₀ + Sx * φ₁ = Sy`,
`Sx * φ₀ + Sxx * φ₁ = Sxy`.

If `D = I * Sxx - Sx²` is nonzero, solving these equations gives
`φ₀ = (Sy * Sxx - Sx * Sxy) / D` and
`φ₁ = (I * Sxy - Sx * Sy) / D`.
These explicit parameters witness the existence of a global minimizer.
For nonempty data, `D = I * Σᵢ (xᵢ - mean(x))²`, so the nonzero condition
means that the inputs are not all identical. The minimizing pair is then unique.

At any stationary parameter pair `φ`, the two normal equations make the cross
term vanish in the loss expansion:

`L[ψ] = L[φ] + Σᵢ ((ψ₀ - φ₀) + (ψ₁ - φ₁) * xᵢ)² ≥ L[φ]`.

Thus every stationary pair is a global minimizer, without assumptions on the
number or distinctness of inputs. Conversely, every minimizer is stationary.

For nonempty data with all inputs equal to `c`, any pair satisfying
`φ₀ + c * φ₁ = mean(y)` minimizes the loss; one choice is `(mean(y), 0)`.
With no training pairs, the loss is zero for every parameter pair.

More complex models may require iterative fitting because their derivative
equations have no closed-form solution. Iteration alone does not guarantee
that a global minimum is reached.
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

variable {I : Nat}

/-- The determinant of the two normal equations. -/
noncomputable def normalEquationDeterminant (data : TrainingDataset ℝ 1 1 I) : ℝ :=
  (I : ℝ) * (∑ i : Fin I, (data.x i)[0] ^ 2) - (∑ i : Fin I, (data.x i)[0]) ^ 2

/-- The normal equations: the sums of residuals and of residuals times inputs
are zero. These are the two least-squares partial derivatives divided by two. -/
def IsLeastSquaresStationary (data : TrainingDataset ℝ 1 1 I)
    (φ : Parameters ℝ) : Prop :=
  (∑ i : Fin I, (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0])) = 0 ∧
    (∑ i : Fin I,
      (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) * (data.x i)[0]) = 0

private theorem residual_sum (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    (∑ i : Fin I, (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0])) =
      (I : ℝ) * φ.φ₀ + (∑ i : Fin I, (data.x i)[0]) * φ.φ₁ -
        ∑ i : Fin I, (data.y i)[0] := by
  simp [Finset.sum_add_distrib, Finset.sum_sub_distrib, ← Finset.mul_sum, mul_comm]

private theorem weighted_residual_sum (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    (∑ i : Fin I, (φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]) * (data.x i)[0]) =
      (∑ i : Fin I, (data.x i)[0]) * φ.φ₀ +
        (∑ i : Fin I, (data.x i)[0] ^ 2) * φ.φ₁ -
          ∑ i : Fin I, (data.x i)[0] * (data.y i)[0] := by
  simp only [Finset.sum_mul, ← Finset.sum_add_distrib, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Expanding the residual sums gives the usual two linear normal equations. -/
theorem isLeastSquaresStationary_iff_normalEquations
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    IsLeastSquaresStationary data φ ↔
      (I : ℝ) * φ.φ₀ + (∑ i : Fin I, (data.x i)[0]) * φ.φ₁ =
        (∑ i : Fin I, (data.y i)[0]) ∧
      (∑ i : Fin I, (data.x i)[0]) * φ.φ₀ +
        (∑ i : Fin I, (data.x i)[0] ^ 2) * φ.φ₁ =
          (∑ i : Fin I, (data.x i)[0] * (data.y i)[0]) := by
  simp only [IsLeastSquaresStationary, residual_sum, weighted_residual_sum, sub_eq_zero]

/-- Changing a stationary parameter pair adds a sum of squares to its loss. -/
theorem leastSquaresLoss_eq_stationary_add
    (data : TrainingDataset ℝ 1 1 I) (φ ψ : Parameters ℝ)
    (hφ : IsLeastSquaresStationary data φ) :
    leastSquaresLoss data ψ = leastSquaresLoss data φ +
      ∑ i : Fin I, ((ψ.φ₀ - φ.φ₀) + (ψ.φ₁ - φ.φ₁) * (data.x i)[0]) ^ 2 := by
  let r := fun i : Fin I => φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]
  let d₀ := ψ.φ₀ - φ.φ₀
  let d₁ := ψ.φ₁ - φ.φ₁
  obtain ⟨h₀, h₁⟩ := hφ
  change (∑ i, r i) = 0 at h₀
  change (∑ i, r i * (data.x i)[0]) = 0 at h₁
  have hCross : (∑ i, 2 * r i * (d₀ + d₁ * (data.x i)[0])) = 0 := by
    calc
      _ = 2 * d₀ * (∑ i, r i) + 2 * d₁ * (∑ i, r i * (data.x i)[0]) := by
        simp only [Finset.mul_sum, ← Finset.sum_add_distrib]
        apply Finset.sum_congr rfl
        intro i _
        ring
      _ = 0 := by rw [h₀, h₁]; ring
  simp only [leastSquaresLoss, linearModel, Vector.getElem_singleton, List.sum_ofFn]
  change (∑ i, (ψ.φ₀ + ψ.φ₁ * (data.x i)[0] - (data.y i)[0]) *
      (ψ.φ₀ + ψ.φ₁ * (data.x i)[0] - (data.y i)[0])) =
    (∑ i, r i * r i) + ∑ i, (d₀ + d₁ * (data.x i)[0]) ^ 2
  calc
    _ = ∑ i, (r i * r i + 2 * r i * (d₀ + d₁ * (data.x i)[0]) +
        (d₀ + d₁ * (data.x i)[0]) ^ 2) := by
      apply Finset.sum_congr rfl
      intro i _
      dsimp [r, d₀, d₁]
      ring
    _ = _ := by rw [Finset.sum_add_distrib, Finset.sum_add_distrib, hCross, add_zero]

/-- A solution of the normal equations is a global least-squares minimizer. -/
theorem isLeastSquaresMinimizer_of_stationary
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ)
    (hφ : IsLeastSquaresStationary data φ) : IsLeastSquaresMinimizer data φ := by
  intro ψ
  rw [leastSquaresLoss_eq_stationary_add data φ ψ hφ]
  exact le_add_of_nonneg_right (Finset.sum_nonneg fun i _ => sq_nonneg _)

/-- At a minimizer, both partial derivatives vanish, giving the normal equations. -/
theorem isLeastSquaresStationary_of_minimizer
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ)
    (hMin : IsLeastSquaresMinimizer data φ) : IsLeastSquaresStationary data φ := by
  have hIntercept : IsLocalMin
      (fun intercept : ℝ => leastSquaresLoss data ⟨intercept, φ.φ₁⟩) φ.φ₀ :=
    Filter.Eventually.of_forall fun intercept => hMin ⟨intercept, φ.φ₁⟩
  have hSlope : IsLocalMin
      (fun slope : ℝ => leastSquaresLoss data ⟨φ.φ₀, slope⟩) φ.φ₁ :=
    Filter.Eventually.of_forall fun slope => hMin ⟨φ.φ₀, slope⟩
  have h₀ := hIntercept.hasDerivAt_eq_zero
    (leastSquaresGradient_intercept_hasDerivAt data φ)
  have h₁ := hSlope.hasDerivAt_eq_zero
    (leastSquaresGradient_slope_hasDerivAt data φ)
  simp only [leastSquaresGradient, linearModel, Vector.getElem_singleton,
    List.sum_ofFn] at h₀ h₁
  constructor <;> linarith

/-- For this quadratic loss, stationarity and global optimality are equivalent. -/
theorem isLeastSquaresMinimizer_iff_stationary
    (data : TrainingDataset ℝ 1 1 I) (φ : Parameters ℝ) :
    IsLeastSquaresMinimizer data φ ↔ IsLeastSquaresStationary data φ :=
  ⟨isLeastSquaresStationary_of_minimizer data φ,
    isLeastSquaresMinimizer_of_stationary data φ⟩

/-- The explicit least-squares parameters, valid when the determinant is nonzero. -/
noncomputable def closedFormParameters (data : TrainingDataset ℝ 1 1 I) : Parameters ℝ :=
  let Sx := ∑ i : Fin I, (data.x i)[0]
  let Sy := ∑ i : Fin I, (data.y i)[0]
  let Sxx := ∑ i : Fin I, (data.x i)[0] ^ 2
  let Sxy := ∑ i : Fin I, (data.x i)[0] * (data.y i)[0]
  let D := normalEquationDeterminant data
  { φ₀ := (Sy * Sxx - Sx * Sxy) / D
    φ₁ := ((I : ℝ) * Sxy - Sx * Sy) / D }

/-- A minimizer exists when the normal-equation determinant is nonzero. -/
theorem exists_leastSquaresMinimizer (data : TrainingDataset ℝ 1 1 I)
    (hD : normalEquationDeterminant data ≠ 0) :
    ∃ φ : Parameters ℝ, IsLeastSquaresMinimizer data φ := by
  refine ⟨closedFormParameters data, ?_⟩
  -- It suffices to verify the two normal equations for this witness.
  apply isLeastSquaresMinimizer_of_stationary
  apply (isLeastSquaresStationary_iff_normalEquations data _).mpr
  dsimp [closedFormParameters]
  constructor <;> field_simp [hD] <;> dsimp [normalEquationDeterminant] <;> ring

private theorem normalEquations_unique (n s q t u a b a' b' : ℝ)
    (hD : n * q - s ^ 2 ≠ 0)
    (h₀ : n * a + s * b = t) (h₁ : s * a + q * b = u)
    (h₀' : n * a' + s * b' = t) (h₁' : s * a' + q * b' = u) :
    a = a' ∧ b = b' := by
  constructor
  · apply mul_left_cancel₀ hD
    linear_combination q * h₀ - s * h₁ - q * h₀' + s * h₁'
  · apply mul_left_cancel₀ hD
    linear_combination n * h₁ - s * h₀ - n * h₁' + s * h₀'

/-- A nonzero normal-equation determinant makes the stationary pair unique. -/
theorem leastSquaresStationary_unique
    (data : TrainingDataset ℝ 1 1 I) (hD : normalEquationDeterminant data ≠ 0)
    (φ ψ : Parameters ℝ) (hφ : IsLeastSquaresStationary data φ)
    (hψ : IsLeastSquaresStationary data ψ) : φ = ψ := by
  obtain ⟨h₀, h₁⟩ := (isLeastSquaresStationary_iff_normalEquations data φ).mp hφ
  obtain ⟨h₀', h₁'⟩ := (isLeastSquaresStationary_iff_normalEquations data ψ).mp hψ
  obtain ⟨ha, hb⟩ := normalEquations_unique _ _ _ _ _ _ _ _ _ hD h₀ h₁ h₀' h₁'
  cases φ
  cases ψ
  cases ha
  cases hb
  rfl

/-- With a common input, the constant predictor at the mean output is optimal. -/
theorem constantInput_mean_minimizer
    (data : TrainingDataset ℝ 1 1 I) (hI : 0 < I) (c : ℝ)
    (hInput : ∀ i : Fin I, (data.x i)[0] = c) :
    IsLeastSquaresMinimizer data ⟨(∑ i : Fin I, (data.y i)[0]) / (I : ℝ), 0⟩ := by
  apply isLeastSquaresMinimizer_of_stationary
  apply (isLeastSquaresStationary_iff_normalEquations data _).mpr
  have hn : (I : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hI)
  constructor
  · simp only [mul_zero, add_zero]
    field_simp
  · simp only [hInput, mul_zero, add_zero, Finset.sum_const, Finset.card_univ,
      Fintype.card_fin, nsmul_eq_mul, ← Finset.mul_sum]
    field_simp

end SupervisedLearning.Discriminative.LinearRegression
