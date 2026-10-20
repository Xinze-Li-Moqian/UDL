module

public import ProbabilityModels.Dirac
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure
public import Mathlib.Algebra.BigOperators.Field

/-!
# Empirical distributions

The empirical distribution puts equal weight on each of I observations:

    q = (1 / I) ∑ᵢ δ(zᵢ),    I > 0.

Repeated observations contribute repeatedly. This is a probability measure,
including for real-valued data; no density relative to volume is asserted.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory
open scoped BigOperators ENNReal

variable {Z : Type*} [MeasurableSpace Z] {I : ℕ} [NeZero I]

/-- Equation (5.28): give each observation mass 1 / I. -/
noncomputable def empiricalDistribution (z : Fin I → Z) : ProbabilityMeasure Z :=
  ⟨(I : ℝ≥0∞)⁻¹ • ∑ i, Measure.dirac (z i), by
    constructor
    simp only [Measure.smul_apply, Measure.finsetSum_apply, Measure.dirac_apply_of_mem
      (Set.mem_univ _), Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, mul_one]
    exact ENNReal.inv_mul_cancel (by exact_mod_cast NeZero.ne I) (by simp)⟩

@[simp] theorem coe_empiricalDistribution (z : Fin I → Z) :
    (empiricalDistribution z : Measure Z) =
      (I : ℝ≥0∞)⁻¹ • ∑ i, Measure.dirac (z i) := rfl

/-- One observed value has exactly its Dirac distribution. -/
@[simp] theorem empiricalDistribution_single (a : Z) :
    empiricalDistribution (fun _ : Fin 1 => a) = diracDistribution a := by
  apply Subtype.ext
  simp

variable [MeasurableSingletonClass Z]

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

omit [NormedSpace ℝ E] [CompleteSpace E] in
/-- Any real or vector-valued function is integrable under the finite empirical distribution. -/
theorem integrable_empiricalDistribution (z : Fin I → Z) (g : Z → E) :
    Integrable g (empiricalDistribution z : Measure Z) := by
  rw [coe_empiricalDistribution]
  apply Integrable.smul_measure
  · exact integrable_finsetSum_measure.mpr (fun i _ => integrable_dirac (by simp))
  · simp [NeZero.ne I]

/-- An expectation under the empirical distribution is a sample average.

    𝔼[g; q] = (1 / I) ∑ᵢ g(zᵢ).
-/
theorem expectation_empiricalDistribution (z : Fin I → Z) (g : Z → ℝ) :
    𝔼[g; (empiricalDistribution z : Measure Z)] = (∑ i, g (z i)) / I := by
  rw [coe_empiricalDistribution, integral_smul_measure,
    integral_finsetSum_measure (fun i _ => integrable_dirac (by simp))]
  simp [div_eq_mul_inv, mul_comm]

/-- Vector-valued expectation under an empirical distribution is a sample average.
This includes vectors of model gradients. -/
theorem expectation_empiricalDistribution_smul (z : Fin I → Z) (g : Z → E) :
    𝔼[g; (empiricalDistribution z : Measure Z)] = (I : ℝ)⁻¹ • ∑ i, g (z i) := by
  rw [coe_empiricalDistribution, integral_smul_measure,
    integral_finsetSum_measure (fun i _ => integrable_dirac (by simp))]
  simp

end ProbabilityModels
