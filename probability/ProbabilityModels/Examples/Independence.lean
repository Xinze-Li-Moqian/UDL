module

public import ProbabilityModels.Independence

/-!
# Variance of a sample mean

Pairwise independent variables with common finite variance `v` have sample
mean variance `v / n`. Their means and full distributions need not be equal.
The positive sample size is explicit, so there is no division by zero.
-/

@[expose] public section

namespace ProbabilityModels.Examples

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {P : Measure Ω}

/-- Averaging `n` pairwise independent observations divides their common variance by `n`. -/
theorem variance_sampleMean {n : ℕ} (hn : 0 < n) (X : Fin n → Ω → ℝ) (v : ℝ)
    (hIndependent : Pairwise (fun i j => IndepFun (X i) (X j) P))
    (hX : ∀ i, MemLp (X i) 2 P) (hVariance : ∀ i, Var[X i; P] = v) :
    Var[(fun ω => (∑ i, X i ω) / (n : ℝ)); P] = v / (n : ℝ) := by
  have hn' : (n : ℝ) ≠ 0 := by exact_mod_cast (Nat.ne_of_gt hn)
  have hSum : Var[(fun ω => ∑ i, X i ω); P] = (n : ℝ) * v := by
    rw [variance_sum_of_pairwise_independent Finset.univ
      (fun _ _ _ _ hij => hIndependent hij) (fun i _ => hX i)]
    simp [hVariance]
  calc
    _ = (n : ℝ)⁻¹ ^ 2 * Var[(fun ω => ∑ i, X i ω); P] := by
      simpa only [div_eq_mul_inv, mul_comm] using!
        variance_const_mul (n : ℝ)⁻¹ (fun ω => ∑ i, X i ω) P
    _ = v / (n : ℝ) := by
      rw [hSum]
      field_simp

end ProbabilityModels.Examples
