module

public import Mathlib.Analysis.SpecialFunctions.Exp
public import Mathlib.Analysis.SpecialFunctions.Log.Basic
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Field

/-!
# Softmax

Source: Understanding Deep Learning, Section 5.5, equation (5.22).

    softmaxₖ(z) = exp(zₖ) / ∑ⱼ exp(zⱼ).

Unlike a scalar activation, softmax transforms a whole vector of scores.
For a nonempty label set, its entries are positive and sum to one. They can
serve as categorical probabilities. This module specifies the real-valued
transformation; probability measures are constructed by the caller.
-/

@[expose] public section

namespace NeuralNetworks

open scoped BigOperators

variable {K : ℕ}

/-- Exponentiate the scores and divide each by their common total.

    λₖ = softmaxₖ(z) = exp(zₖ) / ∑ⱼ exp(zⱼ).
-/
noncomputable def softmax (z : Fin K → ℝ) (k : Fin K) : ℝ :=
  Real.exp (z k) / ∑ j : Fin K, Real.exp (z j)

variable [NeZero K]

/-- With at least one label, the normalizing denominator is strictly positive. -/
theorem softmax_denominator_pos (z : Fin K → ℝ) :
    0 < ∑ j : Fin K, Real.exp (z j) :=
  Finset.sum_pos (fun _ _ => Real.exp_pos _) Finset.univ_nonempty

/-- Every label receives positive probability for finite real scores. -/
theorem softmax_pos (z : Fin K → ℝ) (k : Fin K) : 0 < softmax z k :=
  div_pos (Real.exp_pos _) (softmax_denominator_pos z)

/-- The output entries form a normalized probability vector. -/
@[simp] theorem sum_softmax (z : Fin K → ℝ) : ∑ k : Fin K, softmax z k = 1 := by
  simp only [softmax, ← Finset.sum_div]
  exact div_self (ne_of_gt (softmax_denominator_pos z))

/-- Each output is at most one; the single-label case attains one. -/
theorem softmax_le_one (z : Fin K → ℝ) (k : Fin K) : softmax z k ≤ 1 := by
  rw [softmax, div_le_one (softmax_denominator_pos z)]
  exact Finset.single_le_sum (fun j _ => (Real.exp_pos (z j)).le) (Finset.mem_univ k)

/-- Within one score vector, softmax preserves the ordering of labels. -/
theorem softmax_le_softmax_iff (z : Fin K → ℝ) (i j : Fin K) :
    softmax z i ≤ softmax z j ↔ z i ≤ z j := by
  rw [softmax, softmax, div_le_div_iff_of_pos_right (softmax_denominator_pos z)]
  exact Real.exp_le_exp

/-- The negative log probability separates into log-sum-exp and the label's score.

    −log softmaxᵧ(z) = log(∑ⱼ exp(zⱼ)) − zᵧ.
-/
theorem neg_log_softmax (z : Fin K → ℝ) (y : Fin K) :
    -Real.log (softmax z y) =
      Real.log (∑ j : Fin K, Real.exp (z j)) - z y := by
  rw [softmax, Real.log_div (Real.exp_ne_zero _)
    (ne_of_gt (softmax_denominator_pos z)), Real.log_exp]
  ring

omit [NeZero K] in
/-- Equal zero scores give the uniform distribution on the labels. -/
@[simp] theorem softmax_zero (k : Fin K) :
    softmax (fun _ : Fin K => (0 : ℝ)) k = 1 / (K : ℝ) := by
  simp [softmax]

end NeuralNetworks
