module

public import FittingModels.StochasticGradientDescent.Basic
public import ProbabilityModels.EmpiricalDistribution
public import Mathlib.MeasureTheory.Integral.Pi

/-!
# Uniform batch sampling

Each of the B batch positions independently selects one of the I training
indices with probability 1/I. Sampling is with replacement, so repetitions
are allowed. Both I and B must be positive.

For differentiable sample losses and a fixed parameter φ, the main theorem is:

    E[∇L_batch(φ)] = (B/I) ∇L(φ).

Both batch and full-data losses use sums, as in equation (6.10).
-/

@[expose] public section

namespace FittingModels.StochasticGradientDescent

open MeasureTheory ProbabilityModels LossFunctions
open scoped BigOperators ProbabilityModels Gradient

variable {I B : ℕ} [NeZero I]

/-- Independent uniform draws, with replacement: a product of B uniform index laws.
The empirical distribution of all indices gives each index mass 1/I. -/
noncomputable def uniformBatchLaw (I B : ℕ) [NeZero I] : ProbabilityMeasure (Batch I B) :=
  ⟨Measure.pi (fun _ : Fin B => (empiricalDistribution (id : Fin I → Fin I) :
    Measure (Fin I))), inferInstance⟩

/-- Expectation over independent uniform batch draws, with replacement.
For example: `𝔼ᵤ[batch : Batch I B, g batch]`. -/
scoped notation3 "𝔼ᵤ[" (...) ", " r:60:(scoped f => f) "]" =>
  MeasureTheory.integral (MeasureTheory.ProbabilityMeasure.toMeasure (uniformBatchLaw _ _)) r

open scoped FittingModels.StochasticGradientDescent

section Expectation

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]

/-- Each batch position has the uniform-index expectation. -/
lemma expectation_batchPosition (g : Fin I → E) (j : Fin B) :
    𝔼ᵤ[batch : Batch I B, g (batch j)] =
      (I : ℝ)⁻¹ • ∑ i, g i := by
  change (∫ batch, g (batch j) ∂Measure.pi (fun _ : Fin B =>
    (empiricalDistribution (id : Fin I → Fin I) : Measure (Fin I)))) = _
  rw [integral_comp_eval (integrable_empiricalDistribution id g).aestronglyMeasurable]
  exact expectation_empiricalDistribution_smul id g

/-- Linearity of expectation gives B/I times the full sum. -/
lemma expectation_batchSum (g : Fin I → E) :
    𝔼ᵤ[batch : Batch I B, ∑ j, g (batch j)] =
      ((B : ℝ) / I) • ∑ i, g i := by
  rw [integral_finsetSum Finset.univ (fun _ _ => Integrable.of_finite)]
  simp_rw [expectation_batchPosition]
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin]
  rw [← Nat.cast_smul_eq_nsmul ℝ, smul_smul, div_eq_mul_inv]

end Expectation

variable {X Prediction Y Parameters : Type*}
variable [NormedAddCommGroup Parameters] [InnerProductSpace ℝ Parameters]
variable [CompleteSpace Parameters]
variable [NeZero B]

variable (ℓ : Prediction → Y → ℝ) (f : X → Parameters → Prediction)
variable (x : Fin I → X) (y : Fin I → Y)

-- L is the full-data loss; Lᵦ batch is the loss on the selected batch.
local notation "L" => empiricalLoss ℓ f x y
local notation "Lᵦ" => batchLoss ℓ f x y

/-- The expected batch gradient is B/I times the full-data gradient.

    E[∇L_batch(φ)] = (B/I) ∇L(φ).

The current parameters φ are fixed before sampling. Each sample loss must be
differentiable at φ. Both losses are sums, not averages.
-/
theorem expectation_batchGradient
    (φ : Parameters) (h : ∀ i, DifferentiableAt ℝ (fun ψ => ℓ (f (x i) ψ) (y i)) φ) :
    𝔼ᵤ[batch : Batch I B, ∇ (Lᵦ batch) φ] = ((B : ℝ) / I) • ∇ L φ := by
  let g : Fin I → Parameters := fun i => ∇ (fun ψ => ℓ (f (x i) ψ) (y i)) φ
  have hfull : ∇ L φ = ∑ i, g i :=
    (hasGradientAt_sum _ g (fun i => (h i).hasGradientAt)).gradient
  have hbatch (batch : Batch I B) : ∇ (Lᵦ batch) φ = ∑ j, g (batch j) :=
    (hasGradientAt_sum _ (fun j => g (batch j))
      (fun j => (h (batch j)).hasGradientAt)).gradient
  simp_rw [hbatch, hfull]
  exact expectation_batchSum g

end FittingModels.StochasticGradientDescent
