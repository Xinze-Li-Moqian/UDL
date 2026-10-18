module

public import LossFunctions.EmpiricalLoss
public import LossFunctions.LogLikelihood
public import Argmin
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Negative log-likelihood

Source: Understanding Deep Learning, Section 5.1.4, equation (5.4).

Negating the log-likelihood turns maximization into minimization of a loss.
This sign reversal needs no positivity hypothesis. Equivalence with the
original likelihood additionally requires the positive-density hypotheses
from `LogLikelihood`.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open scoped LossFunctions ProbabilityModels Optimization

variable {X Y Θ Parameters : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y} {I : ℕ}

/-- Equation (5.4): fixing the density, model, and data gives the loss
`L[φ] = -∑ᵢ log p(yᵢ | f[xᵢ, φ])` as a function of parameters. -/
noncomputable def negativeLogLikelihood (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) : LossFunction ℝ Parameters :=
  empiricalLoss (fun θ target => -Real.log Pr[target | θ; p]) f x y

/-- Summing the single-example negative log densities gives minus the log-likelihood. -/
theorem negativeLogLikelihood_eq_neg_logLikelihood (p : family.Density ν)
    (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) :
    negativeLogLikelihood p f x y φ = -logLikelihood p f x y φ := by
  simp only [negativeLogLikelihood, empiricalLoss, Finset.sum_neg_distrib,
    logLikelihood]

/-- Equation (5.4), written with the book's loss and probability notation. -/
theorem negativeLogLikelihood_apply (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) :
    let L := negativeLogLikelihood p f x y
    L[φ] = -∑ i : Fin I, Real.log Pr[y i | f (x i) φ; p] :=
  negativeLogLikelihood_eq_neg_logLikelihood p f x y φ

/-- The candidate `φ̂` minimizes the negative log-likelihood loss. -/
def IsNegativeLogLikelihoodMinimizer (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) : Prop :=
  let L := negativeLogLikelihood p f x y
  «φ̂» ∈ argmin φ, L[φ]

/-- Maximizing log-likelihood is equivalent to minimizing its negative. -/
theorem isMaximumLogLikelihoodEstimator_iff_isNegativeLogLikelihoodMinimizer
    (p : family.Density ν) (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y)
    («φ̂» : Parameters) :
    IsMaximumLogLikelihoodEstimator p f x y «φ̂» ↔
      IsNegativeLogLikelihoodMinimizer p f x y «φ̂» := by
  change (∀ φ, logLikelihood p f x y φ ≤ logLikelihood p f x y «φ̂») ↔
    (∀ φ, negativeLogLikelihood p f x y «φ̂» ≤ negativeLogLikelihood p f x y φ)
  simp only [negativeLogLikelihood_eq_neg_logLikelihood]
  exact forall_congr' fun _ => neg_le_neg_iff.symm

/-- Under positive observed densities, the original likelihood and the final
loss select exactly the same parameters. -/
theorem isMaximumLikelihoodEstimator_iff_isNegativeLogLikelihoodMinimizer
    (p : family.Density ν) (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y)
    (hpos : ∀ φ i, 0 < Pr[y i | f (x i) φ; p]) («φ̂» : Parameters) :
    IsMaximumLikelihoodEstimator p f x y «φ̂» ↔
      IsNegativeLogLikelihoodMinimizer p f x y «φ̂» :=
  (isMaximumLikelihoodEstimator_iff_isMaximumLogLikelihoodEstimator p f x y hpos «φ̂»).trans
    (isMaximumLogLikelihoodEstimator_iff_isNegativeLogLikelihoodMinimizer p f x y «φ̂»)

end LossFunctions
