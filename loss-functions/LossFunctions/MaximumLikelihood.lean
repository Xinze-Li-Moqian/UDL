module

public import ProbabilityModels.Density
public import Argmax

/-!
# Maximum likelihood

Source: Understanding Deep Learning, Section 5.1.2, equation (5.1).

Fix observations `(x i, y i)` and a density `p` for a distribution family.
The predictor `f (x i) φ` supplies the distribution parameters for observation
`i`. For a neural network, use `f := fun x φ => toParameters (network a φ x)`,
the same parameter map used by `networkPredictiveDistribution`.

The product likelihood models conditionally independent outputs given the
fixed inputs, as in equation (5.2). This assumption is a modeling choice here;
independence is not inferred from the data or from using the same family.
Different inputs may select different distributions within that family.

With counting measure, the factors are probability masses; with Lebesgue
measure, they are densities, not probabilities of individual output values.

Use `open scoped ProbabilityModels Optimization` for `Pr[y | θ; p]` and
`argmax φ, ...`. Lean's quoted identifier `«φ̂»` denotes the candidate estimate,
while the bound `φ` ranges over all model parameters.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open scoped ProbabilityModels Optimization

variable {X Y Θ Parameters : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y} {I : ℕ}

/-- Equation (5.1): multiply the predictive densities at the observed outputs.
The dataset is fixed; the likelihood is a function of model parameters `φ`.
The reference measure `ν` is fixed across all samples and parameter choices. -/
noncomputable def likelihood (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) : ℝ :=
  ∏ i : Fin I, Pr[y i | f (x i) φ; p]

/-- Membership in the argmax in equation (5.1): `φ̂` attains a global
maximum of the likelihood. This specifies optimality without asserting
existence or uniqueness, or supplying a training algorithm. -/
def IsMaximumLikelihoodEstimator (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) : Prop :=
  «φ̂» ∈ argmax φ, ∏ i : Fin I, Pr[y i | f (x i) φ; p]

/-- Expanding the notation recovers comparison with every candidate parameter. -/
theorem isMaximumLikelihoodEstimator_iff (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) :
    IsMaximumLikelihoodEstimator p f x y «φ̂» ↔
      ∀ φ, likelihood p f x y φ ≤ likelihood p f x y «φ̂» := Iff.rfl

/-- Likelihoods are nonnegative; continuous densities need not be at most one. -/
theorem likelihood_nonneg (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) :
    0 ≤ likelihood p f x y φ := by
  apply Finset.prod_nonneg
  intro i _
  exact p.nonneg (f (x i) φ) (y i)

/-- Strictly positive densities at every observation give a positive likelihood. -/
theorem likelihood_pos (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters)
    (hpos : ∀ i, 0 < Pr[y i | f (x i) φ; p]) :
    0 < likelihood p f x y φ :=
  Finset.prod_pos (fun i _ => hpos i)

end LossFunctions
