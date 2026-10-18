module

public import ProbabilityModels.Bernoulli

/-!
# Binary cross-entropy

For a predicted probability λ and observed binary label y, the single-example
loss is the Bernoulli negative log mass:

    ℓ(λ, y) = −log p(y | λ)
            = −(1 − y) log(1 − λ) − y log λ.

This definition is independent of the network and probability predictor.
`empiricalLoss` sums it over a dataset. The usual finite loss assumes
0 < λ < 1; sigmoid supplies this condition. At the endpoints the definition
inherits Lean's totalized real logarithm, not an infinite loss for an
impossible observation.
-/

@[expose] public section

namespace LossFunctions

open ProbabilityModels

/-- The single-example binary cross-entropy, reusing the Bernoulli mass.

    ℓ(λ, y) = −log p(y | λ),    0 < λ < 1,    y ∈ {0, 1}.
-/
noncomputable def binaryCrossEntropy (p : unitInterval) (y : Fin 2) : ℝ :=
  -Real.log (bernoulliMass p y)

/-- The usual binary cross-entropy formula, before choosing any model. -/
theorem binaryCrossEntropy_eq (p : unitInterval) (y : Fin 2) :
    binaryCrossEntropy p y =
      -(1 - (y.val : ℝ)) * Real.log (1 - (p : ℝ)) -
        (y.val : ℝ) * Real.log (p : ℝ) :=
  neg_log_bernoulliMass p y

/-- For label 1, penalize a small probability of label 1. -/
@[simp] theorem binaryCrossEntropy_one (p : unitInterval) :
    binaryCrossEntropy p 1 = -Real.log (p : ℝ) := by
  simp only [binaryCrossEntropy, bernoulliMass_one]

/-- For label 0, penalize a small probability of label 0. -/
@[simp] theorem binaryCrossEntropy_zero (p : unitInterval) :
    binaryCrossEntropy p 0 = -Real.log (1 - (p : ℝ)) := by
  simp only [binaryCrossEntropy, bernoulliMass_zero]

end LossFunctions
