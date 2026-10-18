module

public import ProbabilityModels.Categorical
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Multiclass cross-entropy for a categorical label

Source: Understanding Deep Learning, Section 5.5, equation (5.24).

For a predicted probability vector λ and observed label y, the single-example
loss is the negative logarithm of the probability assigned to the true label:

    ℓ(λ, y) = −log p(y | λ) = −log λᵧ.

The definition is independent of the network and softmax. `empiricalLoss`
sums it over a dataset. The usual finite interpretation assumes λᵧ > 0;
softmax ensures this. Zero masses inherit Lean's totalized real logarithm,
not an infinite loss for an impossible observation.
-/

@[expose] public section

namespace LossFunctions

open ProbabilityModels

/-- Penalize a small probability assigned to the observed label.

    ℓ(λ, y) = −log λᵧ,    λᵧ > 0.
-/
noncomputable def multiclassCrossEntropy {K : ℕ}
    (p : CategoricalParameters K) (y : Fin K) : ℝ :=
  -Real.log (categoricalMass p y)

/-- Expand the loss as the negative logarithm of the true label's probability. -/
@[simp] theorem multiclassCrossEntropy_eq {K : ℕ}
    (p : CategoricalParameters K) (y : Fin K) :
    multiclassCrossEntropy p y = -Real.log (p.weights y) := rfl

end LossFunctions
