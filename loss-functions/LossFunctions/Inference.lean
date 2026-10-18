module

public import ProbabilityModels.Gaussian
public import Argmax

/-!
# Inference

Source: Understanding Deep Learning, Section 5.1.5, equation (5.5).

Training optimizes model parameters `φ`. Inference fixes parameters `φ̂` and
a new input `x`, then chooses an output `ŷ` maximizing the predictive density.
Such an output is a mode of the chosen density. The criterion below specifies
this property without assuming existence or uniqueness for every family.

Returning the full distribution already uses `Family.predictiveDistribution`
or `networkPredictiveDistribution`. For a positive-variance Gaussian, returning
the predicted mean satisfies the mode criterion.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open scoped ProbabilityModels Optimization

variable {X Y Θ Parameters : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y}

/-- Equation (5.5): with the input and parameters fixed, `ŷ` maximizes the
chosen predictive density over outputs. The fixed parameters are typically
obtained from training; this criterion also applies to any supplied parameters. -/
def IsModePrediction (p : family.Density ν) (f : X → Parameters → Θ)
    (x : X) («φ̂» : Parameters) (ŷ : Y) : Prop :=
  ŷ ∈ argmax y, Pr[y | f x «φ̂»; p]

/-- The predicted output has density at least as large as any other output. -/
theorem isModePrediction_iff (p : family.Density ν) (f : X → Parameters → Θ)
    (x : X) («φ̂» : Parameters) (ŷ : Y) :
    IsModePrediction p f x «φ̂» ŷ ↔
      ∀ y, Pr[y | f x «φ̂»; p] ≤ Pr[ŷ | f x «φ̂»; p] := Iff.rfl

/-- For Gaussian predictions, the predicted mean is a valid mode prediction.
The general density inequality is supplied by the probability foundation. -/
theorem gaussian_mean_isModePrediction (f : X → Parameters → PositiveGaussianParameters)
    (x : X) («φ̂» : Parameters) :
    IsModePrediction positiveGaussianDensity f x «φ̂» (f x «φ̂»).1 := by
  intro y
  exact positiveGaussianDensity_le_mean (f x «φ̂») y

end LossFunctions
