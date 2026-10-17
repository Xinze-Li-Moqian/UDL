module

public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Parameterized probability distributions

Source: Understanding Deep Learning, Section 5.1.1.

A distribution family assigns a probability measure on the output space to
each distribution parameter `θ : Θ`. A predictor `f x φ` supplies `θ` from an
input `x` and model parameters `φ`; the resulting measure models the output
distribution at that input.

Mathlib's `ProbabilityMeasure` supplies the measure axioms and normalization.
The parameter type `Θ` should encode any restrictions needed by the family,
such as a strictly positive variance when using a Gaussian density.

This is a pointwise family of distributions. `Family.toKernel` and
`Family.predictiveKernel` in `ConditionalDistribution` additionally assume
measurability to construct Markov kernels.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

/-- A family of probability distributions on `Y`, indexed by parameters `Θ`. -/
abbrev Family (Θ Y : Type*) [MeasurableSpace Y] :=
  Θ → ProbabilityMeasure Y

/-- The distribution `family θ`, with `·` marking the output placeholder.
This denotes the entire `ProbabilityMeasure`, not a density or event probability;
`| θ` specifies the family parameter. Available with `open scoped ProbabilityModels`. -/
scoped notation "Pr[" "·" " | " θ "; " family "]" =>
  (family θ : MeasureTheory.ProbabilityMeasure _)

/-- The predictive distribution `model φ x` at input `x` with model parameters `φ`.
The model includes the chosen family and the conversion to distribution parameters.
Available with `open scoped ProbabilityModels`. -/
scoped notation "Pr[" "·" " | " x "; " model ", " φ "]" =>
  (model φ x : MeasureTheory.ProbabilityMeasure _)

/-- The output distribution at input `x`, when `f x φ` predicts its parameters.

`Θ` contains distribution parameters, while `Parameters` contains model
parameters. The predictor can be a neural network or any other function with
this signature.
-/
def Family.predictiveDistribution {Θ Y X Parameters : Type*} [MeasurableSpace Y]
    (family : Family Θ Y) (f : X → Parameters → Θ) (x : X) (φ : Parameters) :
    ProbabilityMeasure Y :=
  family (f x φ)

/-- For this model, the predicted output distribution depends on `x` through
`θ = f x φ`. The equality follows from the construction of the model. -/
@[simp] theorem Family.predictiveDistribution_eq {Θ Y X Parameters : Type*}
    [MeasurableSpace Y] (family : Family Θ Y) (f : X → Parameters → Θ)
    (x : X) (φ : Parameters) :
    family.predictiveDistribution f x φ = Pr[· | f x φ; family] := rfl

end ProbabilityModels
