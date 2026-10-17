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

This is a pointwise family of distributions. Measurability in the input and
parameter, needed to construct a Markov kernel, is a separate requirement.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

/-- A family of probability distributions on `Y`, indexed by parameters `Θ`. -/
abbrev Family (Θ Y : Type*) [MeasurableSpace Y] :=
  Θ → ProbabilityMeasure Y

/-- The output distribution at input `x`, when `f x φ` predicts its parameters.

`Θ` contains distribution parameters, while `Parameters` contains model
parameters. The predictor can be a neural network or any other function with
this signature.
-/
def Family.predictiveDistribution {Θ Y X Parameters : Type*} [MeasurableSpace Y]
    (family : Family Θ Y) (f : X → Parameters → Θ) (x : X) (φ : Parameters) :
    ProbabilityMeasure Y :=
  family (f x φ)

end ProbabilityModels
