module

public import ProbabilityModels.Expectation
public import Mathlib.MeasureTheory.Measure.DiracProba

/-!
# Dirac distributions

The Dirac distribution δₐ assigns all probability to a single point a:

    δₐ(S) = 1 if a ∈ S, and 0 otherwise,
    ∫ g(y) dδₐ(y) = g(a).

On the real line, the book writes this as δ[y − a]. It is a probability
measure, not an ordinary density function relative to volume.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

variable {Y : Type*} [MeasurableSpace Y]

/-- All probability is concentrated at a; reuse Mathlib's Dirac probability measure. -/
noncomputable abbrev diracDistribution (a : Y) : ProbabilityMeasure Y := diracProba a

@[simp] theorem coe_diracDistribution (a : Y) :
    (diracDistribution a : Measure Y) = Measure.dirac a := rfl

/-- Integrating against δₐ selects the value at a. -/
@[simp] theorem expectation_diracDistribution [MeasurableSingletonClass Y]
    (a : Y) (g : Y → ℝ) :
    𝔼[g; (diracDistribution a : Measure Y)] = g a :=
  integral_dirac g a

end ProbabilityModels
