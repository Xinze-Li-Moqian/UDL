module

public import Mathlib.InformationTheory.KullbackLeibler.Basic
public import Mathlib.MeasureTheory.Measure.ProbabilityMeasure

/-!
# Kullback–Leibler divergence

Reuse Mathlib's measure-theoretic KL divergence for probability measures.
When q and p have densities relative to a common reference measure:

    KL(q ‖ p) = ∫ q(z) log(q(z) / p(z)) dz.

The value can be infinite. In particular, q must be absolutely continuous
with respect to p for finite KL: every p-null event must also be q-null.
KL is generally asymmetric, so it is not a metric.
-/

@[expose] public section

namespace InformationTheoryModels

open MeasureTheory
open scoped ENNReal

variable {Y : Type*} [MeasurableSpace Y]

/-- KL from the reference distribution q to the model distribution p.
This is Mathlib's `InformationTheory.klDiv`, with bundled probability measures. -/
noncomputable abbrev klDivergence (q p : ProbabilityMeasure Y) : ℝ≥0∞ :=
  InformationTheory.klDiv (q : Measure Y) (p : Measure Y)

/-- A distribution agrees with itself. -/
@[simp] theorem klDivergence_self (q : ProbabilityMeasure Y) :
    klDivergence q q = 0 :=
  InformationTheory.klDiv_self _

/-- KL vanishes exactly when the two probability measures are equal. -/
theorem klDivergence_eq_zero_iff (q p : ProbabilityMeasure Y) :
    klDivergence q p = 0 ↔ q = p := by
  rw [klDivergence, InformationTheory.klDiv_eq_zero_iff]
  exact Subtype.val_injective.eq_iff

/-- If the model excludes an event that q allows, KL is infinite. -/
theorem klDivergence_of_not_ac (q p : ProbabilityMeasure Y)
    (h : ¬ (q : Measure Y) ≪ (p : Measure Y)) :
    klDivergence q p = ∞ :=
  InformationTheory.klDiv_of_not_ac h

end InformationTheoryModels
