module

public import ProbabilityModels.Expectation
public import Mathlib.Probability.ConditionalProbability
public import Mathlib.MeasureTheory.Integral.Bochner.Set

/-!
# Probability conditioned on an event

Use `ProbabilityTheory.cond μ B`, or `μ[|B]` in the `ProbabilityTheory` scope,
for the measure obtained by conditioning on an event `B`. Its evaluation on
an event `A` is written `μ[A | B]`.

For a probability measure and a measurable event of positive probability,
this is the normalized restriction to `B` and gives the usual ratio
`μ (B ∩ A) / μ B`. Mathlib supplies the corresponding Bayes identities.

Conditioning on a random variable, including continuous observations whose
singletons can have probability zero, is covered by `ConditionalDistribution`.

The results below expose normalization, the ratio formula, the multiplication
rule, total probability, Bayes' rule, and the expectation of an indicator.
Mathlib defines conditioning on a null event to be the zero measure; the
normalization theorem therefore explicitly requires positive event mass.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}
variable {P : Measure Ω} {A B : Set Ω}

/-- Conditioning on an event of nonzero probability gives a probability measure. -/
theorem conditionOnEvent_isProbability [IsProbabilityMeasure P] (hB : P B ≠ 0) :
    IsProbabilityMeasure P[|B] :=
  cond_isProbabilityMeasure hB

/-- The ratio formula for conditioning on a measurable event. For the usual
conditional-probability interpretation, require `P B ≠ 0`. -/
theorem conditionalProbability_eq_ratio (hB : MeasurableSet B) :
    P[A | B] = P (B ∩ A) / P B := by
  rw [cond_apply hB P A, div_eq_mul_inv, mul_comm]

/-- Given a nonnull event, that event has conditional probability one. -/
theorem conditionalProbability_self [IsProbabilityMeasure P] (hB : P B ≠ 0) :
    P[B | B] = 1 :=
  cond_apply_self hB (measure_ne_top P B)

/-- The multiplication rule for joint event probabilities. -/
theorem conditionalProbability_mul [IsProbabilityMeasure P] (hB : MeasurableSet B) :
    P[A | B] * P B = P (B ∩ A) :=
  cond_mul_eq_inter hB A P

/-- Total probability, splitting according to an event and its complement. -/
theorem totalProbability_event_compl [IsProbabilityMeasure P] (hB : MeasurableSet B) :
    P[A | B] * P B + P[A | Bᶜ] * P Bᶜ = P A :=
  cond_add_cond_compl_eq hB P

/-- Bayes' rule in ratio form, using Mathlib's convention on null events. -/
theorem conditionalProbability_bayes [IsProbabilityMeasure P]
    (hA : MeasurableSet A) (hB : MeasurableSet B) :
    P[A | B] = P[B | A] * P A / P B := by
  rw [cond_eq_inv_mul_cond_mul hB hA P, div_eq_mul_inv]
  ac_rfl

/-- Event conditioning also computes probabilities as expectations of indicators.
The left side is converted from `ℝ≥0∞` to `ℝ`. -/
theorem conditionalProbability_eq_expectation_indicator (hA : MeasurableSet A) :
    (P[A | B]).toReal = 𝔼[A.indicator (fun _ => (1 : ℝ)); P[|B]] := by
  simpa using! (integral_indicator_one (μ := P[|B]) hA).symm

end ProbabilityModels
