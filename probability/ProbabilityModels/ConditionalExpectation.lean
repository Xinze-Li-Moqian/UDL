module

public import ProbabilityModels.Expectation
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
public import Mathlib.Probability.ConditionalExpectation

/-!
# Conditional expectation

Use `MeasureTheory.condExp m μ X`, written `μ[X | m]`, to condition on the
information in a sub-sigma-algebra `m`. Conditioning on a random variable `Z`
uses the sigma-algebra `MeasurableSpace.comap Z` of its codomain sigma-algebra.

Mathlib supplies integrability, measurability, linearity, the tower property
(`condExp_condExp_of_le`), and the identity under independence
(`condExp_indep_eq`). These identities carry their integrability and
sigma-algebra hypotheses and generally assert almost-everywhere equality.

The notation `𝔼[X | s; P]` is declared here, with the measure explicit.
For `s ≤ mΩ` and integrable `X`, the defining property is preservation of
integrals over every `s`-measurable event. Conditioning changes the random
variable into an `s`-measurable one, rather than returning a single number.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

/-- Conditional expectation under an explicit measure, given a sub-sigma-algebra. -/
scoped notation "𝔼[" X " | " s "; " P "]" => MeasureTheory.condExp s P X

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}
variable {P : Measure Ω} {X Y : Ω → ℝ} {s t : MeasurableSpace Ω}

/-- Conditional expectation is measurable with respect to the given information. -/
theorem conditionalExpectation_measurable :
    StronglyMeasurable[s] (𝔼[X | s; P]) :=
  stronglyMeasurable_condExp

/-- The conditional expectation supplied by Mathlib is integrable. -/
theorem conditionalExpectation_integrable : Integrable (𝔼[X | s; P]) P :=
  integrable_condExp

/-- The defining integral identity on every event in the conditioning sigma-algebra. -/
theorem conditionalExpectation_setIntegral [IsProbabilityMeasure P] (hs : s ≤ mΩ)
    (hX : Integrable X P) {A : Set Ω} (hA : MeasurableSet[s] A) :
    ∫ ω in A, (𝔼[X | s; P]) ω ∂P = ∫ ω in A, X ω ∂P :=
  setIntegral_condExp hs hX hA

/-- Constants are unchanged by conditioning. -/
theorem conditionalExpectation_const [IsProbabilityMeasure P] (hs : s ≤ mΩ) (c : ℝ) :
    𝔼[(fun _ : Ω => c) | s; P] = fun _ => c :=
  condExp_const hs c

/-- An integrable variable already measurable with respect to the available
information is unchanged by conditioning. -/
theorem conditionalExpectation_of_measurable [IsProbabilityMeasure P] (hs : s ≤ mΩ)
    (hXs : StronglyMeasurable[s] X) (hX : Integrable X P) :
    𝔼[X | s; P] = X :=
  condExp_of_stronglyMeasurable hs hXs hX

/-- Conditional expectation preserves addition almost everywhere. -/
theorem conditionalExpectation_add (hX : Integrable X P) (hY : Integrable Y P) :
    𝔼[X + Y | s; P] =ᵐ[P] 𝔼[X | s; P] + 𝔼[Y | s; P] :=
  condExp_add hX hY s

/-- Taking the ordinary expectation after conditioning recovers the original mean. -/
theorem expectation_conditionalExpectation [IsProbabilityMeasure P] (hs : s ≤ mΩ)
    (hX : Integrable X P) : 𝔼[𝔼[X | s; P]; P] = 𝔼[X; P] := by
  simpa using! conditionalExpectation_setIntegral hs hX MeasurableSet.univ

/-- The tower property: first conditioning on more information and then on less
information is the same as conditioning directly on the smaller sigma-algebra. -/
theorem conditionalExpectation_tower [IsProbabilityMeasure P]
    (hst : s ≤ t) (ht : t ≤ mΩ) :
    𝔼[𝔼[X | t; P] | s; P] =ᵐ[P] 𝔼[X | s; P] :=
  condExp_condExp_of_le hst ht

/-- With only the trivial sigma-algebra, conditional expectation is the constant mean. -/
theorem conditionalExpectation_bot [IsProbabilityMeasure P] :
    𝔼[X | ⊥; P] = fun _ => 𝔼[X; P] :=
  condExp_bot X

end ProbabilityModels
