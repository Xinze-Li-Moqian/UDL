module

public import ProbabilityModels.Family
public import ProbabilityModels.ConditionalExpectation
public import Mathlib.Probability.Kernel.CondDistrib

/-!
# Conditional distributions

Use `ProbabilityTheory.Kernel X Y` with `IsMarkovKernel` for a measurable
family of probability measures on `Y`, indexed by inputs in `X`.

`ProbabilityTheory.condDistrib Y X μ` constructs a regular conditional law
from random variables and an underlying measure, under Mathlib's standard
Borel and finiteness hypotheses. Its defining relation to conditional
expectation holds almost everywhere.

A specified predictive kernel is a model for a conditional law.
`Family.toKernel` and `Family.predictiveKernel` provide the connection from a
measurable parameterized family and a measurable predictor. Identifying that
model with the conditional law of a given data distribution is a separate claim.

The remaining results connect Mathlib's regular conditional distribution to
joint laws, probabilities of events, and conditional expectations. The output
space is assumed nonempty and standard Borel where the construction needs it.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

section Models

variable {Θ Input Output Parameters : Type*}
variable [MeasurableSpace Θ] [MeasurableSpace Input] [MeasurableSpace Output]

/-- A measurable family of probability measures defines a Markov kernel. -/
def Family.toKernel (family : Family Θ Output) (hfamily : Measurable family) :
    Kernel Θ Output where
  toFun θ := (family θ : Measure Output)
  measurable' := measurable_subtype_coe.comp hfamily

instance Family.toKernel_isMarkovKernel (family : Family Θ Output)
    (hfamily : Measurable family) : IsMarkovKernel (family.toKernel hfamily) where
  isProbabilityMeasure θ := (family θ).prop

@[simp] theorem Family.toKernel_apply (family : Family Θ Output)
    (hfamily : Measurable family) (θ : Θ) :
    family.toKernel hfamily θ = (family θ : Measure Output) := rfl

/-- At fixed model parameters, a measurable predictor selects a measurable
probability distribution for each input. -/
def Family.predictiveKernel (family : Family Θ Output) (hfamily : Measurable family)
    (f : Input → Parameters → Θ) (φ : Parameters) (hf : Measurable (fun x => f x φ)) :
    Kernel Input Output :=
  Family.toKernel (fun x => family (f x φ)) (hfamily.comp hf)

instance Family.predictiveKernel_isMarkovKernel (family : Family Θ Output)
    (hfamily : Measurable family) (f : Input → Parameters → Θ) (φ : Parameters)
    (hf : Measurable (fun x => f x φ)) :
    IsMarkovKernel (family.predictiveKernel hfamily f φ hf) := by
  unfold Family.predictiveKernel
  infer_instance

@[simp] theorem Family.predictiveKernel_apply (family : Family Θ Output)
    (hfamily : Measurable family) (f : Input → Parameters → Θ) (φ : Parameters)
    (hf : Measurable (fun x => f x φ)) (x : Input) :
    family.predictiveKernel hfamily f φ hf x =
      (family.predictiveDistribution f x φ : Measure Output) := rfl

end Models

section ConditionalLaws

variable {Ω Input Output : Type*} {mΩ : MeasurableSpace Ω}
variable {mInput : MeasurableSpace Input} [MeasurableSpace Output]
variable [StandardBorelSpace Output] [Nonempty Output]
variable {P : Measure Ω} [IsProbabilityMeasure P]
variable {X : Ω → Input} {Y : Ω → Output}

/-- Integrating the conditional law against the input marginal recovers the
joint law of input and output. -/
theorem conditionalDistribution_joint (hX : Measurable X) (hY : Measurable Y) :
    (P.map X) ⊗ₘ condDistrib Y X P = P.map (fun ω => (X ω, Y ω)) :=
  compProd_map_condDistrib hX.aemeasurable hY.aemeasurable

/-- At an input with positive probability mass, the conditional distribution
agrees with the usual joint-probability ratio. -/
theorem conditionalDistribution_eq_ratio [MeasurableSingletonClass Input]
    (hX : Measurable X) (hY : Measurable Y) (x : Input) (hx : P.map X {x} ≠ 0)
    (A : Set Output) :
    condDistrib Y X P x A =
      (P.map X {x})⁻¹ * P.map (fun ω => (X ω, Y ω)) ({x} ×ˢ A) :=
  condDistrib_apply_of_ne_zero hX hY x hx A

/-- Conditional probabilities are conditional expectations of event indicators.
The equality holds almost everywhere under the original probability measure. -/
theorem conditionalDistribution_probability (hX : Measurable X) (hY : Measurable Y)
    {A : Set Output} (hA : MeasurableSet A) :
    (fun ω => (condDistrib Y X P (X ω)).real A) =ᵐ[P]
      𝔼[(Y ⁻¹' A).indicator (fun _ => (1 : ℝ)) | mInput.comap X; P] :=
  condDistrib_ae_eq_condExp hX hY hA

end ConditionalLaws

/-- For a real integrable output, conditional expectation is the mean of the
conditional distribution, evaluated at the observed input. -/
theorem conditionalExpectation_eq_integral_conditionalDistribution
    {Ω Input : Type*} {mΩ : MeasurableSpace Ω} {mInput : MeasurableSpace Input}
    {P : Measure Ω} [IsProbabilityMeasure P] {X : Ω → Input} {Y : Ω → ℝ}
    (hX : Measurable X) (hY : Integrable Y P) :
    𝔼[Y | mInput.comap X; P] =ᵐ[P]
      fun ω => 𝔼[id; condDistrib Y X P (X ω)] := by
  simpa only [id_eq] using! condExp_ae_eq_integral_condDistrib' hX hY

end ProbabilityModels
