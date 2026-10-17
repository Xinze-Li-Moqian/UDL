module

public import ProbabilityModels.Family
public import ProbabilityModels.Expectation
public import Mathlib.Probability.Density
public import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap

/-!
# Probability densities

Use `MeasureTheory.HasPDF X P μ` for a random variable whose law under `P`
has a density with respect to a reference measure `μ`.
`MeasureTheory.pdf X P μ` is the corresponding Radon-Nikodym derivative.
`MeasureTheory.Measure.withDensity` constructs a measure from a density.

`Family.Density` attaches chosen, finite real-valued densities to a distribution
family, relative to one fixed reference measure. The densities are nonnegative
and measurable, and `withDensity_eq` identifies the generated measures with
the family. Normalization and integrability follow from this identity.

Counting measure gives probability masses; Lebesgue measure gives continuous
densities. A continuous density value is not the probability of a singleton
and need not be at most one. Densities may vanish: strict positivity, when
needed for logarithms, is a separate hypothesis.

The chosen functions can be evaluated at observations using `predictiveDensity`.
They are specified pointwise, rather than chosen anew from an almost-everywhere
equivalence class. No measurability on the parameter space is required here;
constructing kernels or integrating over parameters needs additional hypotheses.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

/-- Chosen finite densities of a family with respect to a fixed reference measure.

The measure identity, together with the probability-measure property of
`family θ`, implies that each density integrates to one.
-/
structure Family.Density {Θ Y : Type*} [MeasurableSpace Y]
    (family : Family Θ Y) (ν : Measure Y) where
  toFun : Θ → Y → ℝ
  nonneg : ∀ θ y, 0 ≤ toFun θ y
  measurable : ∀ θ, Measurable (toFun θ)
  withDensity_eq : ∀ θ, ν.withDensity (fun y => ENNReal.ofReal (toFun θ y)) =
    (family θ : Measure Y)

namespace Family.Density

variable {Θ Y : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y}

instance : CoeFun (family.Density ν) (fun _ => Θ → Y → ℝ) := ⟨toFun⟩

/-- Event probabilities are obtained by integrating the density on the event. -/
theorem probability_eq_lintegral (p : family.Density ν) (θ : Θ)
    {s : Set Y} (hs : MeasurableSet s) :
    (family θ : Measure Y) s = ∫⁻ y in s, ENNReal.ofReal (p θ y) ∂ν := by
  rw [← p.withDensity_eq θ, withDensity_apply _ hs]

/-- Normalization in the nonnegative extended integral. -/
theorem lintegral_eq_one (p : family.Density ν) (θ : Θ) :
    ∫⁻ y, ENNReal.ofReal (p θ y) ∂ν = 1 := by
  simpa using (p.probability_eq_lintegral θ MeasurableSet.univ).symm

/-- The real-valued probability of an event is its density integral. -/
theorem probability_eq_integral (p : family.Density ν) (θ : Θ)
    {s : Set Y} (hs : MeasurableSet s) :
    (family θ : Measure Y).real s = ∫ y in s, p θ y ∂ν := by
  rw [Measure.real, p.probability_eq_lintegral θ hs]
  symm
  simpa only [ENNReal.toReal_ofReal (p.nonneg θ _)] using
    (integral_toReal (p.measurable θ).ennreal_ofReal.aemeasurable
      (Filter.Eventually.of_forall (fun y => ENNReal.ofReal_lt_top)))

/-- Every chosen density has real integral one. -/
theorem integral_eq_one (p : family.Density ν) (θ : Θ) :
    ∫ y, p θ y ∂ν = 1 := by
  simpa using (p.probability_eq_integral θ MeasurableSet.univ).symm

/-- The density is integrable, so its integral is not a totalization artifact. -/
theorem integrable (p : family.Density ν) (θ : Θ) : Integrable (p θ) ν :=
  integrable_of_integral_eq_one (p.integral_eq_one θ)

/-- Compute expectations by weighting with the density.

Both sides use Mathlib's totalized integral. For a finite expectation, assume
`Integrable g (family θ : Measure Y)` as usual.
-/
theorem expectation_eq_integral (p : family.Density ν) (θ : Θ) (g : Y → ℝ) :
    𝔼[g; (family θ : Measure Y)] = ∫ y, p θ y * g y ∂ν := by
  rw [← p.withDensity_eq θ]
  simpa only [ENNReal.toReal_ofReal (p.nonneg θ _), smul_eq_mul] using!
    (integral_withDensity_eq_integral_toReal_smul (p.measurable θ).ennreal_ofReal
      (Filter.Eventually.of_forall (fun y => ENNReal.ofReal_lt_top)) g)

/-- Compatibility with Mathlib's density interface for the identity variable. -/
theorem hasPDF (p : family.Density ν) (θ : Θ) :
    HasPDF id (family θ : Measure Y) ν := by
  apply hasPDF_of_map_eq_withDensity measurable_id.aemeasurable
    (fun y => ENNReal.ofReal (p θ y)) (p.measurable θ).ennreal_ofReal.aemeasurable
  simpa using (p.withDensity_eq θ).symm

/-- The chosen density at an observed output, with parameters supplied by a predictor. -/
def predictiveDensity {Input Parameters : Type*} (p : family.Density ν)
    (f : Input → Parameters → Θ) (x : Input) (φ : Parameters) (y : Y) : ℝ :=
  p (f x φ) y

/-- The predictive density generates the corresponding predictive distribution. -/
theorem withDensity_predictiveDensity {Input Parameters : Type*} (p : family.Density ν)
    (f : Input → Parameters → Θ) (x : Input) (φ : Parameters) :
    ν.withDensity (fun y => ENNReal.ofReal (p.predictiveDensity f x φ y)) =
      (family.predictiveDistribution f x φ : Measure Y) :=
  p.withDensity_eq (f x φ)

/-- Every input and model parameter choice gives a normalized predictive density. -/
theorem integral_predictiveDensity_eq_one {Input Parameters : Type*} (p : family.Density ν)
    (f : Input → Parameters → Θ) (x : Input) (φ : Parameters) :
    ∫ y, p.predictiveDensity f x φ y ∂ν = 1 :=
  p.integral_eq_one (f x φ)

end Family.Density

/-- The chosen density at output `y` with distribution parameters `θ`.
Here `p : family.Density ν` fixes the reference measure: counting measure
gives probability masses, while Lebesgue measure gives continuous densities.
Use `Pr[· | θ; family]` for the entire distribution. -/
scoped notation "Pr[" y " | " θ "; " p "]" =>
  (Family.Density.toFun p θ y)

end ProbabilityModels
