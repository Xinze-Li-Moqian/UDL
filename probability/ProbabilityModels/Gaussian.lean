module

public import ProbabilityModels.Density
public import ProbabilityModels.StandardDeviation
public import Mathlib.Probability.Distributions.Gaussian.Real

/-!
# Real Gaussian distributions

Mathlib's `ProbabilityTheory.gaussianReal m v` is a probability measure with
mean `m : ℝ` and variance `v : ℝ≥0`. Its mean and variance are characterized by
`integral_id_gaussianReal` and `variance_id_gaussianReal`; all finite moments
exist (`memLp_id_gaussianReal`).

`gaussianFamily` packages this measure as a `ProbabilityModels.Family`, so it
can be used with `Family.predictiveDistribution`.

The density is `gaussianPDFReal m v` (real-valued) or `gaussianPDF m v`
(`ℝ≥0∞`-valued). Their normalization results require `v ≠ 0`. When `v = 0`,
the Gaussian measure is a Dirac mass and does not have a Lebesgue density.
The second parameter is the variance, not the standard deviation.

`positiveGaussianFamily` restricts the variance to be strictly positive.
`positiveGaussianDensity` attaches the usual real Gaussian density to this
family, relative to Lebesgue measure. Its pointwise positivity allows later
likelihood constructions to take the real logarithm.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory
open scoped NNReal

/-- The Gaussian family, indexed by its mean and nonnegative variance. -/
noncomputable def gaussianFamily : Family (ℝ × ℝ≥0) ℝ :=
  fun θ => (gaussianReal θ.1 θ.2).toProbabilityMeasure

@[simp] theorem coe_gaussianFamily (θ : ℝ × ℝ≥0) :
    (gaussianFamily θ : Measure ℝ) = gaussianReal θ.1 θ.2 := rfl

/-- Measurability of the Gaussian family allows measurable predictors to define kernels. -/
theorem measurable_gaussianFamily : Measurable gaussianFamily :=
  measurable_gaussianReal.subtype_mk

/-- A mean and a strictly positive variance, for Gaussian density models. -/
abbrev PositiveGaussianParameters := ℝ × {v : ℝ≥0 // 0 < v}

/-- The Gaussian family restricted to parameters admitting a Lebesgue density. -/
noncomputable def positiveGaussianFamily : Family PositiveGaussianParameters ℝ :=
  fun θ => gaussianFamily (θ.1, θ.2.1)

@[simp] theorem coe_positiveGaussianFamily (θ : PositiveGaussianParameters) :
    (positiveGaussianFamily θ : Measure ℝ) = gaussianReal θ.1 θ.2.1 := rfl

/-- Measurable positive-variance parameters can also be used to build kernels. -/
theorem measurable_positiveGaussianFamily : Measurable positiveGaussianFamily :=
  measurable_gaussianFamily.comp
    (measurable_fst.prodMk (measurable_subtype_coe.comp measurable_snd))

/-- The usual Gaussian density, certified against the Gaussian probability family. -/
noncomputable def positiveGaussianDensity : positiveGaussianFamily.Density volume where
  toFun θ := gaussianPDFReal θ.1 θ.2.1
  nonneg θ y := gaussianPDFReal_nonneg θ.1 θ.2.1 y
  measurable θ := measurable_gaussianPDFReal θ.1 θ.2.1
  withDensity_eq θ := (gaussianReal_of_var_ne_zero θ.1 (ne_of_gt θ.2.2)).symm

/-- The second parameter in this formula is the variance. -/
theorem positiveGaussianDensity_apply (θ : PositiveGaussianParameters) (y : ℝ) :
    positiveGaussianDensity θ y =
      (Real.sqrt (2 * Real.pi * (θ.2.1 : ℝ)))⁻¹ *
        Real.exp (-(y - θ.1) ^ 2 / (2 * (θ.2.1 : ℝ))) := rfl

/-- A positive-variance Gaussian has strictly positive density at every output. -/
theorem positiveGaussianDensity_pos (θ : PositiveGaussianParameters) (y : ℝ) :
    0 < positiveGaussianDensity θ y :=
  gaussianPDFReal_pos θ.1 θ.2.1 y (ne_of_gt θ.2.2)

/-- A Gaussian with variance `v` has standard deviation `√v`. -/
@[simp] theorem standardDeviation_gaussianReal (m : ℝ) (v : ℝ≥0) :
    standardDeviation id (gaussianReal m v) = Real.sqrt (v : ℝ) := by
  simp [standardDeviation]

end ProbabilityModels
