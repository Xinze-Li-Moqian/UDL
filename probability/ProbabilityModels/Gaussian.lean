module

public import ProbabilityModels.Family
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

/-- A Gaussian with variance `v` has standard deviation `√v`. -/
@[simp] theorem standardDeviation_gaussianReal (m : ℝ) (v : ℝ≥0) :
    standardDeviation id (gaussianReal m v) = Real.sqrt (v : ℝ) := by
  simp [standardDeviation]

end ProbabilityModels
