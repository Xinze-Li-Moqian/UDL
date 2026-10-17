module

public import ProbabilityModels.Examples.Conditioning

/-!
# A Gaussian predictive density

For a fixed positive variance, the predictor supplies the Gaussian mean.
The resulting density is normalized at every input and generates the same
conditional model as `gaussianPredictionKernel`. The density can be evaluated
at an observed output before defining a likelihood or training loss.
-/

@[expose] public section

namespace ProbabilityModels.Examples

open MeasureTheory
open scoped NNReal

variable {Input Parameters : Type*}

/-- The Gaussian density whose mean is the predictor's output. -/
noncomputable def gaussianPredictionDensity (f : Input → Parameters → ℝ)
    (v : ℝ≥0) (hv : 0 < v) (x : Input) (φ : Parameters) : ℝ → ℝ :=
  positiveGaussianDensity.predictiveDensity (fun x φ => (f x φ, ⟨v, hv⟩)) x φ

/-- Evaluate the predictive density at an observed output `y`. -/
theorem gaussianPredictionDensity_apply (f : Input → Parameters → ℝ)
    (v : ℝ≥0) (hv : 0 < v) (x : Input) (φ : Parameters) (y : ℝ) :
    gaussianPredictionDensity f v hv x φ y =
      (Real.sqrt (2 * Real.pi * (v : ℝ)))⁻¹ *
        Real.exp (-(y - f x φ) ^ 2 / (2 * (v : ℝ))) := rfl

/-- Normalization is inherited from the family-level density certificate. -/
theorem integral_gaussianPredictionDensity (f : Input → Parameters → ℝ)
    (v : ℝ≥0) (hv : 0 < v) (x : Input) (φ : Parameters) :
    ∫ y, gaussianPredictionDensity f v hv x φ y = 1 :=
  positiveGaussianDensity.integral_predictiveDensity_eq_one
    (fun x φ => (f x φ, ⟨v, hv⟩)) x φ

/-- The density and the earlier predictive kernel specify the same distribution. -/
theorem withDensity_gaussianPredictionDensity [MeasurableSpace Input]
    (f : Input → Parameters → ℝ) (v : ℝ≥0) (hv : 0 < v)
    (x : Input) (φ : Parameters) (hf : Measurable (fun x => f x φ)) :
    volume.withDensity (fun y => ENNReal.ofReal (gaussianPredictionDensity f v hv x φ y)) =
      gaussianPredictionKernel f φ v hf x :=
  positiveGaussianDensity.withDensity_predictiveDensity
    (fun x φ => (f x φ, ⟨v, hv⟩)) x φ

end ProbabilityModels.Examples
