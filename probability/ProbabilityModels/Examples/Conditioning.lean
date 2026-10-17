module

public import ProbabilityModels.ConditionalDistribution
public import ProbabilityModels.Notation

/-!
# A Gaussian conditional model

A measurable predictor of the mean, together with a fixed variance, gives a
Markov kernel from inputs to real outputs. Its mean at each input is exactly
the predictor's output. This illustrates the probabilistic model used before
defining a Gaussian likelihood.

This is a specified predictive model. Equating it to the conditional law of
an independently given data distribution would be an additional assumption.
-/

@[expose] public section

namespace ProbabilityModels.Examples

open MeasureTheory ProbabilityTheory
open scoped ProbabilityModels NNReal

variable {Input Parameters : Type*} [MeasurableSpace Input]

/-- A measurable mean predictor and a fixed variance define a conditional model. -/
noncomputable def gaussianPredictionKernel (f : Input → Parameters → ℝ)
    (φ : Parameters) (v : ℝ≥0) (hf : Measurable (fun x => f x φ)) :
    Kernel Input ℝ :=
  gaussianFamily.predictiveKernel measurable_gaussianFamily
    (fun x φ => (f x φ, v)) φ (hf.prodMk measurable_const)

instance gaussianPredictionKernel_isMarkovKernel (f : Input → Parameters → ℝ)
    (φ : Parameters) (v : ℝ≥0) (hf : Measurable (fun x => f x φ)) :
    IsMarkovKernel (gaussianPredictionKernel f φ v hf) := by
  unfold gaussianPredictionKernel
  infer_instance

/-- At each input the model selects the Gaussian with the predicted mean. -/
@[simp] theorem gaussianPredictionKernel_apply (f : Input → Parameters → ℝ)
    (φ : Parameters) (v : ℝ≥0) (hf : Measurable (fun x => f x φ)) (x : Input) :
    gaussianPredictionKernel f φ v hf x = 𝒩(f x φ, v) := rfl

/-- The predicted distribution's mean agrees with the deterministic model output. -/
theorem gaussianPredictionKernel_mean (f : Input → Parameters → ℝ)
    (φ : Parameters) (v : ℝ≥0) (hf : Measurable (fun x => f x φ)) (x : Input) :
    𝔼[id; gaussianPredictionKernel f φ v hf x] = f x φ := by
  rw [gaussianPredictionKernel_apply]
  exact integral_id_gaussianReal

end ProbabilityModels.Examples
