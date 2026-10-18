module

public import LossFunctions.UnivariateRegression.Model
public import LossFunctions.UnivariateRegression.Heteroscedastic.Gaussian
public import LossFunctions.Inference
public import ProbabilityModels.Gaussian

/-!
# Gaussian univariate regression

Source: Understanding Deep Learning, Sections 5.3–5.3.3, equations (5.8)–(5.13).

The output is a single real number; the input type is unrestricted. The model
predicts a Gaussian mean. The first construction fixes a positive variance:

    Y | x, φ ∼ 𝒩(f(x, φ), v),    v = σ² > 0.

Definitions specify the predictive model and its losses. Theorems verify the
loss identity, equality of minimizers, and the mode prediction for any supplied
predictor `f`. Fitting parameters to data requires a separate training algorithm.
Section 5.3.3 then makes the variance part of the fitted parameters, while
keeping it shared across inputs.
-/

@[expose] public section

namespace LossFunctions.UnivariateRegression

open MeasureTheory ProbabilityModels
open scoped NNReal LossFunctions ProbabilityModels Optimization

variable {X Parameters : Type*} {I : ℕ}

/-- The network supplies the mean; the variance is fixed.

    θ(x, φ) = (f(x, φ), v),    v = σ² > 0.
-/
def gaussianParameters (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : X) (φ : Parameters) : PositiveGaussianParameters :=
  (f x φ, v)

/-- A Gaussian instance of the shared regression interface.

    θ(x, φ) = (f(x, φ), v),    Y | x, φ ∼ 𝒩(f(x, φ), v).

This specializes the input-dependent model to a constant variance predictor.
The distribution family and density are supplied by `ProbabilityModels.Gaussian`.
-/
noncomputable def gaussianModel (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v}) :
    Model X Parameters PositiveGaussianParameters :=
  Heteroscedastic.gaussianModel f (fun _ _ => v)

/-- Equation (5.8): substitute the predicted mean into the Gaussian density.

    p(y | f(x, φ), v) = exp(−(y − f(x, φ))² / (2v)) / √(2πv).
-/
theorem predictiveDensity_eq (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : X) (φ : Parameters) (y : ℝ) :
    Pr[y | gaussianParameters f v x φ; positiveGaussianDensity] =
      (Real.sqrt (2 * Real.pi * (v.1 : ℝ)))⁻¹ *
        Real.exp (-(y - f x φ) ^ 2 / (2 * (v.1 : ℝ))) := rfl

/-- Equation (5.9): the Gaussian negative log-likelihood, with the data fixed.

    L[φ] = −∑ᵢ log p(yᵢ | f(xᵢ, φ), v).

This specializes the existing loss construction to the Gaussian density.
-/
noncomputable def gaussianLoss (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : Fin I → X) (y : Fin I → ℝ) : LossFunction ℝ Parameters :=
  (gaussianModel f v).loss x y

/-- The exact relation between Gaussian negative log-likelihood and squared error.

    L[φ] = (I / 2) log(2πv) + S[φ] / (2v).

Both the additive constant and the positive scale factor are independent of `φ`.
-/
theorem gaussianLoss_eq (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) :
    let L := gaussianLoss f v x y
    let S := leastSquaresLoss f x y
    L[φ] = (I : ℝ) / 2 * Real.log (2 * Real.pi * (v.1 : ℝ)) +
      S[φ] / (2 * (v.1 : ℝ)) := by
  change (∑ i : Fin I, -Real.log
    (positiveGaussianDensity (gaussianParameters f v (x i) φ) (y i))) = _
  simp_rw [neg_log_positiveGaussianDensity]
  simp only [gaussianParameters, Finset.sum_add_distrib, Finset.sum_const,
    Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, ← Finset.sum_div,
    leastSquaresLoss, empiricalLoss, lossFunction_getElem]
  ring

/-- Equation (5.10): fixed positive variance gives exactly the same minimizers.

    φ̂ ∈ argmin_φ L[φ]  ↔  φ̂ ∈ argmin_φ S[φ].

This proves equivalence of optimality conditions; it does not compute a minimizer.
-/
theorem mem_argmin_gaussianLoss_iff (f : X → Parameters → ℝ)
    (v : {v : ℝ≥0 // 0 < v}) (x : Fin I → X) (y : Fin I → ℝ) («φ̂» : Parameters) :
    let L := gaussianLoss f v x y
    let S := leastSquaresLoss f x y
    («φ̂» ∈ argmin φ, L[φ]) ↔ («φ̂» ∈ argmin φ, S[φ]) := by
  change (∀ φ, gaussianLoss f v x y «φ̂» ≤ gaussianLoss f v x y φ) ↔
    (∀ φ, leastSquaresLoss f x y «φ̂» ≤ leastSquaresLoss f x y φ)
  have hv : 0 < (v.1 : ℝ) := v.2
  refine forall_congr' fun φ => ?_
  change (gaussianLoss f v x y)[«φ̂»] ≤ (gaussianLoss f v x y)[φ] ↔ _
  rw [gaussianLoss_eq, gaussianLoss_eq, add_le_add_iff_left]
  exact div_le_div_iff_of_pos_right (mul_pos (by norm_num) hv)

/-- Equation (5.12): the predicted mean is a mode of the predictive Gaussian.

    ŷ = f(x, φ̂) ∈ argmax_y p(y | f(x, φ̂), v),    y ∈ ℝ.
-/
theorem mean_isModePrediction (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : X) («φ̂» : Parameters) :
    IsModePrediction positiveGaussianDensity (gaussianParameters f v) x «φ̂» (f x «φ̂») :=
  gaussian_mean_isModePrediction (gaussianParameters f v) x «φ̂»

/-!
## Estimating variance

Optimize the mean-model parameters and one shared positive variance together:

    ψ = (φ, v),    (φ̂, v̂) ∈ argmin_(φ,v) L[φ, v].

The normalization term now depends on an optimized variable, so it stays
in the loss. This specifies equation (5.13), without supplying a training
algorithm or assuming that a minimizer exists.
-/

/-- The fitted parameters now include the shared variance `v = σ² > 0`. -/
abbrev GaussianFitParameters (Parameters : Type*) :=
  Parameters × {v : ℝ≥0 // 0 < v}

/-- The same Gaussian family, with variance supplied by the parameter pair.

    θ(x, ψ) = (f(x, φ), v),    ψ = (φ, v).

The variance can change during fitting but is shared by every input.
-/
noncomputable def gaussianModelWithLearnedVariance (f : X → Parameters → ℝ) :
    Model X (GaussianFitParameters Parameters) PositiveGaussianParameters :=
  Heteroscedastic.gaussianModel
    (fun x ψ => f x ψ.1) (fun _ ψ => ψ.2)

/-- Equation (5.13): reuse the shared loss with both `φ` and `v` as arguments.

    L[φ, v] = −∑ᵢ log p(yᵢ | f(xᵢ, φ), v).
-/
noncomputable def jointGaussianLoss (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → ℝ) : LossFunction ℝ (GaussianFitParameters Parameters) :=
  (gaussianModelWithLearnedVariance f).loss x y

/-- Fixing the variance in the joint objective recovers the earlier loss. -/
theorem jointGaussianLoss_apply (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) (v : {v : ℝ≥0 // 0 < v}) :
    jointGaussianLoss f x y (φ, v) = gaussianLoss f v x y φ := rfl

/-- Both variance-dependent terms must be retained when optimizing `v`.

    L[φ, v] = (I / 2) log(2πv) + S[φ] / (2v).
-/
theorem jointGaussianLoss_eq (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → ℝ) (ψ : GaussianFitParameters Parameters) :
    let L := jointGaussianLoss f x y
    let S := leastSquaresLoss f x y
    L[ψ] = (I : ℝ) / 2 * Real.log (2 * Real.pi * (ψ.2.1 : ℝ)) +
      S[ψ.1] / (2 * (ψ.2.1 : ℝ)) :=
  gaussianLoss_eq f ψ.2 x y ψ.1

/-- The candidate pair minimizes the loss jointly over model parameters and variance.

    (φ̂, v̂) ∈ argmin_(φ,v) L[φ, v].
-/
def IsGaussianJointEstimator (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → ℝ) («φ̂» : Parameters) (vHat : {v : ℝ≥0 // 0 < v}) : Prop :=
  let L := jointGaussianLoss f x y
  («φ̂», vHat) ∈ argmin ψ, L[ψ]

/-- Inference uses the fitted variance in the distribution and the mean as its mode.

    Y | x, ψ̂ ∼ 𝒩(f(x, φ̂), v̂),    ŷ = f(x, φ̂).
-/
theorem learnedVariance_mean_isModePrediction (f : X → Parameters → ℝ)
    (x : X) (ψHat : GaussianFitParameters Parameters) :
    IsModePrediction positiveGaussianDensity
      (gaussianModelWithLearnedVariance f).predictParameters x ψHat (f x ψHat.1) :=
  gaussian_mean_isModePrediction (gaussianModelWithLearnedVariance f).predictParameters x ψHat

end LossFunctions.UnivariateRegression
