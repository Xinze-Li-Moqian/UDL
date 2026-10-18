module

public import LossFunctions.UnivariateRegression.Model
public import LossFunctions.Inference
public import ProbabilityModels.Gaussian

/-!
# Gaussian regression with input-dependent variance

Source: Understanding Deep Learning, Section 5.3.4, equations (5.14)–(5.15).

    Y | x, φ ∼ 𝒩(μ(x, φ), v(x, φ)),    v(x, φ) > 0.

Both mean and variance may depend on the input and model parameters. A constant
variance predictor recovers homoscedastic regression; actual variation is not
required by the interface. The target is still one real number.
-/

@[expose] public section

namespace LossFunctions.UnivariateRegression.Heteroscedastic

open MeasureTheory ProbabilityModels
open scoped NNReal LossFunctions ProbabilityModels

variable {X Parameters : Type*} {I : ℕ}

/-- Use the predicted mean and positive variance as distribution parameters.

    θ(x, φ) = (μ(x, φ), v(x, φ)).
-/
def gaussianParameters (μ : X → Parameters → ℝ)
    (v : X → Parameters → {v : ℝ≥0 // 0 < v}) (x : X) (φ : Parameters) :
    PositiveGaussianParameters :=
  (μ x φ, v x φ)

/-- Instantiate the shared regression interface with input-dependent Gaussian parameters.

    p(y | x, φ) = p(y | μ(x, φ), v(x, φ)).

The family and certified density come from `ProbabilityModels.Gaussian`.
-/
noncomputable def gaussianModel (μ : X → Parameters → ℝ)
    (v : X → Parameters → {v : ℝ≥0 // 0 < v}) :
    Model X Parameters PositiveGaussianParameters where
  family := positiveGaussianFamily
  referenceMeasure := volume
  density := positiveGaussianDensity
  predictParameters := gaussianParameters μ v

/-- Reuse the shared negative log-likelihood for the predicted mean and variance.

    L[φ] = −∑ᵢ log p(yᵢ | μ(xᵢ, φ), v(xᵢ, φ)).
-/
noncomputable def gaussianLoss (μ : X → Parameters → ℝ)
    (v : X → Parameters → {v : ℝ≥0 // 0 < v})
    (x : Fin I → X) (y : Fin I → ℝ) : LossFunction ℝ Parameters :=
  (gaussianModel μ v).loss x y

/-- Expand the loss using the existing Gaussian negative log-density identity.

    L[φ] = ∑ᵢ [log(2πvᵢ) / 2 + (yᵢ − μᵢ)² / (2vᵢ)],
    μᵢ = μ(xᵢ, φ),    vᵢ = v(xᵢ, φ).

The variance-dependent normalization and residual scale both remain in the sum.
-/
theorem gaussianLoss_eq (μ : X → Parameters → ℝ)
    (v : X → Parameters → {v : ℝ≥0 // 0 < v})
    (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) :
    let L := gaussianLoss μ v x y
    L[φ] = ∑ i : Fin I,
      (Real.log (2 * Real.pi * ((v (x i) φ).1 : ℝ)) / 2 +
        (y i - μ (x i) φ) ^ 2 / (2 * ((v (x i) φ).1 : ℝ))) := by
  change (∑ i : Fin I, -Real.log
    (positiveGaussianDensity (μ (x i) φ, v (x i) φ) (y i))) = _
  simp only [neg_log_positiveGaussianDensity]

/-- The predicted mean is a mode even when the variance depends on the input.

    ŷ = μ(x, φ̂) ∈ argmax_y p(y | μ(x, φ̂), v(x, φ̂)).
-/
theorem mean_isModePrediction (μ : X → Parameters → ℝ)
    (v : X → Parameters → {v : ℝ≥0 // 0 < v}) (x : X) («φ̂» : Parameters) :
    IsModePrediction positiveGaussianDensity (gaussianParameters μ v) x «φ̂» (μ x «φ̂») :=
  gaussian_mean_isModePrediction (gaussianParameters μ v) x «φ̂»

/-- Equation (5.14): turn two network outputs into the mean and variance.

    μ(x, φ) = f₁(x, φ),    v(x, φ) = f₂(x, φ)².

Lean indexes these outputs by `0` and `1`. The second output must be nonzero:
squaring alone only guarantees nonnegativity, whereas this density needs v > 0.
-/
noncomputable def squaredOutputModel (f : X → Parameters → Fin 2 → ℝ)
    (hne : ∀ x φ, f x φ 1 ≠ 0) : Model X Parameters PositiveGaussianParameters :=
  gaussianModel (fun x φ => f x φ 0)
    (fun x φ => ⟨⟨(f x φ 1) ^ 2, sq_nonneg _⟩, by
      exact_mod_cast sq_pos_of_ne_zero (hne x φ)⟩)

/-- Equation (5.15), after expanding the Gaussian negative logarithm.

    L[φ] = ∑ᵢ [log(2π f₂(xᵢ, φ)²) / 2
                 + (yᵢ − f₁(xᵢ, φ))² / (2 f₂(xᵢ, φ)²)].
-/
theorem squaredOutputModel_loss_eq (f : X → Parameters → Fin 2 → ℝ)
    (hne : ∀ x φ, f x φ 1 ≠ 0) (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) :
    let L := (squaredOutputModel f hne).loss x y
    L[φ] = ∑ i : Fin I,
      (Real.log (2 * Real.pi * (f (x i) φ 1) ^ 2) / 2 +
        (y i - f (x i) φ 0) ^ 2 / (2 * (f (x i) φ 1) ^ 2)) := by
  change (gaussianLoss (fun x φ => f x φ 0) _ x y)[φ] = _
  exact gaussianLoss_eq _ _ x y φ

end LossFunctions.UnivariateRegression.Heteroscedastic
