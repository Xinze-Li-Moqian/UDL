module

public import LossFunctions.MultipleOutputs
public import LossFunctions.UnivariateRegression.Gaussian
public import LossFunctions.Examples.BinaryClassification
public import LossFunctions.Examples.MulticlassClassification

/-!
# Single-output models as one-coordinate product models

The generic `MultipleOutputs.loss_single` theorem applies to every existing
family and certified density. These examples check its use with the concrete
Gaussian, Bernoulli, and softmax categorical prediction interfaces.
-/

@[expose] public section

namespace LossFunctions.Examples.MultipleOutputs

open ProbabilityModels
open scoped NNReal LossFunctions

variable {X Parameters : Type*} {I : ℕ}

/-- A one-coordinate Gaussian product has the original regression loss. -/
theorem gaussian_single_loss (f : X → Parameters → ℝ) (v : {v : ℝ≥0 // 0 < v})
    (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) :
    LossFunctions.MultipleOutputs.loss (fun _ : Fin 1 => positiveGaussianDensity)
        (fun _ => UnivariateRegression.gaussianParameters f v) x (fun i _ => y i) φ =
      (UnivariateRegression.gaussianLoss f v x y)[φ] :=
  LossFunctions.MultipleOutputs.loss_single positiveGaussianDensity
    (UnivariateRegression.gaussianParameters f v) x y φ

/-- A one-coordinate Bernoulli product has the original binary cross-entropy loss. -/
theorem bernoulli_single_loss (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → Fin 2) (φ : Parameters) :
    LossFunctions.MultipleOutputs.loss (fun _ : Fin 1 => bernoulliMass)
        (fun _ => BinaryClassification.predictedProbability f) x (fun i _ => y i) φ =
      (BinaryClassification.crossEntropyLoss f x y)[φ] :=
  LossFunctions.MultipleOutputs.loss_single bernoulliMass
    (BinaryClassification.predictedProbability f) x y φ

/-- A one-coordinate categorical product has the original multiclass loss. -/
theorem categorical_single_loss {K : ℕ} [NeZero K]
    (f : X → Parameters → Fin K → ℝ) (x : Fin I → X) (y : Fin I → Fin K)
    (φ : Parameters) :
    LossFunctions.MultipleOutputs.loss (fun _ : Fin 1 => categoricalMass)
        (fun _ => MulticlassClassification.predictedParameters f) x (fun i _ => y i) φ =
      (MulticlassClassification.crossEntropyLoss f x y)[φ] :=
  LossFunctions.MultipleOutputs.loss_single categoricalMass
    (MulticlassClassification.predictedParameters f) x y φ

end LossFunctions.Examples.MultipleOutputs
