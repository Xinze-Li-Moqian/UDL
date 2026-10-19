module

public import Discriminative.Gabor
public import LossFunctions.UnivariateRegression.Gaussian

@[expose] public section

namespace FittingModels.Examples.Gabor

open SupervisedLearning
open SupervisedLearning.Discriminative.Gabor
open LossFunctions ProbabilityModels
open scoped NNReal

/-- Select the scalar output from the shared model's one-element output vector. -/
noncomputable def prediction (x : Input ℝ 1) (φ : Parameters) : ℝ :=
  (gaborModel x φ)[0]

/-- Reuse the Gaussian recipe: the Gabor prediction is the mean, v is fixed.

    Y | x, φ ∼ Normal(f[x, φ], v),    v > 0.
-/
noncomputable def recipe (v : {v : ℝ≥0 // 0 < v}) :
    Recipe (Input ℝ 1) ℝ Parameters PositiveGaussianParameters :=
  UnivariateRegression.gaussianModel prediction v

/-- The recipe constructs the negative log-likelihood from the training data. -/
noncomputable def loss {I : ℕ} (v : {v : ℝ≥0 // 0 < v})
    (data : TrainingDataset ℝ 1 1 I) : LossFunctions.LossFunction ℝ Parameters :=
  (recipe v).loss data.x (fun i => (data.y i)[0])

/-- Equation (6.9): the book's squared-error loss, using the shared constructor.

    S[φ] = ∑ᵢ (yᵢ − f[xᵢ, φ])².

The existing fixed-variance Gaussian theorem gives the same minimizers for
this loss and the recipe's negative log-likelihood.
-/
noncomputable def leastSquaresLoss {I : ℕ} (data : TrainingDataset ℝ 1 1 I) :
    LossFunctions.LossFunction ℝ Parameters :=
  UnivariateRegression.leastSquaresLoss prediction data.x (fun i => (data.y i)[0])

end FittingModels.Examples.Gabor
