module

public import LossFunctions.Recipe

/-!
# Univariate regression interface

The target is a single real number. Inputs and distribution parameters may
be multidimensional. A model combines an existing probability family, its
chosen density, and a predictor of distribution parameters:

    x, φ → θ(x, φ) → p(y | θ(x, φ)),    y ∈ ℝ.

Each distribution module supplies these choices. Prediction and loss reuse
`Family.predictiveDistribution` and `negativeLogLikelihood`.
-/

@[expose] public section

namespace LossFunctions.UnivariateRegression

open MeasureTheory ProbabilityModels
open scoped LossFunctions ProbabilityModels

/-- A distribution model for a real-valued target; `Θ` need not be one-dimensional. -/
abbrev Model (X Parameters Θ : Type*) := Recipe X ℝ Parameters Θ

variable {X Parameters Θ : Type*} {I : ℕ}

/-- Return the complete predictive distribution at a new input.

    Pr[· | x; model.predictiveDistribution, φ] = family(θ(x, φ)).
-/
def Model.predictiveDistribution (model : Model X Parameters Θ)
    (φ : Parameters) (x : X) : ProbabilityMeasure ℝ :=
  Recipe.predictiveDistribution model φ x

/-- Construct the training loss through the existing negative log-likelihood.

    L[φ] = −∑ᵢ log p(yᵢ | θ(xᵢ, φ)).
-/
noncomputable def Model.loss (model : Model X Parameters Θ)
    (x : Fin I → X) (y : Fin I → ℝ) : LossFunction ℝ Parameters :=
  Recipe.loss model x y

/-- The shared loss formula, for any choice of distribution family. -/
theorem Model.loss_apply (model : Model X Parameters Θ)
    (x : Fin I → X) (y : Fin I → ℝ) (φ : Parameters) :
    let L := model.loss x y
    L[φ] = -∑ i : Fin I, Real.log Pr[y i | model.predictParameters (x i) φ; model.density] :=
  negativeLogLikelihood_apply model.density model.predictParameters x y φ

/-- Equation (5.11): squared error is available for any real-valued predictor.

    S[φ] = ∑ᵢ (yᵢ − f(xᵢ, φ))².
-/
noncomputable def leastSquaresLoss (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → ℝ) : LossFunction ℝ Parameters :=
  empiricalLoss (fun prediction target => (target - prediction) ^ 2) f x y

end LossFunctions.UnivariateRegression
