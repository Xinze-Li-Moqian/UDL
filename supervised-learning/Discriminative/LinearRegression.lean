module

public import Discriminative.Model
public import LossFunctions.EmpiricalLoss
public import Argmin

/-!
# Discriminative linear regression

The single-input, single-output model from Chapter 2 of Understanding Deep Learning:
`y = f[x, φ] = φ₀ + φ₁ * x`.

The scalar input and output are represented by one-element vectors to fit `Model`.
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

open scoped LossFunctions Optimization

/-- The model parameters `φ = (φ₀, φ₁)`. -/
structure Parameters (Scalar : Type) where
  /-- The intercept `φ₀`. -/
  φ₀ : Scalar
  /-- The slope `φ₁`. -/
  φ₁ : Scalar

variable {Scalar : Type} {I : Nat}

/-- The linear model `f[x, φ] = φ₀ + φ₁ * x`.
For real scalars, a nonzero intercept makes this an affine function of `x`.
The conventional name is linear regression. -/
def linearModel [Add Scalar] [Mul Scalar] : Model Scalar (Parameters Scalar) 1 1 :=
  fun x φ => #v[φ.φ₀ + φ.φ₁ * x[0]]

/-- The least-squares loss from Section 2.2.2, equation (2.5):
`L[φ] = Σᵢ (f[xᵢ, φ] - yᵢ)² = Σᵢ (φ₀ + φ₁ * xᵢ - yᵢ)²`.

Fixing `data` gives a loss function of the parameters `φ`.
`empiricalLoss` sums the single-example squared errors over the training pairs.
We write the square as `deviation * deviation`; the sum is not divided by `I`.
Lean indexes the `I` pairs from `0` to `I - 1` using `Fin I`.
-/
def leastSquaresLoss [AddCommMonoid Scalar] [Sub Scalar] [Mul Scalar]
    (data : TrainingDataset Scalar 1 1 I) : LossFunction Scalar (Parameters Scalar) :=
  LossFunctions.empiricalLoss
    (fun prediction target =>
      let deviation := prediction[0] - target[0]
      deviation * deviation)
    (linearModel (Scalar := Scalar)) data.x data.y

/-- The training objective in equation (2.6): `φBest` minimizes this loss.
This is the condition on a least-squares solution, with no claim of existence
or uniqueness and no choice of training algorithm. -/
def IsLeastSquaresMinimizer [AddCommMonoid Scalar] [Sub Scalar] [Mul Scalar]
    [LE Scalar] (data : TrainingDataset Scalar 1 1 I) (φBest : Parameters Scalar) : Prop :=
  let L := leastSquaresLoss data
  φBest ∈ argmin φ, L[φ]

end SupervisedLearning.Discriminative.LinearRegression
