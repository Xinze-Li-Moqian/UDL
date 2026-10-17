module

public import SupervisedLearning
public import Argmin

/-!
# Linear regression

The single-input, single-output model from Chapter 2 of Understanding Deep Learning:
`y = f[x, φ] = φ₀ + φ₁ * x`.

The scalar input and output are represented by one-element vectors to fit `Model`.
-/

@[expose] public section

namespace SupervisedLearning.LinearRegression

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
`List.ofFn` forms one squared deviation per training pair, and `.sum` adds them.
We write the square as `deviation * deviation`; the sum is not divided by `I`.
Lean indexes the `I` pairs from `0` to `I - 1` using `Fin I`.
-/
def leastSquaresLoss [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
    (data : TrainingDataset Scalar 1 1 I) : LossFunction Scalar (Parameters Scalar) :=
  let f := linearModel (Scalar := Scalar)
  fun φ =>
    (List.ofFn fun i : Fin I =>
      let deviation := (f (data.x i) φ)[0] - (data.y i)[0]
      deviation * deviation).sum

/-- The training objective in equation (2.6): `φBest` minimizes this loss.
This is the condition on a least-squares solution, with no claim of existence
or uniqueness and no choice of training algorithm. -/
def IsLeastSquaresMinimizer [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
    [LE Scalar] (data : TrainingDataset Scalar 1 1 I) (φBest : Parameters Scalar) : Prop :=
  let L := leastSquaresLoss data
  Optimization.IsArgmin L φBest

end SupervisedLearning.LinearRegression
