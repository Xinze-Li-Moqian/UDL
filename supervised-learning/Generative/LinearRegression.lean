module

public import Generative.Model
public import Argmin
public import Mathlib.Algebra.Field.Defs

/-!
# Linear regression as a generative model

The model is `x = g[y, φ] = φ₀ + φ₁ * y`. Its least-squares loss measures
errors in `x`. Once fitted, a nonzero slope permits inference of `y` from `x`.
-/

@[expose] public section

namespace SupervisedLearning.Generative.LinearRegression

/-- The parameters of `x = φ₀ + φ₁ * y`, fitted for the generative direction. -/
structure Parameters (Scalar : Type) where
  /-- The intercept `φ₀`. -/
  φ₀ : Scalar
  /-- The slope `φ₁`. -/
  φ₁ : Scalar

variable {Scalar : Type} {I : Nat}

/-- The generative linear model `g[y, φ] = φ₀ + φ₁ * y`. -/
def linearModel [Add Scalar] [Mul Scalar] : Model Scalar (Parameters Scalar) 1 1 :=
  fun y φ => #v[φ.φ₀ + φ.φ₁ * y[0]]

/-- The loss `L[φ] = Σᵢ (g[yᵢ, φ] - xᵢ)²`, using the original training pairs. -/
def leastSquaresLoss [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
    (data : TrainingDataset Scalar 1 1 I) : LossFunction Scalar (Parameters Scalar) :=
  let g := linearModel (Scalar := Scalar)
  fun φ =>
    (List.ofFn fun i : Fin I =>
      let deviation := (g (data.y i) φ)[0] - (data.x i)[0]
      deviation * deviation).sum

/-- The generative training objective: minimize the loss over the parameters. -/
def IsLeastSquaresMinimizer [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
    [LE Scalar] (data : TrainingDataset Scalar 1 1 I) (φBest : Parameters Scalar) : Prop :=
  let L := leastSquaresLoss data
  Optimization.IsArgmin L φBest

/-- Inference by inversion: `y = g⁻¹[x, φ] = (x - φ₀) / φ₁`.
The scalar type is a field (e.g. ℝ or ℚ), and callers must supply a proof that
the slope is nonzero. Fitting the model alone does not guarantee this condition. -/
def inverseModel [Field Scalar] (x : Input Scalar 1) (φ : Parameters Scalar)
    (_hSlope : φ.φ₁ ≠ 0) : Output Scalar 1 :=
  #v[(x[0] - φ.φ₀) / φ.φ₁]

end SupervisedLearning.Generative.LinearRegression
