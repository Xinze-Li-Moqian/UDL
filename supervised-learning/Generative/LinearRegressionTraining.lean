module

public import Generative.LinearRegression
public import GradientDescent

/-!
# Training generative linear regression

This model supplies its own loss gradient to the shared gradient descent loop
in `optimization/GradientDescent.lean`. The training pairs keep their `(x, y)`
roles: the model predicts `x` from `y`, so the slope gradient is weighted by `y`.
-/

@[expose] public section

namespace SupervisedLearning.Generative.LinearRegression

variable {Scalar : Type} {I : Nat}

/-- Subtract the two parameter coordinates separately. -/
instance [Sub Scalar] : Sub (Parameters Scalar) where
  sub φ ψ := ⟨φ.φ₀ - ψ.φ₀, φ.φ₁ - ψ.φ₁⟩

/-- Multiply both parameter coordinates by the step size. -/
instance [Mul Scalar] : SMul Scalar (Parameters Scalar) where
  smul stepSize φ := ⟨stepSize * φ.φ₀, stepSize * φ.φ₁⟩

variable [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar] [OfNat Scalar 2]

/-- For residuals `rᵢ = g[yᵢ, φ] - xᵢ`, the gradient is
`(2 Σᵢ rᵢ, 2 Σᵢ rᵢ yᵢ)`. `Verification/GradientFormula.lean` verifies
these two partial derivatives over the reals. -/
def leastSquaresGradient (data : TrainingDataset Scalar 1 1 I)
    (φ : Parameters Scalar) : Parameters Scalar :=
  let g := linearModel (Scalar := Scalar)
  let residual := fun i : Fin I => (g (data.y i) φ)[0] - (data.x i)[0]
  { φ₀ := (2 : Scalar) * (List.ofFn residual).sum
    φ₁ := (2 : Scalar) * (List.ofFn fun i : Fin I => residual i * (data.y i)[0]).sum }

/-- Fit the generative parameters using the same optimizer as the discriminative model.
A finite run does not assert exact optimality or a nonzero fitted slope. -/
def trainLinearRegression (data : TrainingDataset Scalar 1 1 I)
    (stepSize : Scalar) (initial : Parameters Scalar) (steps : Nat) : Parameters Scalar :=
  Optimization.gradientDescent (leastSquaresGradient data) stepSize initial steps

end SupervisedLearning.Generative.LinearRegression
