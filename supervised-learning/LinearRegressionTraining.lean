module

public import LinearRegression
public import GradientDescent

/-! Gradient descent for the linear regression model in Figure 2.4. -/

@[expose] public section

namespace SupervisedLearning.LinearRegression

variable {Scalar : Type} {I : Nat}

/-- Subtract the two parameter coordinates separately. -/
instance [Sub Scalar] : Sub (Parameters Scalar) where
  sub φ ψ := ⟨φ.φ₀ - ψ.φ₀, φ.φ₁ - ψ.φ₁⟩

/-- Multiply both parameter coordinates by the step size. -/
instance [Mul Scalar] : SMul Scalar (Parameters Scalar) where
  smul stepSize φ := ⟨stepSize * φ.φ₀, stepSize * φ.φ₁⟩

variable [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar] [OfNat Scalar 2]

/-- The real-valued least-squares gradient formula:
`∂L/∂φ₀ = 2 Σᵢ rᵢ` and `∂L/∂φ₁ = 2 Σᵢ rᵢ xᵢ`, where `rᵢ = f[xᵢ, φ] - yᵢ`.
For `Scalar = ℝ`, `Verification/GradientFormula.lean` proves that these coordinates are the
corresponding partial derivatives of `leastSquaresLoss`. -/
def leastSquaresGradient (data : TrainingDataset Scalar 1 1 I)
    (φ : Parameters Scalar) : Parameters Scalar :=
  let f := linearModel (Scalar := Scalar)
  let residual := fun i : Fin I => (f (data.x i) φ)[0] - (data.y i)[0]
  { φ₀ := (2 : Scalar) * (List.ofFn residual).sum
    φ₁ := (2 : Scalar) * (List.ofFn fun i : Fin I => residual i * (data.x i)[0]).sum }

/-- Run a specified number of gradient descent updates on the training data.
The returned parameters are not asserted to satisfy `IsArgmin`. -/
def trainLinearRegression (data : TrainingDataset Scalar 1 1 I)
    (stepSize : Scalar) (initial : Parameters Scalar) (steps : Nat) : Parameters Scalar :=
  Optimization.gradientDescent (leastSquaresGradient data) stepSize initial steps

end SupervisedLearning.LinearRegression
