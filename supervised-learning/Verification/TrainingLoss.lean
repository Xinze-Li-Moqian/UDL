module

public import LinearRegressionTraining

/-!
# Training loss verification

A single-step nonincrease assumption gives a loss bound for any finite training run.
`Verification.StepSize` proves this assumption for real-valued linear regression.
-/

@[expose] public section

namespace SupervisedLearning.LinearRegression

variable {Scalar : Type} {I : Nat}
variable [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
variable (data : TrainingDataset Scalar 1 1 I)

variable [OfNat Scalar 2] [LE Scalar] [Std.IsPreorder Scalar]

/-- If every single update does not increase loss, neither does any finite run.
`hStep` is an assumption: this theorem does not establish the gradient formula,
a suitable step size, convergence, or optimality of the returned parameters. -/
theorem trainLinearRegression_loss_le_initial (stepSize : Scalar)
    (hStep : ∀ φ, leastSquaresLoss data (trainLinearRegression data stepSize φ 1) ≤
      leastSquaresLoss data φ)
    (initial : Parameters Scalar) (steps : Nat) :
    leastSquaresLoss data (trainLinearRegression data stepSize initial steps) ≤
      leastSquaresLoss data initial := by
  induction steps with
  | zero => exact Std.le_refl _
  | succ k ih =>
      exact Std.le_trans (hStep (trainLinearRegression data stepSize initial k)) ih

end SupervisedLearning.LinearRegression
