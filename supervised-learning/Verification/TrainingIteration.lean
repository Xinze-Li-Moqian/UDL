module

public import Discriminative.LinearRegressionTraining

/-!
# Training iteration verification

Zero steps preserve the initial parameters, and training can be continued in stages.
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

variable {Scalar : Type} {I : Nat}
variable [Zero Scalar] [Add Scalar] [Sub Scalar] [Mul Scalar]
variable (data : TrainingDataset Scalar 1 1 I)

variable [OfNat Scalar 2]

/-- Zero training steps preserve the initial parameters. -/
theorem trainLinearRegression_zero_steps (stepSize : Scalar) (initial : Parameters Scalar) :
    trainLinearRegression data stepSize initial 0 = initial := rfl

/-- With the same data and step size, training can be continued in two stages. -/
theorem trainLinearRegression_add_steps (stepSize : Scalar)
    (initial : Parameters Scalar) (m n : Nat) :
    trainLinearRegression data stepSize initial (m + n) =
      trainLinearRegression data stepSize
        (trainLinearRegression data stepSize initial m) n := by
  induction n with
  | zero => rfl
  | succ n ih =>
      exact congrArg (Optimization.gradientDescentStep (leastSquaresGradient data) stepSize) ih

end SupervisedLearning.Discriminative.LinearRegression
