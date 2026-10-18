module

public import Discriminative.LinearRegression

/-!
# Loss formula verification

The least-squares loss agrees with the expanded expression in equation (2.5).
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

variable {Scalar : Type} {I : Nat}
variable [AddCommMonoid Scalar] [Sub Scalar] [Mul Scalar]
variable (data : TrainingDataset Scalar 1 1 I)

/-- The book's expanded objective, written once for the proofs below. -/
def expandedLeastSquaresLoss : LossFunction Scalar (Parameters Scalar) :=
  fun φ =>
    ∑ i : Fin I,
      let deviation := φ.φ₀ + φ.φ₁ * (data.x i)[0] - (data.y i)[0]
      deviation * deviation

/-- Equation (2.5): the two expressions define the same loss function. -/
theorem leastSquaresLoss_eq_expanded :
    leastSquaresLoss data = expandedLeastSquaresLoss data := rfl

end SupervisedLearning.Discriminative.LinearRegression
