module

public import Verification.LossFormula

/-!
# Optimality verification

The two loss expressions have the same minimizer condition, as in equation (2.6).
-/

@[expose] public section

namespace SupervisedLearning.Discriminative.LinearRegression

variable {Scalar : Type} {I : Nat}
variable [AddCommMonoid Scalar] [Sub Scalar] [Mul Scalar]
variable (data : TrainingDataset Scalar 1 1 I)

/-- Equation (2.6): both expressions have the same minimizer condition.
No existence or uniqueness of a minimizer is assumed. -/
theorem isLeastSquaresMinimizer_iff_expanded [LE Scalar] (φBest : Parameters Scalar) :
    IsLeastSquaresMinimizer data φBest ↔
      Optimization.IsArgmin (expandedLeastSquaresLoss data) φBest := Iff.rfl

end SupervisedLearning.Discriminative.LinearRegression
