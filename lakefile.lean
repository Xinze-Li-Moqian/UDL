import Lake
open Lake DSL

package fintech

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0"

lean_lib Optimization where
  srcDir := "optimization"
  roots := #[`Argmin, `GradientDescent]

@[default_target]
lean_lib SupervisedLearning where
  srcDir := "supervised-learning"
  roots := #[`SupervisedLearning, `LinearRegression, `LinearRegressionTraining,
    `Verification]
