import Lake
open Lake DSL

package udl

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0"

lean_lib Optimization where
  srcDir := "optimization"
  roots := #[`Argmin, `GradientDescent]

@[default_target]
lean_lib SupervisedLearning where
  srcDir := "supervised-learning"
  roots := #[`SupervisedLearning, `Discriminative.Model,
    `Discriminative.LinearRegression, `Discriminative.LinearRegressionTraining,
    `Generative.Model, `Generative.LinearRegression, `Generative.LinearRegressionTraining,
    `Examples, `Verification]

@[default_target]
lean_lib NeuralNetworks where
  srcDir := "neural-networks"
  roots := #[`Bias, `ActivationFunction, `ShallowNetwork,
    `Example.Plot, `Example.WithBias, `Example.WithoutBias]
