import Lake
open Lake DSL

package udl

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0"

lean_lib Optimization where
  srcDir := "optimization"
  roots := #[`Argmin, `Argmax, `GradientDescent]

@[default_target]
lean_lib ProbabilityModels where
  srcDir := "probability"
  roots := #[`ProbabilityModels]

@[default_target]
lean_lib LossFunctions where
  srcDir := "loss-functions"
  roots := #[`LossFunctions]

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
  roots := #[`Bias, `ActivationFunction, `Network, `ShallowNetwork, `ComposedNetwork, `DeepNetwork,
    `Example.Plot, `Example.WithBias, `Example.WithoutBias]
