import Lake
open Lake DSL

package udl where
  license := "Apache-2.0"

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.35.0"

require floatlib from git
  "https://github.com/lean-dojo/FloatLib.git" @ "1f5771b3e36a98f17fcac2c6d76aaa024b36aa37"

lean_lib Optimization where
  srcDir := "optimization"
  roots := #[`Argmin, `Argmax, `GradientDescent]

@[default_target]
lean_lib ProbabilityModels where
  srcDir := "probability"
  roots := #[`ProbabilityModels]

@[default_target]
lean_lib InformationTheoryModels where
  srcDir := "information-theory"
  roots := #[`InformationTheoryModels]

@[default_target]
lean_lib LossFunctions where
  srcDir := "loss-functions"
  roots := #[`LossFunctions]

@[default_target]
lean_lib FittingModels where
  srcDir := "fitting-models"
  roots := #[`FittingModels, `FittingModels.Examples.Quadratic,
    `FittingModels.Examples.LinearRegression,
    `FittingModels.Examples.LinearRegressionVerification,
    `FittingModels.Examples.FloatLinearRegression,
    `FittingModels.Examples.FloatLinearRegressionVerification,
    `FittingModels.Examples.Gabor, `FittingModels.Examples.GaborNonconvex,
    `FittingModels.Examples.SaddlePoint,
    `FittingModels.Examples.GaborLocalMinimum]

@[default_target]
lean_lib SupervisedLearning where
  srcDir := "supervised-learning"
  roots := #[`SupervisedLearning, `Discriminative.Model,
    `Discriminative.Gabor,
    `Discriminative.LinearRegression, `Discriminative.LinearRegressionTraining,
    `Generative.Model, `Generative.LinearRegression, `Generative.LinearRegressionTraining,
    `Examples, `Verification]

@[default_target]
lean_lib NeuralNetworks where
  srcDir := "neural-networks"
  roots := #[`Bias, `ActivationFunction, `ActivationFunction.Sigmoid, `ActivationFunction.Softmax,
    `Network, `ShallowNetwork, `ComposedNetwork, `DeepNetwork,
    `Example.Plot, `Example.WithBias, `Example.WithoutBias]
