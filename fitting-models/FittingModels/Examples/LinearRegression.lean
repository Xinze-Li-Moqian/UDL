module

public import LossFunctions.Examples.Data.ToyRegression
public import Discriminative.LinearRegressionTraining
public meta import LossFunctions.Examples.Data.ToyRegression
public meta import Discriminative.LinearRegressionTraining
public meta import Batteries.Data.Float.Rat

/-!
# Example: data, loss, training, and prediction

Source: Understanding Deep Learning, Section 6.1.1, equations (6.4)–(6.7).

Reuse the ten synthetic observations in `LossFunctions.Examples.Data.ToyRegression`
and the existing Chapter 2 linear model and training loop:

    f(x, φ) = φ₀ + φ₁ x,
    L[φ] = ∑ᵢ (φ₀ + φ₁ xᵢ − yᵢ)²,
    g₀ = 2∑ᵢ rᵢ,    g₁ = 2∑ᵢ rᵢ xᵢ,    rᵢ = f(xᵢ, φ) − yᵢ,
    φ_next = φ − α g(φ).

The gradient formulas over the reals are already proved in
`Verification.GradientFormula`. This example executes the supplied formulas
over exact rationals; floating-point conversion is used only for display.
The fitted parameters come from a finite run, not an asserted exact optimum.
-/

@[expose] public section

namespace FittingModels.Examples.LinearRegression

open SupervisedLearning
open SupervisedLearning.Discriminative.LinearRegression
open LossFunctions.Examples.Data

/-- Fix the data to obtain the function of parameters that training will minimize. -/
def trainingLoss : LossFunctions.LossFunction ℚ (Parameters ℚ) :=
  leastSquaresLoss toyRegression

/-- Start with the horizontal line y = 0. -/
def initial : Parameters ℚ := ⟨0, 0⟩

/-- The loss is a sum over ten samples, so use the small learning rate 0.025. -/
def learningRate : ℚ := 1 / 40

/-- Compute the model parameters after k updates, using the actual training data. -/
def parametersAfter (k : ℕ) : Parameters ℚ :=
  trainLinearRegression toyRegression learningRate initial k

/-- Fix the parameters obtained after 100 updates. -/
def fittedParameters : Parameters ℚ := parametersAfter 100

/-- The trained model: only the new input remains to be supplied. -/
def fittedModel (x : ℚ) : ℚ :=
  (linearModel #v[x] fittedParameters)[0]

-- Display the observations first, then the optimization trace and a new prediction.
#eval do
  IO.println "Training data (x, y):"
  for sample in toyRegressionSamples.toList do
    IO.println s!"  ({sample.1.toFloat}, {sample.2.toFloat})"
  IO.println "Model: f(x, phi) = intercept + slope*x"
  IO.println "Loss: sum of squared prediction errors on these ten pairs"
  IO.println "Learning rate: 0.025; initial parameters: (0, 0)"
  for k in [0, 1, 2, 5, 10, 25, 100] do
    let φ := parametersAfter k
    let g := leastSquaresGradient toyRegression φ
    IO.println s!"step {k}: phi ~ ({φ.φ₀.toFloat}, {φ.φ₁.toFloat}), gradient ~ ({g.φ₀.toFloat}, {g.φ₁.toFloat}), loss ~ {(trainingLoss φ).toFloat}"
  IO.println s!"Trained model: y ~ {fittedParameters.φ₀.toFloat} + {fittedParameters.φ₁.toFloat}*x"
  IO.println s!"New input x=0.75: predicted y ~ {(fittedModel (3 / 4)).toFloat}"

end FittingModels.Examples.LinearRegression
