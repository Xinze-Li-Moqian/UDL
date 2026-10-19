module

public import FittingModels.FloatingPoint
public import LossFunctions.Examples.Data.ToyRegression
public import Discriminative.LinearRegressionTraining
public import Batteries.Data.Float.Rat
public meta import FittingModels.FloatingPoint
public meta import LossFunctions.Examples.Data.ToyRegression
public meta import Discriminative.LinearRegressionTraining
public meta import LossFunctions.EmpiricalLoss
public meta import Batteries.Data.Float.Rat

/-!
# Linear regression executed in binary32

The same ten observations, model, gradient formula, and update loop as the
rational example, now using FloatLib operations. Input data and learning rate
are rounded once from their original rational values. Every subsequent
arithmetic operation uses binary32; sums have a fixed right-associated order.

The printed approximate numbers are for readability. Exact decoding of the
final parameters is included for training-result certificates. The companion
`FloatLinearRegressionVerification` checks selected steps of this floating-point
run separately from the existing real-gradient and rational-run proofs.
-/

@[expose] public section

namespace FittingModels.Examples.FloatLinearRegression

open SupervisedLearning
open SupervisedLearning.Discriminative.LinearRegression
open LossFunctions.Examples.Data

/-- Round each existing observation once to the execution format. -/
def data : TrainingDataset Binary32 1 1 10 where
  x i := #v[binary32OfRat toyRegressionSamples[i].1]
  y i := #v[binary32OfRat toyRegressionSamples[i].2]

/-- Evaluate squared errors in a fixed order without assuming associativity. -/
def trainingLoss : LossFunctions.LossFunction Binary32 (Parameters Binary32) :=
  LossFunctions.orderedEmpiricalLoss
    (fun prediction target =>
      let r := prediction[0] - target[0]
      r * r)
    (linearModel (Scalar := Binary32)) data.x data.y

def initial : Parameters Binary32 := ⟨0, 0⟩

/-- The actual binary32 learning rate is the nearest representation of 1/40. -/
def learningRate : Binary32 := binary32OfRat (1 / 40)

/-- Execute the existing gradient formula and update loop in binary32. -/
def parametersAfter (k : ℕ) : Parameters Binary32 :=
  trainLinearRegression data learningRate initial k

def fittedParameters : Parameters Binary32 := parametersAfter 100

/-- Prediction also executes in the same floating-point format. -/
def fittedModel (x : Binary32) : Binary32 :=
  (linearModel #v[x] fittedParameters)[0]

/-- A readable display; exceptional values are shown explicitly. -/
def display (x : Binary32) : String :=
  match binary32ToRat? x with
  | some q => s!"{q.toFloat}"
  | none => "NaN or infinity"

#eval do
  IO.println "FloatLib binary32 linear regression; ten rounded training pairs"
  IO.println s!"Learning rate (exact stored value): {repr (binary32ToRat? learningRate)}"
  for k in [0, 1, 2, 5, 10, 25, 100] do
    let φ := parametersAfter k
    let g := leastSquaresGradient data φ
    IO.println s!"step {k}: phi ~ ({display φ.φ₀}, {display φ.φ₁}), gradient ~ ({display g.φ₀}, {display g.φ₁}), loss ~ {display (trainingLoss φ)}"
  IO.println s!"Trained model: y ~ {display fittedParameters.φ₀} + {display fittedParameters.φ₁}*x"
  IO.println s!"New input x=0.75: predicted y ~ {display (fittedModel (binary32OfRat (3 / 4)))}"
  IO.println s!"Exact stored parameters: {repr (binary32ToRat? fittedParameters.φ₀, binary32ToRat? fittedParameters.φ₁)}"

end FittingModels.Examples.FloatLinearRegression
