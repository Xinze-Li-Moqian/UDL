module

public import SupervisedLearning
public import Mathlib.Algebra.Field.Rat

/-!
# Ten toy regression samples

Handwritten observations near `y = 1 + 2x`, with small fixed perturbations.
These are synthetic training data, stored as exact rational input/output pairs.
-/

@[expose] public section

namespace LossFunctions.Examples.Data

open SupervisedLearning

/-- Ten observations `(x, y)`; inputs lie in `[-1, 1]`. -/
def toyRegressionSamples : Vector (ℚ × ℚ) 10 := #v[
  (-1,     -11 / 10),
  (-4 / 5, -1 / 2),
  (-3 / 5, -1 / 4),
  (-2 / 5,  1 / 4),
  (-1 / 5,  1 / 2),
  ( 1 / 5,  3 / 2),
  ( 2 / 5,  7 / 4),
  ( 3 / 5,  9 / 4),
  ( 4 / 5,  5 / 2),
  ( 1,      31 / 10)]

/-- Package each scalar input and output as the training interface's one-element vectors. -/
def toyRegression : TrainingDataset ℚ 1 1 10 where
  x i := #v[(toyRegressionSamples[i]).1]
  y i := #v[(toyRegressionSamples[i]).2]

end LossFunctions.Examples.Data
