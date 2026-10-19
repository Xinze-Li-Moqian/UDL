module

public import FittingModels.Examples.LinearRegression
public import Verification.LeastSquaresSolution
public import Mathlib.Tactic.NormNum
public import Mathlib.Data.Rat.BigOperators

/-!
# Verification of the ten-sample linear regression run

Use the same rational observations as the executable example. Their exact
real least-squares solution is (1, 89/44), with minimum loss 149/2200.
Separately check the actual 100-step rational run against explicit tolerances.
The finite result is close to the optimum; it is not claimed to equal it.
-/

@[expose] public section

namespace FittingModels.Examples.LinearRegression

open SupervisedLearning
open SupervisedLearning.Discriminative.LinearRegression
open LossFunctions.Examples.Data

set_option maxRecDepth 10000

/-- Interpret the existing rational observations as real numbers for calculus. -/
def realData : TrainingDataset ℝ 1 1 10 where
  x i := #v[(toyRegressionSamples[i].1 : ℝ)]
  y i := #v[(toyRegressionSamples[i].2 : ℝ)]

/-- The exact least-squares parameters for these observations. -/
noncomputable def exactFit : Parameters ℝ := ⟨1, 89 / 44⟩

/-- The exact fit satisfies both normal equations. -/
theorem exactFit_stationary : IsLeastSquaresStationary realData exactFit := by
  have h :
      (∑ i : Fin 10, ((1 : ℚ) + (89 / 44) * toyRegressionSamples[i].1 -
        toyRegressionSamples[i].2)) = 0 ∧
      (∑ i : Fin 10, ((1 : ℚ) + (89 / 44) * toyRegressionSamples[i].1 -
        toyRegressionSamples[i].2) * toyRegressionSamples[i].1) = 0 := by
    decide +kernel
  change
    (∑ i : Fin 10, ((1 : ℝ) + (89 / 44) * (toyRegressionSamples[i].1 : ℝ) -
      (toyRegressionSamples[i].2 : ℝ))) = 0 ∧
    (∑ i : Fin 10, ((1 : ℝ) + (89 / 44) * (toyRegressionSamples[i].1 : ℝ) -
      (toyRegressionSamples[i].2 : ℝ)) * (toyRegressionSamples[i].1 : ℝ)) = 0
  constructor
  · simpa only [Rat.cast_sum, Rat.cast_sub, Rat.cast_add, Rat.cast_mul,
      Rat.cast_div, Rat.cast_one, Rat.cast_ofNat, Rat.cast_zero] using
      congrArg (fun r : ℚ => (r : ℝ)) h.1
  · simpa only [Rat.cast_sum, Rat.cast_sub, Rat.cast_add, Rat.cast_mul,
      Rat.cast_div, Rat.cast_one, Rat.cast_ofNat, Rat.cast_zero] using
      congrArg (fun r : ℚ => (r : ℝ)) h.2

/-- This is a global loss minimum over all real intercepts and slopes. -/
theorem exactFit_isGlobalMinimizer : IsLeastSquaresMinimizer realData exactFit :=
  isLeastSquaresMinimizer_of_stationary realData exactFit exactFit_stationary

/-- The exact minimum loss is positive because the samples do not lie on one line. -/
theorem exactFit_loss : leastSquaresLoss realData exactFit = (149 / 2200 : ℝ) := by
  have h : (∑ i : Fin 10, ((1 : ℚ) + (89 / 44) * toyRegressionSamples[i].1 -
      toyRegressionSamples[i].2) ^ 2) = 149 / 2200 := by decide +kernel
  change (∑ i : Fin 10, ((1 : ℝ) + (89 / 44) * (toyRegressionSamples[i].1 : ℝ) -
    (toyRegressionSamples[i].2 : ℝ)) *
    ((1 : ℝ) + (89 / 44) * (toyRegressionSamples[i].1 : ℝ) -
      (toyRegressionSamples[i].2 : ℝ))) = 149 / 2200
  simp only [← pow_two]
  simpa only [Rat.cast_sum, Rat.cast_pow, Rat.cast_sub, Rat.cast_add, Rat.cast_mul,
    Rat.cast_div, Rat.cast_one, Rat.cast_ofNat] using
    congrArg (fun r : ℚ => (r : ℝ)) h

set_option maxRecDepth 10000 in
set_option maxHeartbeats 8000000 in
/-- Check the actual computed parameters, not their rounded printed values.
Both parameter errors are below 10⁻⁹; excess loss is below 10⁻¹⁸. -/
theorem fittedParameters_verified :
    |fittedParameters.φ₀ - 1| < (1 / 10 ^ 9 : ℚ) ∧
    |fittedParameters.φ₁ - 89 / 44| < (1 / 10 ^ 9 : ℚ) ∧
    0 ≤ trainingLoss fittedParameters - 149 / 2200 ∧
    trainingLoss fittedParameters - 149 / 2200 < (1 / 10 ^ 18 : ℚ) := by
  decide +kernel

end FittingModels.Examples.LinearRegression
