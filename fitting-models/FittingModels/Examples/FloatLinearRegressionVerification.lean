module

public import FittingModels.Examples.FloatLinearRegression
public import FittingModels.Examples.LinearRegressionVerification

/-!
# Error bounds for floating-point training checkpoints

Decode the actual binary32 parameters, rather than their printed decimals.
Compare them both with the exact optimum and with exact rational training
after the same number of updates. Loss bounds use the original rational
observations and exact arithmetic, not the rounded training-loss display.
-/

@[expose] public section

namespace FittingModels.Examples.FloatLinearRegression

open SupervisedLearning.Discriminative.LinearRegression

/-- Exact values of the actual floating-point parameters, if both are finite. -/
def decodedParametersAfter (k : ℕ) : Option (Parameters ℚ) := do
  let φ := parametersAfter k
  let a ← binary32ToRat? φ.φ₀
  let b ← binary32ToRat? φ.φ₁
  return ⟨a, b⟩

/-- Bounds on total parameter error, error relative to exact training, and
excess loss on the original data. Exceptional floating-point values fail. -/
def CheckpointBounds (k : ℕ) (parameterError roundingError lossError : ℚ) : Prop :=
  match decodedParametersAfter k with
  | none => False
  | some φ =>
      |φ.φ₀ - 1| ≤ parameterError ∧ |φ.φ₁ - 89 / 44| ≤ parameterError ∧
      |φ.φ₀ - (LinearRegression.parametersAfter k).φ₀| ≤ roundingError ∧
      |φ.φ₁ - (LinearRegression.parametersAfter k).φ₁| ≤ roundingError ∧
      0 ≤ LinearRegression.trainingLoss φ - 149 / 2200 ∧
      LinearRegression.trainingLoss φ - 149 / 2200 ≤ lossError

instance (k : ℕ) (parameterError roundingError lossError : ℚ) :
    Decidable (CheckpointBounds k parameterError roundingError lossError) := by
  unfold CheckpointBounds
  split <;> infer_instance

set_option maxRecDepth 20000 in
set_option maxHeartbeats 2000000 in
/-- Check the actual first binary32 update using kernel reduction. -/
theorem checkpoint_one : CheckpointBounds 1 (8 / 5) (1 / 10 ^ 6) 14 := by
  decide +kernel

set_option maxRecDepth 20000 in
set_option maxHeartbeats 8000000 in
/-- The actual 100-step binary32 output is close to the optimum and exact run. -/
theorem checkpoint_hundred :
    CheckpointBounds 100 (1 / 10 ^ 6) (1 / 10 ^ 6) (1 / 10 ^ 11) := by
  decide +kernel

/-- Rows are (step count, total parameter-error bound, excess-loss bound).
The error relative to exact training is bounded separately by 10⁻⁶ in each row. -/
def checkpoints : List (ℕ × ℚ × ℚ) :=
  [(0, 3, 29), (1, 8 / 5, 14), (2, 5 / 4, 8), (5, 3 / 5, 8 / 5),
    (10, 1 / 5, 13 / 100), (25, 1 / 200, 1 / 10000),
    (100, 1 / 10 ^ 6, 1 / 10 ^ 11)]

set_option maxRecDepth 20000 in
set_option maxHeartbeats 8000000 in
/-- Each listed checkpoint certifies the actual program's output, including finiteness. -/
theorem checkpoints_verified : ∀ row ∈ checkpoints,
    CheckpointBounds row.1 row.2.1 (1 / 10 ^ 6) row.2.2 := by
  intro row hrow
  simp only [checkpoints, List.mem_cons, List.not_mem_nil, or_false] at hrow
  rcases hrow with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · decide +kernel
  · exact checkpoint_one
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · decide +kernel
  · exact checkpoint_hundred

end FittingModels.Examples.FloatLinearRegression
