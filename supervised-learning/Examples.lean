module

public meta import Discriminative.LinearRegressionTraining
public meta import Generative.LinearRegressionTraining
public meta import Verification.ModelComparison
public meta import Mathlib.Algebra.Field.Rat
public meta import Batteries.Data.Float.Rat

/-!
# The same formula in different modeling scenarios

The first three examples use the arithmetic rule `h(z) = 1 + 2 * z`.
Those coefficients are chosen for illustration, not fitted to a dataset.

In a toy car-position example, keep the variable meanings fixed:
- `x`: an observed sensor reading.
- `y`: the car's position along one axis, which we want to infer.

The model direction determines which quantity the formula takes and produces.
We use rational numbers (`ℚ`) for exact arithmetic. The final example trains
both models on the same three pairs and displays decimal approximations.
-/

namespace SupervisedLearning.Examples

/- Discriminative: y = f[x, φ] = 1 + 2*x.
   A sensor reading x = 3 predicts position y = 7. -/
#eval
  let f := Discriminative.LinearRegression.linearModel (Scalar := ℚ)
  let φ : Discriminative.LinearRegression.Parameters ℚ := ⟨1, 2⟩
  let x : Input ℚ 1 := #v[3]
  (f x φ)[0] -- 7

/- Generative: x = g[y, φ] = 1 + 2*y.
   Position y = 3 predicts sensor reading x = 7.
   Here we evaluate the forward model; we have not inferred a position from a reading. -/
#eval
  let g := Generative.LinearRegression.linearModel (Scalar := ℚ)
  let φ : Generative.LinearRegression.Parameters ℚ := ⟨1, 2⟩
  let y : Output ℚ 1 := #v[3]
  (g y φ)[0] -- 7

/- Generative inference: y = g⁻¹[x, φ] = (x - 1)/2.
   A sensor reading x = 3 now predicts position y = 1.
   The slope is nonzero, so the inverse can be used. -/
#eval
  let gInverse := Generative.LinearRegression.inverseModel (Scalar := ℚ)
  let φ : Generative.LinearRegression.Parameters ℚ := ⟨1, 2⟩
  let x : Input ℚ 1 := #v[3]
  (gInverse x φ (by decide))[0] -- 1

/- The arithmetic rule is the same, but y = 1 + 2*x and x = 1 + 2*y are
   different relations between the observed x and the desired y. -/

/- Training: fit both models to (0, 0), (1, 1), (2, 1).
   Both start at (0, 0), use step size 0.05, and run the same optimizer for 500 steps.
   Each model supplies its own gradient. The printed losses measure errors in
   different quantities, so compare each model's loss before and after training.
   Training uses exact rational arithmetic; only the displayed values are rounded.
   This finite run illustrates learning; it is not a convergence proof. -/
#eval do
  let data := ModelComparison.threePoints ℚ
  let stepSize : ℚ := 1 / 20
  let steps := 500
  let x : Input ℚ 1 := #v[3 / 2]

  let initialD : Discriminative.LinearRegression.Parameters ℚ := ⟨0, 0⟩
  let trainedD := Discriminative.LinearRegression.trainLinearRegression
    data stepSize initialD steps
  let Ld := Discriminative.LinearRegression.leastSquaresLoss data
  let f := Discriminative.LinearRegression.linearModel (Scalar := ℚ)
  IO.println s!"Discriminative y = phi0 + phi1*x ({steps} updates):"
  IO.println s!"  phi ≈ ({trainedD.φ₀.toFloat}, {trainedD.φ₁.toFloat})"
  IO.println s!"  loss ≈ {(Ld initialD).toFloat} -> {(Ld trainedD).toFloat}"
  IO.println s!"  prediction at x=1.5 ≈ {((f x trainedD)[0]).toFloat}"

  let initialG : Generative.LinearRegression.Parameters ℚ := ⟨0, 0⟩
  let trainedG := Generative.LinearRegression.trainLinearRegression
    data stepSize initialG steps
  let Lg := Generative.LinearRegression.leastSquaresLoss data
  IO.println s!"Generative x = phi0 + phi1*y ({steps} updates):"
  IO.println s!"  phi ≈ ({trainedG.φ₀.toFloat}, {trainedG.φ₁.toFloat})"
  IO.println s!"  loss ≈ {(Lg initialG).toFloat} -> {(Lg trainedG).toFloat}"
  if hSlope : trainedG.φ₁ ≠ 0 then
    let gInverse := Generative.LinearRegression.inverseModel (Scalar := ℚ)
    IO.println s!"  prediction at x=1.5 ≈ {((gInverse x trainedG hSlope)[0]).toFloat}"
  else
    IO.println "  inverse inference is unavailable: the fitted slope is zero"

end SupervisedLearning.Examples
