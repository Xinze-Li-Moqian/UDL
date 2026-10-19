module

public import FittingModels.GradientDescent
public import Mathlib.Analysis.Calculus.Deriv.Pow
public import Mathlib.Tactic.Ring
public meta import GradientDescent
public meta import Mathlib.Algebra.Field.Rat

/-!
# Example: fitting one parameter by gradient descent

A small example of the two steps in Section 6.1:

    L[φ] = (φ − 3)²,    g(φ) = 2(φ − 3),
    φₖ₊₁ = φₖ − (1/4) g(φₖ),    φ₀ = 0.

The real-valued proof verifies g as the gradient. The executable example
uses the same algebraic formulas over rational numbers for exact arithmetic;
there is no floating-point rounding or automatic differentiation here.
-/

@[expose] public section

namespace FittingModels.Examples.Quadratic

open LossFunctions
open scoped LossFunctions

variable {Scalar : Type*} [Ring Scalar]

/-- The target parameter value is 3; measure its squared error. -/
def loss : LossFunction Scalar Scalar := fun φ => (φ - 3) ^ 2

/-- Supply a computable formula for the gradient. Its real interpretation is proved below. -/
def gradientFormula (φ : Scalar) : Scalar := 2 * (φ - 3)

/-- Mathlib verifies that the supplied formula is the derivative of the loss. -/
theorem hasGradientAt_loss (φ : ℝ) :
    HasGradientAt (loss (Scalar := ℝ)) (gradientFormula φ) φ := by
  convert! (((hasDerivAt_id φ).sub_const (3 : ℝ)).pow 2).hasGradientAt' using 1
  simp [gradientFormula]

/-- The supplied formula produces exactly the mathematical real-gradient iterates. -/
theorem iterates_eq (α initial : ℝ) :
    FittingModels.gradientDescent loss α initial =
      Optimization.gradientDescent gradientFormula α initial :=
  gradientDescent_eq_of_hasGradientAt loss gradientFormula hasGradientAt_loss α initial

/-- With learning rate 1/4, every update halves the error relative to 3. -/
theorem step_error (φ : ℝ) :
    Optimization.gradientDescentStep gradientFormula (1 / 4 : ℝ) φ - 3 =
      (φ - 3) / 2 := by
  simp only [Optimization.gradientDescentStep, gradientFormula, smul_eq_mul]
  ring

/-- The same update divides the loss by four at each step. -/
theorem step_loss (φ : ℝ) :
    loss (Optimization.gradientDescentStep gradientFormula (1 / 4 : ℝ) φ) =
      loss φ / 4 := by
  unfold loss
  rw [step_error]
  ring

/-- Execute k updates using exact rational arithmetic, starting from zero. -/
def iterate (k : ℕ) : ℚ :=
  Optimization.gradientDescent gradientFormula (1 / 4 : ℚ) 0 k

-- Each row is (iteration, parameter, gradient, loss).
#eval (List.range 5).map fun k =>
  let φ := iterate k
  (k, φ, gradientFormula φ, loss φ)

end FittingModels.Examples.Quadratic
