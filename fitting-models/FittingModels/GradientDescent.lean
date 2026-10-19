module

public import FittingModels.Gradient
public import GradientDescent

/-!
# Gradient descent on a loss

Source: Understanding Deep Learning, Section 6.1, equation (6.3).

    φₖ₊₁ = φₖ − α ∇L(φₖ),    α > 0.

The definitions connect the mathematical gradient to the existing update loop
in `Optimization`. They allow any real step size; gradient descent uses α > 0.
No automatic differentiation, floating-point execution, convergence, or
loss-decrease guarantee for an arbitrary α is asserted.
-/

@[expose] public section

namespace FittingModels

open LossFunctions
open scoped Gradient

variable {Parameters : Type*} [NormedAddCommGroup Parameters]
variable [InnerProductSpace ℝ Parameters] [CompleteSpace Parameters]

/-- Equation (6.3), using the mathematical gradient of the supplied loss. -/
noncomputable def gradientDescentStep (L : LossFunction ℝ Parameters)
    (α : ℝ) (φ : Parameters) : Parameters :=
  Optimization.gradientDescentStep (gradient L) α φ

/-- Repeat the same update from initial parameters φ₀. -/
noncomputable def gradientDescent (L : LossFunction ℝ Parameters)
    (α : ℝ) (initial : Parameters) : ℕ → Parameters :=
  Optimization.gradientDescent (gradient L) α initial

/-- A verified gradient formula g can be substituted into the update rule.
This connects a supplied gradient implementation to the mathematical definition. -/
theorem gradientDescentStep_eq_of_hasGradientAt
    (L : LossFunction ℝ Parameters) (α : ℝ) (φ g : Parameters)
    (hg : HasGradientAt L g φ) :
    gradientDescentStep L α φ = φ - α • g := by
  simp only [gradientDescentStep, Optimization.gradientDescentStep, hg.gradient]

/-- A verified gradient function gives exactly the same sequence of real iterates. -/
theorem gradientDescent_eq_of_hasGradientAt
    (L : LossFunction ℝ Parameters) (g : Parameters → Parameters)
    (hg : ∀ φ, HasGradientAt L (g φ) φ) (α : ℝ) (initial : Parameters) :
    gradientDescent L α initial = Optimization.gradientDescent g α initial := by
  unfold gradientDescent
  rw [gradient_eq hg]

/-- Zero gradient makes the current parameters a fixed point of the update. -/
theorem gradientDescentStep_eq_self_of_gradient_eq_zero
    (L : LossFunction ℝ Parameters) (α : ℝ) (φ : Parameters)
    (hzero : ∇ L φ = 0) : gradientDescentStep L α φ = φ := by
  simp [gradientDescentStep, Optimization.gradientDescentStep, hzero]

/-- At a differentiable local minimum, an update leaves the parameters unchanged. -/
theorem gradientDescentStep_eq_self_of_isLocalMin
    (L : LossFunction ℝ Parameters) (α : ℝ) (φ : Parameters)
    (hmin : IsLocalMin L φ) (hL : DifferentiableAt ℝ L φ) :
    gradientDescentStep L α φ = φ :=
  gradientDescentStep_eq_self_of_gradient_eq_zero L α φ
    (gradient_eq_zero_of_isLocalMin hmin hL)

end FittingModels
