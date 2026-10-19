module

public import LossFunctions.EmpiricalLoss
public import Mathlib.Analysis.Calculus.Gradient.Basic
public import Mathlib.Analysis.Calculus.Deriv.Comp
public import Mathlib.Analysis.Calculus.Deriv.Add
public import Mathlib.Analysis.Calculus.Deriv.Mul
public import Mathlib.Analysis.Calculus.LocalExtr.Basic
public import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Gradients of a loss

Source: Understanding Deep Learning, Section 6.1, equation (6.2).

Use Mathlib's gradient and `HasGradientAt` on a real Hilbert space. With
`open scoped Gradient`, write `∇ L φ`. In Euclidean coordinates this is
the vector of partial derivatives:

    (∇ L φ)ⱼ = ∂L/∂φⱼ.

The gradient is mathematical, not an executable differentiation routine.
Mathlib assigns zero when the derivative does not exist; the gradient
interpretation below always requires differentiability or `HasGradientAt`.
-/

@[expose] public section

namespace FittingModels

open LossFunctions
open scoped Gradient

variable {Parameters : Type*} [NormedAddCommGroup Parameters]
variable [InnerProductSpace ℝ Parameters] [CompleteSpace Parameters]
variable {L : LossFunction ℝ Parameters} {φ g : Parameters}

/-- The loss derivative along a direction v is its inner product with the gradient.

    d/dt L(φ + t v) at t = 0 equals ⟨g, v⟩.
-/
theorem hasDerivAt_along (hg : HasGradientAt L g φ) (v : Parameters) :
    HasDerivAt (fun t : ℝ => L (φ + t • v)) (inner ℝ g v) 0 := by
  have hline : HasDerivAt (fun t : ℝ => φ + t • v) v 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const v).const_add φ
  convert! hg.hasFDerivAt.comp_hasDerivAt_of_eq 0 hline (by simp) using 1

/-- The negative gradient is a downhill direction whenever g is nonzero.

    d/dt L(φ − t g) at t = 0 equals −‖g‖².

This first-order identity does not guarantee descent for an arbitrary step size.
-/
theorem hasDerivAt_negative_gradient (hg : HasGradientAt L g φ) :
    HasDerivAt (fun t : ℝ => L (φ - t • g)) (-‖g‖ ^ 2) 0 := by
  simpa [smul_neg, sub_eq_add_neg, inner_neg_right, real_inner_self_eq_norm_sq]
    using hasDerivAt_along hg (-g)

/-- At a differentiable unconstrained local minimum, the gradient is zero. -/
theorem gradient_eq_zero_of_isLocalMin (hmin : IsLocalMin L φ)
    (hL : DifferentiableAt ℝ L φ) : ∇ L φ = 0 := by
  apply (InnerProductSpace.toDual ℝ Parameters).injective
  have h := hmin.hasFDerivAt_eq_zero hL.hasGradientAt.hasFDerivAt
  simpa using h

section Coordinates

variable {N : ℕ}

/-- Equation (6.2): varying just coordinate j gives gradient component j.
The path uses an increment t, so its derivative is evaluated at zero. -/
theorem hasDerivAt_coordinate {L : LossFunction ℝ (EuclideanSpace ℝ (Fin N))}
    {φ g : EuclideanSpace ℝ (Fin N)} (hg : HasGradientAt L g φ) (j : Fin N) :
    HasDerivAt
      (fun t : ℝ => L (φ + t • EuclideanSpace.basisFun (Fin N) ℝ j)) (g j) 0 := by
  simpa only [EuclideanSpace.inner_basisFun_real] using
    hasDerivAt_along hg (EuclideanSpace.basisFun (Fin N) ℝ j)

end Coordinates

end FittingModels
