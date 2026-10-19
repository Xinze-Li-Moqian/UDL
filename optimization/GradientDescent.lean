module

/-!
# Gradient descent

The caller supplies a gradient function, a step size, and initial parameters.
The update rule is shared across models, including both directions of linear
regression. Model-specific training files supply the appropriate loss gradient.
These definitions implement the update rule. They impose no step-size
restriction; properties such as decreasing loss require assumptions and proofs.
-/

@[expose] public section

namespace Optimization

universe u v

variable {Scalar : Type u} {Parameters : Type v}
variable [Sub Parameters] [SMul Scalar Parameters]

/-- One update: `φ_next = φ - stepSize • gradient φ`. -/
def gradientDescentStep (gradient : Parameters → Parameters)
    (stepSize : Scalar) (φ : Parameters) : Parameters :=
  φ - stepSize • gradient φ

/-- The parameters after `k` updates, starting from `initial` at step zero. -/
def gradientDescent (gradient : Parameters → Parameters)
    (stepSize : Scalar) (initial : Parameters) : Nat → Parameters
  | 0 => initial
  | k + 1 => gradientDescentStep gradient stepSize
      (gradientDescent gradient stepSize initial k)

end Optimization
