module

public import Bias
public import Mathlib.Algebra.Order.Monoid.Unbundled.Basic

/-!
# Activation functions

Source: Understanding Deep Learning, Section 3.1, equation (3.2).

An activation function `a` takes a scalar preactivation `z` and returns a scalar
activation `a[z]`. ReLU is one choice: negative inputs become zero and
nonnegative inputs are preserved. The `relu` interface accepts an optional
bias: `relu z` uses zero, while `relu z bias` adds that bias before rectifying.
Passing `relu` as an `ActivationFunction` uses the default zero bias.
Import `ActivationFunction.Sigmoid` for the real logistic sigmoid and its
probability-range and score-threshold properties.
Import `ActivationFunction.Softmax` for a vector output transformation that
normalizes real scores into categorical probabilities.
-/

@[expose] public section

namespace NeuralNetworks

/-- The type of a scalar activation function `a[z]`. -/
abbrev ActivationFunction (Scalar : Type) := Scalar → Scalar

/-- Rectify a value after adding an optional bias, which defaults to zero. -/
def relu {Scalar : Type} [Zero Scalar] [Add Scalar] [LE Scalar] [DecidableLE Scalar]
    (z : Scalar) (bias : Scalar := 0) : Scalar :=
  let shifted := addBias bias z
  if 0 ≤ shifted then
    shifted
  else
    0

-- Omit the bias argument: its default value is zero.
example : relu (3 : Int) = 3 := rfl

-- Pass the bias as the second positional argument.
example : relu (3 : Int) (-1) = 2 := rfl

-- Pass the same bias by name.
example : relu (3 : Int) (bias := -1) = 2 := rfl

/-- ReLU with its default zero bias is `max z 0`. -/
theorem relu_eq_max {Scalar : Type} [LinearOrder Scalar] [AddZeroClass Scalar]
    (z : Scalar) : relu z = max z 0 := by
  show (if 0 ≤ 0 + z then 0 + z else 0) = max z 0
  rw [zero_add]
  split_ifs with h
  · exact (max_eq_left h).symm
  · exact (max_eq_right (not_le.mp h).le).symm

end NeuralNetworks
