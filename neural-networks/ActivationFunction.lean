module

public import Bias

/-!
# Activation functions

Source: Understanding Deep Learning, Section 3.1, equation (3.2).

An activation function `a` takes a scalar preactivation `z` and returns a scalar
activation `a[z]`. ReLU is one choice: negative inputs become zero and
nonnegative inputs are preserved. The `relu` interface accepts an optional
bias: `relu z` uses zero, while `relu z bias` adds that bias before rectifying.
Passing `relu` as an `ActivationFunction` uses the default zero bias.
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

end NeuralNetworks
