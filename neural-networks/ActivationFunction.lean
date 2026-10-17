module

/-!
# Activation functions

Source: Understanding Deep Learning, Section 3.1, equation (3.2).

An activation function `a` takes a scalar preactivation `z` and returns a scalar
activation `a[z]`. ReLU is one choice: negative inputs become zero and
nonnegative inputs are preserved.
-/

@[expose] public section

namespace NeuralNetworks

/-- The type of a scalar activation function `a[z]`. -/
abbrev ActivationFunction (Scalar : Type) := Scalar → Scalar

/-- Rectified linear unit: `ReLU(z) = z` if `0 ≤ z`, and `0` otherwise. -/
def relu {Scalar : Type} [Zero Scalar] [LE Scalar] [DecidableLE Scalar] :
    ActivationFunction Scalar :=
  fun z =>
    if 0 ≤ z then
      z
    else
      0

end NeuralNetworks
