module

/-!
# Activation functions

Source: Understanding Deep Learning, Section 3.1, equations (3.2)–(3.3).

An activation function `a` takes a scalar preactivation `z` and returns a scalar
activation `a[z]`. ReLU is one choice: negative inputs become zero and
nonnegative inputs are preserved.
A hidden unit applies the chosen activation to an affine input: `a[bias + weight * x]`.
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

/-- A hidden unit: `h(x) = a[bias + weight * x]`. -/
def hiddenUnit {Scalar : Type} [Add Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (bias weight x : Scalar) : Scalar :=
  activation (bias + weight * x)

end NeuralNetworks
