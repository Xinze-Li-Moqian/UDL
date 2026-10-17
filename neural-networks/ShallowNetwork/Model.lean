module

public import ActivationFunction
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Shallow neural networks

Source: Understanding Deep Learning, Section 3.2, equations (3.5)–(3.6).

The network still has one input, one output, and one hidden layer. Its width `D`
is the number of hidden units. Each unit computes `h_d = a[θ_d0 + θ_d1 * x]`,
and the output is `φ₀ + ∑ d, φ_d * h_d`.

`Fin D` indexes the units from `0` to `D - 1`; the book numbers them from `1`
to `D`. There are `3 * D + 1` scalar parameters: three per hidden unit and one
output bias. This module defines the network family used in the universal
approximation theorem; it does not prove that theorem. Properties of the model
are collected in `ShallowNetwork.Verification`.
-/

@[expose] public section

namespace NeuralNetworks

/-- A hidden unit from equations (3.3) and (3.5): `h(x) = a[bias + weight * x]`. -/
def hiddenUnit {Scalar : Type} [Add Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (bias weight x : Scalar) : Scalar :=
  activation (addBias bias (weight * x))

/-- Parameters for a single-input, single-output network with `D` hidden units. -/
structure ShallowNetworkParameters (Scalar : Type) (D : ℕ) where
  /-- The output offset `φ₀`. -/
  outputBias : Scalar
  /-- The output coefficients `φ_d`. -/
  outputWeights : Fin D → Scalar
  /-- The hidden offsets `θ_d0`. -/
  hiddenBiases : Fin D → Scalar
  /-- The input coefficients `θ_d1`. -/
  hiddenWeights : Fin D → Scalar

/-- Network capacity measured by hidden width, as in Section 3.2.
This counts all `D` hidden units, regardless of their parameter values or outputs.
It measures the network architecture, not a norm of its parameters or output
function; different widths can represent the same function. -/
def networkCapacity {Scalar : Type} {D : ℕ}
    (_ : ShallowNetworkParameters Scalar D) : ℕ :=
  D

/-- Equations (3.5)–(3.6), evaluated with a chosen activation function. -/
def shallowNetwork {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    {D : ℕ} (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ : ShallowNetworkParameters Scalar D) : Scalar :=
  let h := fun d : Fin D =>
    hiddenUnit activation (Φ.hiddenBiases d) (Φ.hiddenWeights d) x
  addBias Φ.outputBias (∑ d : Fin D, Φ.outputWeights d * h d)

end NeuralNetworks
