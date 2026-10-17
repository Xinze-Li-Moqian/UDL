module

public import Network
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Shallow neural networks

Source: Understanding Deep Learning, Section 3.2, equations (3.5)–(3.6).

The network still has one input, one output, and one hidden layer. Its width `D`
is the number of hidden units. Each unit computes `h_d = a[θ_d0 + θ_d1 * x]`,
and the output is `φ₀ + ∑ d, φ_d * h_d`.

The parameters keep the book's Chapter 3 names. The network itself is computed
by the matrix-notation foundation in `Network`: `toNetworkParameters` arranges
the parameters as in equation (4.14), `h = a[θ₀ + θ x]` and `y = φ₀ + φ h`, for
the architecture `Architecture.ofList 1 [D] 1`. `shallowNetwork_eq` recovers the
scalar formula of equations (3.5)–(3.6), which proofs about the model use.

`Fin D` indexes the units from `0` to `D - 1`; the book numbers them from `1`
to `D`. There are `3 * D + 1` scalar parameters: three per hidden unit and one
output bias. This module defines the network family used in the universal
approximation theorem, which is stated and proved in
`ShallowNetwork.UniversalApproximation`. Proved properties of the model are
collected in `ShallowNetwork.Verification`.
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

/-- Equation (4.14): the shallow network in matrix notation, `h = a[θ₀ + θ x]` and
`y = φ₀ + φ h`. The hidden layer has bias vector `θ_d0` and the `D × 1` weight matrix
`θ_d1`; the output layer has bias `φ₀` and the `1 × D` weight matrix `φ_d`. -/
def ShallowNetworkParameters.toNetworkParameters {Scalar : Type} {D : ℕ}
    (Φ : ShallowNetworkParameters Scalar D) :
    NetworkParameters Scalar (Architecture.ofList 1 [D] 1) :=
  .cons Φ.hiddenBiases (Matrix.of fun d _ => Φ.hiddenWeights d) <|
  .single (fun _ => Φ.outputBias) (Matrix.of fun _ d => Φ.outputWeights d)

/-- Network capacity measured by hidden width, as in Section 3.2.
This is the capacity of the architecture `Architecture.ofList 1 [D] 1`, which
counts all `D` hidden units regardless of their parameter values or outputs.
It measures the network architecture, not a norm of its parameters or output
function; different widths can represent the same function. -/
def networkCapacity {Scalar : Type} {D : ℕ}
    (_ : ShallowNetworkParameters Scalar D) : ℕ :=
  (Architecture.ofList 1 [D] 1).capacity

/-- The shallow network, evaluated through the matrix-notation foundation `network`. -/
def shallowNetwork {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {D : ℕ} (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ : ShallowNetworkParameters Scalar D) : Scalar :=
  network activation Φ.toNetworkParameters (fun _ : Fin 1 => x) (0 : Fin 1)

/-- Equations (3.5)–(3.6): the output is `φ₀ + ∑ d, φ_d * h_d` with hidden units
`h_d = a[θ_d0 + θ_d1 * x]`. -/
theorem shallowNetwork_eq {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {D : ℕ} (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ : ShallowNetworkParameters Scalar D) :
    shallowNetwork activation x Φ =
      addBias Φ.outputBias
        (∑ d : Fin D, Φ.outputWeights d *
          hiddenUnit activation (Φ.hiddenBiases d) (Φ.hiddenWeights d) x) := by
  simp [shallowNetwork, ShallowNetworkParameters.toNetworkParameters, hiddenUnit, addBias,
    Matrix.mulVec, dotProduct]

end NeuralNetworks
