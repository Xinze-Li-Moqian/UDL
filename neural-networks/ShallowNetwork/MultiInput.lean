module

public import ShallowNetwork.Model

/-!
# Shallow networks with multiple inputs

Source: Understanding Deep Learning, Sections 3.3–3.4.

The input is now a vector `x` with `Dᵢ` components, indexed by `Fin Dᵢ`.
Each hidden unit has one bias and one weight per input,
`h_d = a[θ_d0 + ∑ i, θ_di * x_i]`, so it is constant along hyperplanes
perpendicular to its weight vector. The output is still a single scalar,
`φ₀ + ∑ d, φ_d * h_d`. With `Dᵢ = 2` and ReLU, the network's graph is a
surface made of flat polygonal pieces.

As in the one-input case, the parameters keep the book's names and the network is
computed by the matrix-notation foundation: `Parameters.toNetworkParameters`
arranges them for the architecture `Architecture.ofList Dᵢ [D] 1`, and
`network_eq` recovers the scalar formula used in proofs.

There are `(Dᵢ + 2) * D + 1` scalar parameters. The one-input model of
`ShallowNetwork.Model` is the case `Dᵢ = 1`: `ShallowNetworkParameters.toMultiInput`
embeds it, and `ShallowNetwork.Verification` proves that both compute the same output.
-/

@[expose] public section

namespace NeuralNetworks.MultiInput

/-- A hidden unit with `Dᵢ` inputs: `h(x) = a[bias + ∑ i, weights i * x i]`. -/
def hiddenUnit {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar] {Dᵢ : ℕ}
    (activation : ActivationFunction Scalar)
    (bias : Scalar) (weights x : Fin Dᵢ → Scalar) : Scalar :=
  activation (addBias bias (∑ i : Fin Dᵢ, weights i * x i))

/-- Parameters for a network with `Dᵢ` inputs, `D` hidden units, and one output. -/
structure Parameters (Scalar : Type) (Dᵢ D : ℕ) where
  /-- The output offset `φ₀`. -/
  outputBias : Scalar
  /-- The output coefficients `φ_d`. -/
  outputWeights : Fin D → Scalar
  /-- The hidden offsets `θ_d0`. -/
  hiddenBiases : Fin D → Scalar
  /-- The input coefficients `θ_di`: one weight vector per hidden unit. -/
  hiddenWeights : Fin D → Fin Dᵢ → Scalar

/-- The parameters in matrix notation for the architecture `Architecture.ofList Dᵢ [D] 1`:
the hidden layer has bias vector `θ_d0` and the `D × Dᵢ` weight matrix `θ_di`; the
output layer has bias `φ₀` and the `1 × D` weight matrix `φ_d`. -/
def Parameters.toNetworkParameters {Scalar : Type} {Dᵢ D : ℕ} (Φ : Parameters Scalar Dᵢ D) :
    NetworkParameters Scalar (Architecture.ofList Dᵢ [D] 1) :=
  .cons Φ.hiddenBiases (Matrix.of Φ.hiddenWeights) <|
  .single (fun _ => Φ.outputBias) (Matrix.of fun _ d => Φ.outputWeights d)

/-- The shallow network with `Dᵢ` inputs, evaluated through the foundation `network`. -/
def network {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar)
    (x : Fin Dᵢ → Scalar) (Φ : Parameters Scalar Dᵢ D) : Scalar :=
  NeuralNetworks.network activation Φ.toNetworkParameters x (0 : Fin 1)

/-- The output is `φ₀ + ∑ d, φ_d * h_d` with hidden units `h_d = a[θ_d0 + ∑ i, θ_di * x_i]`. -/
theorem network_eq {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar)
    (x : Fin Dᵢ → Scalar) (Φ : Parameters Scalar Dᵢ D) :
    network activation x Φ =
      addBias Φ.outputBias
        (∑ d : Fin D, Φ.outputWeights d *
          hiddenUnit activation (Φ.hiddenBiases d) (Φ.hiddenWeights d) x) := by
  simp [network, Parameters.toNetworkParameters, hiddenUnit, addBias, Matrix.mulVec, dotProduct]

end NeuralNetworks.MultiInput

namespace NeuralNetworks

/-- View one-input parameters as parameters for a network with `Dᵢ = 1`. -/
def ShallowNetworkParameters.toMultiInput {Scalar : Type} {D : ℕ}
    (Φ : ShallowNetworkParameters Scalar D) : MultiInput.Parameters Scalar 1 D :=
  { outputBias := Φ.outputBias
    outputWeights := Φ.outputWeights
    hiddenBiases := Φ.hiddenBiases
    hiddenWeights := fun d _ => Φ.hiddenWeights d }

end NeuralNetworks
