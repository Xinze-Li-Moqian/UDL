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

/-- The shallow network with `Dᵢ` inputs, evaluated with a chosen activation function. -/
def network {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar)
    (x : Fin Dᵢ → Scalar) (Φ : Parameters Scalar Dᵢ D) : Scalar :=
  let h := fun d : Fin D =>
    hiddenUnit activation (Φ.hiddenBiases d) (Φ.hiddenWeights d) x
  addBias Φ.outputBias (∑ d : Fin D, Φ.outputWeights d * h d)

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
