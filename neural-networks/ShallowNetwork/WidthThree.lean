module

public import ShallowNetwork.Model

/-!
# A shallow network with three hidden units

Source: Understanding Deep Learning, Section 3.1.1, equations (3.3)–(3.4).

The model is defined in `ShallowNetwork.Model`. This module fixes
the hidden width to three and retains the book's ten named parameters.
Both interfaces below call `shallowNetwork` with `D = 3`.
-/

@[expose] public section

namespace NeuralNetworks.WidthThree

/-- The ten parameters, collected into the book's single parameter object `Φ`.
The φ fields combine the hidden activations; each θ pair defines a hidden unit. -/
structure Parameters (Scalar : Type) where
  φ₀ : Scalar
  φ₁ : Scalar
  φ₂ : Scalar
  φ₃ : Scalar
  θ₁₀ : Scalar
  θ₁₁ : Scalar
  θ₂₀ : Scalar
  θ₂₁ : Scalar
  θ₃₀ : Scalar
  θ₃₁ : Scalar

/-- The book's ten named parameters, indexed by `Fin 3` for the model. -/
def Parameters.toShallowNetworkParameters {Scalar : Type}
    (Φ : Parameters Scalar) : ShallowNetworkParameters Scalar 3 :=
  { outputBias := Φ.φ₀
    outputWeights := ![Φ.φ₁, Φ.φ₂, Φ.φ₃]
    hiddenBiases := ![Φ.θ₁₀, Φ.θ₂₀, Φ.θ₃₀]
    hiddenWeights := ![Φ.θ₁₁, Φ.θ₂₁, Φ.θ₃₁] }

/-- The width-three case, written as `fₐ(x; Φ)` with the book's named parameters. -/
def networkWithParameters {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ : Parameters Scalar) : Scalar :=
  shallowNetwork activation x Φ.toShallowNetworkParameters

/-- The same width-three network with the ten parameters supplied individually. -/
def network {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (φ₀ φ₁ φ₂ φ₃ : Scalar)
    (θ₁₀ θ₁₁ θ₂₀ θ₂₁ θ₃₀ θ₃₁ : Scalar)
    (x : Scalar) : Scalar :=
  networkWithParameters activation x
    ⟨φ₀, φ₁, φ₂, φ₃, θ₁₀, θ₁₁, θ₂₀, θ₂₁, θ₃₀, θ₃₁⟩

end NeuralNetworks.WidthThree
