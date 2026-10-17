module

public import ShallowNetwork.WidthThree

/-!
# Shallow network verification

The zero-width network returns its output bias. At width three, the network
sum gives the book's explicit formula from Section 3.1.1, equation (3.4).
The universal approximation theorem is not yet formalized.
-/

@[expose] public section

namespace NeuralNetworks

/-- With no hidden units, only the output bias remains. -/
theorem shallowNetwork_zero_width {Scalar : Type} [AddCommMonoid Scalar]
    [Mul Scalar] (activation : ActivationFunction Scalar) (x : Scalar)
    (Φ : ShallowNetworkParameters Scalar 0) :
    shallowNetwork activation x Φ = Φ.outputBias := by
  simp [shallowNetwork, addBias]

namespace WidthThree

/-- The book's width-three interface calls the model at `D = 3`. -/
theorem networkWithParameters_eq {Scalar : Type} [AddCommMonoid Scalar]
    [Mul Scalar] (activation : ActivationFunction Scalar) (x : Scalar)
    (Φ : Parameters Scalar) :
    networkWithParameters activation x Φ =
      shallowNetwork activation x Φ.toShallowNetworkParameters :=
  rfl

/-- Expanding the sum at width three gives the book's equation (3.4). -/
theorem network_eq {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (φ₀ φ₁ φ₂ φ₃ : Scalar)
    (θ₁₀ θ₁₁ θ₂₀ θ₂₁ θ₃₀ θ₃₁ : Scalar) (x : Scalar) :
    network activation φ₀ φ₁ φ₂ φ₃ θ₁₀ θ₁₁ θ₂₀ θ₂₁ θ₃₀ θ₃₁ x =
      φ₀ + φ₁ * hiddenUnit activation θ₁₀ θ₁₁ x +
        φ₂ * hiddenUnit activation θ₂₀ θ₂₁ x +
        φ₃ * hiddenUnit activation θ₃₀ θ₃₁ x := by
  simp [network, networkWithParameters, shallowNetwork,
    Parameters.toShallowNetworkParameters, Fin.sum_univ_three, addBias, add_assoc]

end WidthThree

end NeuralNetworks
