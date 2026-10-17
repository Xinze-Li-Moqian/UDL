module

public import ShallowNetwork.WidthThree

/-!
# Shallow network verification

The zero-width network returns its output bias. At width three, the output
is `φ₀ + φ₁ * h₁ + φ₂ * h₂ + φ₃ * h₃`.
The universal approximation theorem is proved in
`ShallowNetwork.UniversalApproximation`.
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

/-- At width three, the output is `φ₀ + φ₁ * h₁ + φ₂ * h₂ + φ₃ * h₃`,
where each `hᵢ` is a hidden activation. -/
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
