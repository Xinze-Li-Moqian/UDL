module

public import Network
public import DeepNetwork.WidthThree

/-!
# Deep network verification

The book's two-layer, width-three network expands to the single expression of
equation (4.10) (`WidthThree.deepNetwork_eq`).
-/

@[expose] public section

namespace NeuralNetworks.WidthThree

open Matrix

/-- Equation (4.10): the two-layer, width-three network as one expression. -/
theorem deepNetwork_eq {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (a : ActivationFunction Scalar) (x : Scalar) (Φ : DeepParameters Scalar) :
    deepNetwork a x Φ =
      Φ.φ'₀
        + Φ.φ'₁ * a (Φ.ψ₁₀ + Φ.ψ₁₁ * a (Φ.θ₁₀ + Φ.θ₁₁ * x) + Φ.ψ₁₂ * a (Φ.θ₂₀ + Φ.θ₂₁ * x)
            + Φ.ψ₁₃ * a (Φ.θ₃₀ + Φ.θ₃₁ * x))
        + Φ.φ'₂ * a (Φ.ψ₂₀ + Φ.ψ₂₁ * a (Φ.θ₁₀ + Φ.θ₁₁ * x) + Φ.ψ₂₂ * a (Φ.θ₂₀ + Φ.θ₂₁ * x)
            + Φ.ψ₂₃ * a (Φ.θ₃₀ + Φ.θ₃₁ * x))
        + Φ.φ'₃ * a (Φ.ψ₃₀ + Φ.ψ₃₁ * a (Φ.θ₁₀ + Φ.θ₁₁ * x) + Φ.ψ₃₂ * a (Φ.θ₂₀ + Φ.θ₂₁ * x)
            + Φ.ψ₃₃ * a (Φ.θ₃₀ + Φ.θ₃₁ * x)) := by
  simp [deepNetwork, DeepParameters.toNetworkParameters, dotProduct, Fin.sum_univ_three,
    add_assoc]

end NeuralNetworks.WidthThree
