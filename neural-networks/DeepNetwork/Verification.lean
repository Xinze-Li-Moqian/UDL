module

public import Network
public import DeepNetwork.WidthThree
public import ShallowNetwork.MultiInput

/-!
# Deep network verification

A network with one hidden layer is a shallow network:
`MultiInput.Parameters.toDeep` places a shallow network's hidden and output
layers into `β₀, Ω₀` and `β₁, Ω₁` for the architecture `ofList Dᵢ [D] 1`, and
`network_toDeep` shows that the output agrees with `MultiInput.network`. That
architecture has depth one and capacity `D` (`toDeep_architecture`).

The book's two-layer, width-three network expands to the single expression of
equation (4.10) (`WidthThree.deepNetwork_eq`).
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

/-- A shallow network with `Dᵢ` inputs and width `D`, as a network with one hidden layer. -/
def MultiInput.Parameters.toDeep {Scalar : Type} {Dᵢ D : ℕ}
    (Φ : MultiInput.Parameters Scalar Dᵢ D) :
    NetworkParameters Scalar (Architecture.ofList Dᵢ [D] 1) :=
  .cons Φ.hiddenBiases (Matrix.of Φ.hiddenWeights) <|
  .single (fun _ => Φ.outputBias) (Matrix.of fun _ d => Φ.outputWeights d)

/-- The converted shallow network has depth one and capacity `D`. -/
theorem MultiInput.Parameters.toDeep_architecture (Dᵢ D : ℕ) :
    (Architecture.ofList Dᵢ [D] 1).depth = 1 ∧ (Architecture.ofList Dᵢ [D] 1).capacity = D := by
  simp [Architecture.ofList_depth_capacity]

/-- With one hidden layer, the network computes the shallow network's output. -/
theorem network_toDeep {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar) (x : Fin Dᵢ → Scalar)
    (Φ : MultiInput.Parameters Scalar Dᵢ D) :
    network activation Φ.toDeep x (0 : Fin 1) = MultiInput.network activation x Φ := by
  simp [MultiInput.Parameters.toDeep, MultiInput.network, MultiInput.hiddenUnit, addBias,
    mulVec, dotProduct]

end NeuralNetworks

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
