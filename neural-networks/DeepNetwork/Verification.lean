module

public import DeepNetwork.Model
public import DeepNetwork.WidthThree
public import ShallowNetwork.MultiInput

/-!
# Deep network verification

A deep network with one hidden layer is a shallow network:
`MultiInput.Parameters.toDeep` places a shallow network's hidden and output
layers in the deep format with widths `[D]`, and `deepNetwork_toDeep` shows that
the output agrees with `MultiInput.network`.

The capacity is the sum of the layer widths (`capacity_eq_sum_width`), and a
converted shallow network has the capacity `networkCapacity` assigns it,
its width `D` (`toDeep_capacity`).

The book's two-layer, width-three network expands to the single expression of
equation (4.10) (`WidthThree.deepNetwork_eq`).
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

/-- A shallow network with `Dᵢ` inputs and width `D`, as a deep network with
hidden widths `[D]` and one output. -/
def MultiInput.Parameters.toDeep {Scalar : Type} {Dᵢ D : ℕ}
    (Φ : MultiInput.Parameters Scalar Dᵢ D) : DeepNetworkParameters Scalar Dᵢ [D] 1 :=
  (⟨Φ.hiddenBiases, Matrix.of Φ.hiddenWeights⟩,
    ⟨fun _ => Φ.outputBias, Matrix.of fun _ d => Φ.outputWeights d⟩)

/-- The capacity is the sum of the layer widths `D₁ + ⋯ + D_K`. -/
theorem DeepNetworkParameters.capacity_eq_sum_width {Scalar : Type} {Dᵢ Dₒ : ℕ}
    {widths : List ℕ} (Φ : DeepNetworkParameters Scalar Dᵢ widths Dₒ) :
    Φ.capacity = ∑ k : Fin Φ.depth, Φ.width k :=
  (Fin.sum_univ_getElem widths).symm

/-- A shallow network of width `D` converts to a deep network of depth one and
capacity `D`, matching `networkCapacity` for shallow networks. -/
theorem MultiInput.Parameters.toDeep_capacity {Scalar : Type} {Dᵢ D : ℕ}
    (Φ : MultiInput.Parameters Scalar Dᵢ D) :
    Φ.toDeep.depth = 1 ∧ Φ.toDeep.capacity = D := by
  simp [DeepNetworkParameters.depth, DeepNetworkParameters.capacity]

/-- With one hidden layer, the deep network computes the shallow network's output. -/
theorem deepNetwork_toDeep {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar) (x : Fin Dᵢ → Scalar)
    (Φ : MultiInput.Parameters Scalar Dᵢ D) :
    deepNetwork activation Φ.toDeep x 0 = MultiInput.network activation x Φ := by
  simp [MultiInput.Parameters.toDeep, deepNetwork, LayerParameters.apply,
    MultiInput.network, MultiInput.hiddenUnit, addBias, mulVec, dotProduct]

end NeuralNetworks

namespace NeuralNetworks.WidthThree

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
  simp [deepNetwork, NeuralNetworks.deepNetwork, DeepParameters.toDeepNetworkParameters,
    LayerParameters.apply, addBias, dotProduct, Fin.sum_univ_three, add_assoc]

end NeuralNetworks.WidthThree
