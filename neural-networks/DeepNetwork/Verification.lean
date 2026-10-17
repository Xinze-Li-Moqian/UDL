module

public import DeepNetwork.Model
public import ShallowNetwork.MultiInput

/-!
# Deep network verification

A deep network with one hidden layer is a shallow network:
`MultiInput.Parameters.toDeep` places a shallow network's hidden and output
layers in the deep format with widths `[D]`, and `deepNetwork_toDeep` shows that
the output agrees with `MultiInput.network`.
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

/-- With one hidden layer, the deep network computes the shallow network's output. -/
theorem deepNetwork_toDeep {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {Dᵢ D : ℕ} (activation : ActivationFunction Scalar) (x : Fin Dᵢ → Scalar)
    (Φ : MultiInput.Parameters Scalar Dᵢ D) :
    deepNetwork activation Φ.toDeep x 0 = MultiInput.network activation x Φ := by
  simp [MultiInput.Parameters.toDeep, deepNetwork, LayerParameters.apply,
    MultiInput.network, MultiInput.hiddenUnit, addBias, mulVec, dotProduct]

end NeuralNetworks
