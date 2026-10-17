module

public import ShallowNetwork.Model

/-!
# Verification: one-input model

With no hidden units, the network returns its output bias.
-/

@[expose] public section

namespace NeuralNetworks

/-- With no hidden units, only the output bias remains. -/
theorem shallowNetwork_zero_width {Scalar : Type} [AddCommMonoid Scalar]
    [Mul Scalar] (activation : ActivationFunction Scalar) (x : Scalar)
    (Φ : ShallowNetworkParameters Scalar 0) :
    shallowNetwork activation x Φ = Φ.outputBias := by
  simp [shallowNetwork, addBias]

end NeuralNetworks
