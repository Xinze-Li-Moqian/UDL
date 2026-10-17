module

public import ShallowNetwork.Model

/-!
# Verification: one-input model

With no hidden units, the network returns its output bias. The capacity of a
width-`D` network, read off its architecture `Architecture.ofList 1 [D] 1`, is `D`.
-/

@[expose] public section

namespace NeuralNetworks

/-- With no hidden units, only the output bias remains. -/
theorem shallowNetwork_zero_width {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar) (x : Scalar)
    (Φ : ShallowNetworkParameters Scalar 0) :
    shallowNetwork activation x Φ = Φ.outputBias := by
  simp [shallowNetwork_eq, addBias]

/-- A shallow network of width `D` has capacity `D`. -/
theorem networkCapacity_eq {Scalar : Type} {D : ℕ} (Φ : ShallowNetworkParameters Scalar D) :
    networkCapacity Φ = D :=
  (Architecture.ofList_depth_capacity 1 [D] 1).2.trans (by simp)

end NeuralNetworks
