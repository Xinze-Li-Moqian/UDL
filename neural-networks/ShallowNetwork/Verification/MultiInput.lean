module

public import ShallowNetwork.MultiInput

/-!
# Verification: multi-input model

With no hidden units, the multi-input network returns its output bias for any
number of inputs. With one input it computes the same output as the one-input
model of `ShallowNetwork.Model`.
-/

@[expose] public section

namespace NeuralNetworks

namespace MultiInput

/-- With no hidden units, only the output bias remains, for any number of inputs. -/
theorem network_zero_width {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    {Dᵢ : ℕ} (activation : ActivationFunction Scalar) (x : Fin Dᵢ → Scalar)
    (Φ : Parameters Scalar Dᵢ 0) :
    network activation x Φ = Φ.outputBias := by
  simp [network, addBias]

end MultiInput

/-- The one-input model is the multi-input model with `Dᵢ = 1`:
feeding `x` as the only input component gives the same output. -/
theorem shallowNetwork_eq_multiInput {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    {D : ℕ} (activation : ActivationFunction Scalar) (x : Scalar)
    (Φ : ShallowNetworkParameters Scalar D) :
    shallowNetwork activation x Φ =
      MultiInput.network activation (fun _ => x) Φ.toMultiInput := by
  simp [shallowNetwork_eq, MultiInput.network, hiddenUnit, MultiInput.hiddenUnit,
    ShallowNetworkParameters.toMultiInput]

end NeuralNetworks
