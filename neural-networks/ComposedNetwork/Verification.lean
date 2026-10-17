module

public import ComposedNetwork.Model
public import ComposedNetwork.WidthThree

/-!
# Composed network verification

The book's composition of two width-three networks is the general composition
applied to a list of two networks of width three.
-/

@[expose] public section

namespace NeuralNetworks.WidthThree

/-- Equations (4.1)–(4.4) are `composedNetwork` on the list `[⟨3, Φ⟩, ⟨3, Φ'⟩]`. -/
theorem composedNetwork_eq {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar) (x : Scalar) (Φ Φ' : Parameters Scalar) :
    composedNetwork activation x Φ Φ' =
      NeuralNetworks.composedNetwork activation
        [⟨3, Φ.toShallowNetworkParameters⟩, ⟨3, Φ'.toShallowNetworkParameters⟩] x :=
  rfl

end NeuralNetworks.WidthThree
