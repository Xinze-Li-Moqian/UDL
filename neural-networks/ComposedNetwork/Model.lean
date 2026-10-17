module

public import ShallowNetwork.Model

/-!
# Composed shallow networks

Source: Understanding Deep Learning, Section 4.1.

Two shallow networks can be composed so that the output `y` of the first
becomes the input of the second, which returns `y'`. More generally, a list of
shallow networks can be applied in order, each output feeding the next network.

Each list entry pairs a hidden width `D` with parameters of that width, so the
composed networks may have different widths. The list length is the number of
composed networks; the empty list returns its input unchanged. The book's case
of two width-three networks is in `ComposedNetwork.WidthThree`.
-/

@[expose] public section

namespace NeuralNetworks

/-- Compose shallow networks in order: each output is the next network's input. -/
def composedNetwork {Scalar : Type} [AddCommMonoid Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar) :
    List (Σ D, ShallowNetworkParameters Scalar D) → Scalar → Scalar
  | [], x => x
  | ⟨_, Φ⟩ :: rest, x => composedNetwork activation rest (shallowNetwork activation x Φ)

end NeuralNetworks
