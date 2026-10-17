module

public import ShallowNetwork.Model

/-!
# Composed shallow networks

Source: Understanding Deep Learning, Section 4.1.

Two shallow networks can be composed so that the output `y` of the first
becomes the input of the second, which returns `y'`. More generally, a list of
shallow networks can be applied in order, each output feeding the next network.

Each list entry pairs a hidden width `D` with parameters of that width, so the
composed networks may have different widths. Each one is a `shallowNetwork`,
evaluated on the matrix-notation foundation in `Network`. The list length is the
number of composed networks; the empty list returns its input unchanged. The
book's case of two width-three networks is in `ComposedNetwork.WidthThree`.
`composedFamily activation widths` collects the functions obtained by composing
networks of the given widths.
-/

@[expose] public section

namespace NeuralNetworks

/-- Compose shallow networks in order: each output is the next network's input. -/
def composedNetwork {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar) :
    List (Σ D, ShallowNetworkParameters Scalar D) → Scalar → Scalar
  | [], x => x
  | ⟨_, Φ⟩ :: rest, x => composedNetwork activation rest (shallowNetwork activation x Φ)

/-- The family of functions computed by composing shallow networks of widths
`[D₁, …, D_K]`, one network per width. -/
def composedFamily {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar) (widths : List ℕ) : Set (Scalar → Scalar) :=
  { f | ∃ Ns : List (Σ D, ShallowNetworkParameters Scalar D),
      Ns.map Sigma.fst = widths ∧ f = composedNetwork activation Ns }

end NeuralNetworks
