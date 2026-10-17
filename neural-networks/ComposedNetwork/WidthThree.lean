module

public import ShallowNetwork.WidthThree

/-!
# Composing two width-three networks

Source: Understanding Deep Learning, Section 4.1, equations (4.1)–(4.4).

The first network takes `x` and returns `y` (equations 4.1–4.2); the second
takes `y` and returns `y'` (equations 4.3–4.4). Both have three hidden units and
use the book's named parameters from `ShallowNetwork.WidthThree`: `Φ` for the
first network and `Φ'` for the second.
-/

@[expose] public section

namespace NeuralNetworks.WidthThree

/-- Equations (4.1)–(4.4): the output `y` of the first network is the input of the second. -/
def composedNetwork {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ Φ' : Parameters Scalar) : Scalar :=
  networkWithParameters activation (networkWithParameters activation x Φ) Φ'

end NeuralNetworks.WidthThree
