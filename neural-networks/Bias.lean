module

/-!
# Bias

A bias is a constant offset added to a value. In a network diagram, a constant
node `1` can represent it through an edge labeled `bias`: `bias * 1 = bias`.
The computation below adds the offset directly.
-/

@[expose] public section

namespace NeuralNetworks

/-- Add a bias to a value: `bias + value`. -/
def addBias {Scalar : Type} [Add Scalar] (bias value : Scalar) : Scalar :=
  bias + value

end NeuralNetworks
