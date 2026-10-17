module

public meta import ShallowNetwork
public meta import Mathlib.Algebra.Field.Rat

/-!
# A numerical example with packaged parameters

Run `lake env lean neural-networks/ShallowNetworkExample.lean`.

Choose simple parameters for a triangular piecewise linear function:
`h₁ = ReLU(x)`, `h₂ = ReLU(x - 1)`, `h₃ = ReLU(x - 2)`,
and `y = h₁ - 2*h₂ + h₃`.

These parameters are chosen for illustration. Rational arithmetic keeps the
results exact. At `x = 3/2`, the hidden activations are `3/2`, `1/2`, and `0`,
so the output is `3/2 - 2*(1/2) + 0 = 1/2`.
-/

public meta section

namespace NeuralNetworks.Examples

/-- One object `Φ` containing all ten parameters of this example. -/
def exampleParameters : ShallowNetworkParameters ℚ :=
  { φ₀ := 0
    φ₁ := 1
    φ₂ := -2
    φ₃ := 1
    θ₁₀ := 0
    θ₁₁ := 1
    θ₂₀ := -1
    θ₂₁ := 1
    θ₃₀ := -2
    θ₃₁ := 1 }

#eval do
  let Φ := exampleParameters
  let f := shallowNetworkWithParameters (Scalar := ℚ) relu
  let inputs : List ℚ := [-1, 0, 1 / 2, 1, 3 / 2, 2, 3]
  IO.println "h1 = ReLU(x), h2 = ReLU(x - 1), h3 = ReLU(x - 2)"
  IO.println "y = h1 - 2*h2 + h3 (exact rational arithmetic)"
  for x in inputs do
    let h₁ := hiddenUnit relu Φ.θ₁₀ Φ.θ₁₁ x
    let h₂ := hiddenUnit relu Φ.θ₂₀ Φ.θ₂₁ x
    let h₃ := hiddenUnit relu Φ.θ₃₀ Φ.θ₃₁ x
    IO.println s!"x = {x}: h = ({h₁}, {h₂}, {h₃}), y = {f x Φ}"

end NeuralNetworks.Examples
