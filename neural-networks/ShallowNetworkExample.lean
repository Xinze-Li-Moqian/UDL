module

public meta import ShallowNetwork
public meta import Mathlib.Algebra.Field.Rat

/-!
# A network example with and without biases

Run `lake env lean neural-networks/ShallowNetworkExample.lean`.

Choose simple parameters for a triangular piecewise linear function:
`h₁ = ReLU(x)`, `h₂ = ReLU(x - 1)`, `h₃ = ReLU(x - 2)`,
and `y = h₁ - 2*h₂ + h₃`.

These parameters are chosen for illustration. Rational arithmetic keeps the
results exact. At `x = 3/2`, the hidden activations are `3/2`, `1/2`, and `0`,
so the output is `3/2 - 2*(1/2) + 0 = 1/2`.

The biases `θ₁₀ = 0`, `θ₂₀ = -1`, and `θ₃₀ = -2` make the hidden units
start rising at `x = 0`, `x = 1`, and `x = 2`, respectively. For example,
`relu x (-1)` calls `addBias (-1) x` before rectifying. Omitting the second
argument, as in `relu x`, uses a bias of zero. The named form
`relu x (bias := -1)` has the same meaning as the positional form.
The diagram's orange constant node `1` represents this offset as an edge:
`(-1) * 1 = -1`.

The two evaluations below use the same weights and inputs. The first sets
all biases to zero; the second uses the original biases above.
Without biases, all three hidden units become `ReLU(x)`, and these output
weights cancel: `y = (1 - 2 + 1) * ReLU(x) = 0`. This cancellation depends on
the chosen weights; networks without biases do not always output zero.
The output bias `φ₀` is already zero, so the difference here comes from
the hidden biases.
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

/-- Shared inputs for both evaluations. -/
def exampleInputs : List ℚ := [-1, 0, 1 / 2, 1, 3 / 2, 2, 3]

/-!
## Without bias

Keep the weights and set all biases to zero. Omit the second argument to `relu`
to use its default zero bias.
-/

#eval do
  let Φ := { exampleParameters with φ₀ := 0, θ₁₀ := 0, θ₂₀ := 0, θ₃₀ := 0 }
  let f := shallowNetworkWithParameters (Scalar := ℚ) relu
  IO.println "Without bias: h1 = h2 = h3 = ReLU(x)"
  IO.println "y = h1 - 2*h2 + h3 (exact rational arithmetic)"
  for x in exampleInputs do
    let h₁ := relu (Φ.θ₁₁ * x)
    let h₂ := relu (Φ.θ₂₁ * x)
    let h₃ := relu (Φ.θ₃₁ * x)
    IO.println s!"x = {x}: h = ({h₁}, {h₂}, {h₃}), y = {f x Φ}"

/-!
## With bias

Use the original parameter object. Supply each hidden bias as the second
positional argument to `relu`.
-/

#eval do
  let Φ := exampleParameters
  let f := shallowNetworkWithParameters (Scalar := ℚ) relu
  IO.println "With bias: h1 = ReLU(x), h2 = ReLU(x - 1), h3 = ReLU(x - 2)"
  IO.println "y = h1 - 2*h2 + h3 (exact rational arithmetic)"
  for x in exampleInputs do
    let h₁ := relu (Φ.θ₁₁ * x) Φ.θ₁₀
    let h₂ := relu (Φ.θ₂₁ * x) Φ.θ₂₀
    let h₃ := relu (Φ.θ₃₁ * x) Φ.θ₃₀
    IO.println s!"x = {x}: h = ({h₁}, {h₂}, {h₃}), y = {f x Φ}"

end NeuralNetworks.Examples
