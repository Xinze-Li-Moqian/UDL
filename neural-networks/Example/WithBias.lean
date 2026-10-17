module

public meta import Example.Plot

/-!
# A shallow network with biases

Run `lake env lean neural-networks/Example/WithBias.lean` for the numerical
example, or put the cursor on `#html plot` to view the charts in Lean Infoview.

The three hidden biases place the ReLU corners at zero, one, and two:
`h₁ = ReLU(x)`, `h₂ = ReLU(x - 1)`, and `h₃ = ReLU(x - 2)`.
The output `y = h₁ - 2*h₂ + h₃` forms a triangle: it rises from zero to one
on `[0, 1]`, falls back to zero on `[1, 2]`, and is zero outside `[0, 2]`.

At `x = 3/2`, the hidden activations are `3/2`, `1/2`, and `0`.
Their weighted contributions are `3/2`, `-1`, and `0`, giving `y = 1/2`.
Each call `relu z bias` below passes the hidden bias explicitly as the second
argument. The numerical example uses exact rational arithmetic.
-/

public meta section

namespace NeuralNetworks.Examples.WithBias

/-- The hidden biases shift the three corners; the output bias is zero. -/
def parameters : ShallowNetworkParameters ℚ 3 :=
  { outputBias := 0
    outputWeights := ![1, -2, 1]
    hiddenBiases := ![0, -1, -2]
    hiddenWeights := ![1, 1, 1] }

/-- The same inputs are used by the example without biases. -/
def inputs : List ℚ := [-1, 0, 1 / 2, 1, 3 / 2, 2, 3]

#eval do
  let Φ := parameters
  IO.println "With bias: h1 = ReLU(x), h2 = ReLU(x - 1), h3 = ReLU(x - 2)"
  IO.println s!"Network capacity (hidden units): {networkCapacity Φ}"
  IO.println "y = h1 - 2*h2 + h3 (exact rational arithmetic)"
  for x in inputs do
    let h₁ := relu (Φ.hiddenWeights 0 * x) (Φ.hiddenBiases 0)
    let h₂ := relu (Φ.hiddenWeights 1 * x) (Φ.hiddenBiases 1)
    let h₃ := relu (Φ.hiddenWeights 2 * x) (Φ.hiddenBiases 2)
    IO.println s!"x = {x}: h = ({h₁}, {h₂}, {h₃}), y = {shallowNetwork relu x Φ}"

/-- Samples for the affine inputs, activations, contributions, and output. -/
def plotData : Array Lean.Json := Plotting.networkPlotData parameters

/-- Follow the three hidden units from input to the triangular output. -/
def plot : ProofWidgets.Html := Plotting.networkPlots "With bias" parameters

#html plot

end NeuralNetworks.Examples.WithBias
