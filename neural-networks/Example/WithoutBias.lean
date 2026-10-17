module

public meta import Example.Plot

/-!
# A shallow network without biases

Run `lake env lean neural-networks/Example/WithoutBias.lean` for the numerical
example, or put the cursor on `#html plot` to view the charts in Lean Infoview.

All hidden biases and the output bias are zero. The input and output weights
match the example with biases. All three hidden units compute `ReLU(x)`, so
these particular output weights cancel:
`y = ReLU(x) - 2*ReLU(x) + ReLU(x) = 0`.
This cancellation depends on the chosen weights; networks without biases do
not always output zero.

Each call `relu z` below omits the optional second argument, whose default
bias is zero. The numerical example uses exact rational arithmetic.
-/

public meta section

namespace NeuralNetworks.Examples.WithoutBias

/-- The same weights as the biased example, with every bias set to zero. -/
def parameters : ShallowNetworkParameters ℚ 3 :=
  { outputBias := 0
    outputWeights := ![1, -2, 1]
    hiddenBiases := ![0, 0, 0]
    hiddenWeights := ![1, 1, 1] }

/-- The same inputs are used by the example with biases. -/
def inputs : List ℚ := [-1, 0, 1 / 2, 1, 3 / 2, 2, 3]

#eval do
  let Φ := parameters
  IO.println "Without bias: h1 = h2 = h3 = ReLU(x)"
  IO.println s!"Network capacity (hidden units): {networkCapacity Φ}"
  IO.println "y = h1 - 2*h2 + h3 (exact rational arithmetic)"
  for x in inputs do
    let h₁ := relu (Φ.hiddenWeights 0 * x)
    let h₂ := relu (Φ.hiddenWeights 1 * x)
    let h₃ := relu (Φ.hiddenWeights 2 * x)
    IO.println s!"x = {x}: h = ({h₁}, {h₂}, {h₃}), y = {shallowNetwork relu x Φ}"

/-- Samples for the affine inputs, activations, contributions, and output. -/
def plotData : Array Lean.Json := Plotting.networkPlotData parameters

/-- Follow the three hidden units and the cancellation of their contributions. -/
def plot : ProofWidgets.Html := Plotting.networkPlots "Without bias" parameters

#html plot

end NeuralNetworks.Examples.WithoutBias
