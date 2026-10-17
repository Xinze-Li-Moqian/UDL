module

public meta import ActivationFunction
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Component.Recharts

/-!
# Plotting activation functions in Lean

Put the cursor on `#html activationPlots` below to view the charts in the Lean
Infoview. Both curves evaluate the definitions in `ActivationFunction.lean`.
Samples include the corners at zero and one; linear interpolation preserves them.
-/

public meta section

open Lean ProofWidgets Recharts
open scoped ProofWidgets.Jsx

namespace NeuralNetworks

/-- Plot a scalar function on `[-5, 5]` using 101 evenly spaced samples. -/
def activationPlot (f : Float → Float) (color : String) : Html :=
  let data : Array Json := (Array.range 101).map (fun i =>
    let x := i.toFloat / 10 - 5
    json% {x: $(toJson x), y: $(toJson (f x))});
  <LineChart width={360} height={280} data={data}>
    <XAxis type={.number} domain?={#[toJson (-5), toJson 5]} dataKey?="x" />
    <YAxis type={.number} domain?={#[toJson (-1), toJson 5]} />
    <Line type={.linear} dataKey="y" stroke={color} dot?={Bool.false} />
  </LineChart>

/-- Compare ReLU with a hidden unit whose bias shifts its corner to `x = 1`. -/
def activationPlots : Html :=
  <div style={json% {display: "flex", flexWrap: "wrap", gap: "16px"}}>
    <div>
      <h3>ReLU</h3>
      <p style={json% {color: "#6f42c1"}}>Purple: y = ReLU(x)</p>
      {activationPlot relu "#6f42c1"}
    </div>
    <div>
      <h3>Hidden unit</h3>
      <p style={json% {color: "#087f8c"}}>Teal: y = ReLU(x - 1); bias = -1, weight = 1</p>
      {activationPlot (hiddenUnit relu (-1) 1) "#087f8c"}
    </div>
  </div>

#html activationPlots

end NeuralNetworks
