module

public meta import ShallowNetwork.Model
public meta import Mathlib.Algebra.Field.Rat
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Component.Recharts

/-!
# Shared plots for the width-three examples

Each example supplies its own parameters and displays `networkPlots` in the Lean
Infoview. This example module contains only sampling and rendering helpers.
The computation uses rational numbers; only the plotted coordinates become floats.
Both examples use the same axes, including a range down to -6.5 for the weighted
contributions. Samples include the corners at zero, one, and two; lines are not
smoothed.
-/

public meta section

open Lean ProofWidgets Recharts
open scoped ProofWidgets.Jsx

namespace NeuralNetworks.Plotting

/-- The same 81 samples support every panel and can be exported as JSON. -/
def networkPlotData (Φ : ShallowNetworkParameters ℚ 3) : Array Json :=
  let number : ℚ → Json := fun q => toJson (q.num.toFloat / q.den.toFloat)
  (Array.range 81).map fun (i : Nat) =>
    let x : ℚ := (i : ℚ) / 20 - 1
    let z₁ := addBias (Φ.hiddenBiases 0) (Φ.hiddenWeights 0 * x)
    let z₂ := addBias (Φ.hiddenBiases 1) (Φ.hiddenWeights 1 * x)
    let z₃ := addBias (Φ.hiddenBiases 2) (Φ.hiddenWeights 2 * x)
    let h₁ := hiddenUnit relu (Φ.hiddenBiases 0) (Φ.hiddenWeights 0) x
    let h₂ := hiddenUnit relu (Φ.hiddenBiases 1) (Φ.hiddenWeights 1) x
    let h₃ := hiddenUnit relu (Φ.hiddenBiases 2) (Φ.hiddenWeights 2) x
    let y := shallowNetwork relu x Φ
    json% {x: $(number x),
      z1: $(number z₁), z2: $(number z₂), z3: $(number z₃),
      h1: $(number h₁), h2: $(number h₂), h3: $(number h₃),
      w1: $(number (Φ.outputWeights 0 * h₁)), w2: $(number (Φ.outputWeights 1 * h₂)),
      w3: $(number (Φ.outputWeights 2 * h₃)), y: $(number y)}

private def scaledExpression (weight : ℚ) (value : String) : String :=
  if weight == 1 then value
  else if weight == -1 then s!"-{value}"
  else s!"{weight} * {value}"

private def affineExpression (bias weight : ℚ) : String :=
  let value := scaledExpression weight "x"
  if bias == 0 then value
  else if 0 < bias then s!"{value} + {bias}"
  else s!"{value} - {-bias}"

/-- One labeled panel; the panels in each row share a vertical scale. -/
private def plotPanel (data : Array Json) (label key color : String)
    (low high : Float) : Html :=
  <div>
    <h4 style={json% {color: $(toJson color)}}>{.text label}</h4>
    <LineChart width={260} height={220} data={data}>
      <XAxis type={.number} domain?={#[toJson (-1), toJson 3]} dataKey?="x" />
      <YAxis type={.number} domain?={#[toJson low, toJson high]} />
      <Line type={.linear} dataKey={toJson key} stroke={color} dot?={Bool.false} />
    </LineChart>
  </div>

private def plotRow (panels : Array Html) : Html :=
  <div style={json% {display: "flex", flexWrap: "wrap", gap: "12px"}}>
    {...panels}
  </div>

/-- Plot all stages using the supplied parameters, with matching formula labels. -/
def networkPlots (title : String) (Φ : ShallowNetworkParameters ℚ 3) : Html :=
  let panel := plotPanel (networkPlotData Φ)
  let z₁ := affineExpression (Φ.hiddenBiases 0) (Φ.hiddenWeights 0)
  let z₂ := affineExpression (Φ.hiddenBiases 1) (Φ.hiddenWeights 1)
  let z₃ := affineExpression (Φ.hiddenBiases 2) (Φ.hiddenWeights 2)
  let h₁ := hiddenUnit relu (Φ.hiddenBiases 0) (Φ.hiddenWeights 0) (3 / 2)
  let h₂ := hiddenUnit relu (Φ.hiddenBiases 1) (Φ.hiddenWeights 1) (3 / 2)
  let h₃ := hiddenUnit relu (Φ.hiddenBiases 2) (Φ.hiddenWeights 2) (3 / 2)
  let w₁ := Φ.outputWeights 0 * h₁
  let w₂ := Φ.outputWeights 1 * h₂
  let w₃ := Φ.outputWeights 2 * h₃
  let y := shallowNetwork relu (3 / 2) Φ;
  <div>
    <h2>{.text s!"{title}: from input to output"}</h2>
    <p>Unit 1: purple; unit 2: teal; unit 3: orange. All panels use x from -1 to 3.</p>
    <p>{.text s!"Hidden biases: ({Φ.hiddenBiases 0}, {Φ.hiddenBiases 1}, {Φ.hiddenBiases 2}); output bias: {Φ.outputBias}."}</p>
    <h3>1. Affine inputs</h3>
    {plotRow #[
      panel s!"z1 = {z₁}" "z1" "#6f42c1" (-3.5) 3.5,
      panel s!"z2 = {z₂}" "z2" "#087f8c" (-3.5) 3.5,
      panel s!"z3 = {z₃}" "z3" "#cf6b38" (-3.5) 3.5]}
    <h3>2. Apply ReLU</h3>
    {plotRow #[
      panel s!"h1 = ReLU({z₁})" "h1" "#6f42c1" (-0.3) 3.3,
      panel s!"h2 = ReLU({z₂})" "h2" "#087f8c" (-0.3) 3.3,
      panel s!"h3 = ReLU({z₃})" "h3" "#cf6b38" (-0.3) 3.3]}
    <h3>3. Multiply by output weights</h3>
    {plotRow #[
      panel s!"w1 = {scaledExpression (Φ.outputWeights 0) "h1"}" "w1" "#6f42c1" (-6.5) 3.5,
      panel s!"w2 = {scaledExpression (Φ.outputWeights 1) "h2"}" "w2" "#087f8c" (-6.5) 3.5,
      panel s!"w3 = {scaledExpression (Φ.outputWeights 2) "h3"}" "w3" "#cf6b38" (-6.5) 3.5]}
    <h3>4. Add the contributions and output bias</h3>
    {panel s!"y = {Φ.outputBias} + w1 + w2 + w3" "y" "#4c7899" (-0.2) 1.2}
    <p>{.text s!"At x = 3/2: h = ({h₁}, {h₂}, {h₃}); weighted contributions = ({w₁}, {w₂}, {w₃}); y = {y}."}</p>
  </div>

end NeuralNetworks.Plotting
