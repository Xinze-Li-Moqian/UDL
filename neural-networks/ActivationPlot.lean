module

public meta import ShallowNetworkExample
public meta import ProofWidgets.Component.HtmlDisplay
public meta import ProofWidgets.Component.Recharts

/-!
# Plotting the example network, one step at a time

Put the cursor on `#html activationPlots` below to view the charts in the Lean
Infoview. All panels use the same packaged parameters from `ShallowNetworkExample`.
The computation uses rational numbers; only the plotted coordinates become floats.
Samples include every corner at zero, one, and two; the lines are not smoothed.
-/

public meta section

open Lean ProofWidgets Recharts
open scoped ProofWidgets.Jsx

namespace NeuralNetworks

/-- The same 81 samples support every panel and can be exported as JSON. -/
def examplePlotData : Array Json :=
  let Φ := Examples.exampleParameters
  let number : ℚ → Json := fun q => toJson (q.num.toFloat / q.den.toFloat)
  (Array.range 81).map fun (i : Nat) =>
    let x : ℚ := (i : ℚ) / 20 - 1
    let z₁ := Φ.θ₁₀ + Φ.θ₁₁ * x
    let z₂ := Φ.θ₂₀ + Φ.θ₂₁ * x
    let z₃ := Φ.θ₃₀ + Φ.θ₃₁ * x
    let h₁ := hiddenUnit relu Φ.θ₁₀ Φ.θ₁₁ x
    let h₂ := hiddenUnit relu Φ.θ₂₀ Φ.θ₂₁ x
    let h₃ := hiddenUnit relu Φ.θ₃₀ Φ.θ₃₁ x
    let y := shallowNetworkWithParameters relu x Φ
    json% {x: $(number x),
      z1: $(number z₁), z2: $(number z₂), z3: $(number z₃),
      h1: $(number h₁), h2: $(number h₂), h3: $(number h₃),
      w1: $(number (Φ.φ₁ * h₁)), w2: $(number (Φ.φ₂ * h₂)),
      w3: $(number (Φ.φ₃ * h₃)), y: $(number y)}

/-- One labeled panel; the panels in each row share a vertical scale. -/
def examplePanel (label key color : String) (low high : Float) : Html :=
  <div>
    <h4 style={json% {color: $(toJson color)}}>{.text label}</h4>
    <LineChart width={260} height={220} data={examplePlotData}>
      <XAxis type={.number} domain?={#[toJson (-1), toJson 3]} dataKey?="x" />
      <YAxis type={.number} domain?={#[toJson low, toJson high]} />
      <Line type={.linear} dataKey={toJson key} stroke={color} dot?={Bool.false} />
    </LineChart>
  </div>

def exampleRow (panels : Array Html) : Html :=
  <div style={json% {display: "flex", flexWrap: "wrap", gap: "12px"}}>
    {...panels}
  </div>

/-- Follow each unit down its column, then add its weighted contribution. -/
def activationPlots : Html :=
  <div>
    <h2>One network: from input to output</h2>
    <p>Unit 1: purple; unit 2: teal; unit 3: orange. All panels use x from -1 to 3.</p>
    <h3>1. Affine inputs</h3>
    {exampleRow #[
      examplePanel "z1 = x" "z1" "#6f42c1" (-3.5) 3.5,
      examplePanel "z2 = x - 1" "z2" "#087f8c" (-3.5) 3.5,
      examplePanel "z3 = x - 2" "z3" "#cf6b38" (-3.5) 3.5]}
    <h3>2. Apply ReLU</h3>
    {exampleRow #[
      examplePanel "h1 = ReLU(x)" "h1" "#6f42c1" (-0.3) 3.3,
      examplePanel "h2 = ReLU(x - 1)" "h2" "#087f8c" (-0.3) 3.3,
      examplePanel "h3 = ReLU(x - 2)" "h3" "#cf6b38" (-0.3) 3.3]}
    <h3>3. Multiply by output weights</h3>
    {exampleRow #[
      examplePanel "w1 = h1" "w1" "#6f42c1" (-4.5) 3.5,
      examplePanel "w2 = -2 * h2" "w2" "#087f8c" (-4.5) 3.5,
      examplePanel "w3 = h3" "w3" "#cf6b38" (-4.5) 3.5]}
    <h3>4. Add the contributions and output bias</h3>
    {examplePanel "y = 0 + w1 + w2 + w3" "y" "#22364b" (-0.2) 1.2}
    <p>At x = 1.5: h = (1.5, 0.5, 0), so y = 0 + 1.5 - 1 + 0 = 0.5.</p>
  </div>

#html activationPlots

end NeuralNetworks

/-- Export the same sampled data used by the interactive panels. -/
def main : IO Unit :=
  IO.println (Lean.toJson NeuralNetworks.examplePlotData).compress
