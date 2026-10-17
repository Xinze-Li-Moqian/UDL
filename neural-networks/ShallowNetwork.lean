module

public import ActivationFunction

/-!
# A network with three hidden units

Source: Understanding Deep Learning, Section 3.1.1, equations (3.3)–(3.4).

Each hidden unit applies the same activation function to its own affine
transformation of the input. The output weights these three activations,
sums them, and adds a bias. There is no activation on the output.
-/

@[expose] public section

namespace NeuralNetworks

/-- A hidden unit from equation (3.3): `h(x) = a[bias + weight * x]`. -/
def hiddenUnit {Scalar : Type} [Add Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (bias weight x : Scalar) : Scalar :=
  activation (bias + weight * x)

/-- The one-input, three-hidden-unit, one-output network.
The six θ parameters define the hidden units; the four φ parameters combine
their activations according to equation (3.4). Choose `activation = relu`
for the ReLU network in the book. -/
def shallowNetwork {Scalar : Type} [Add Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (φ₀ φ₁ φ₂ φ₃ : Scalar)
    (θ₁₀ θ₁₁ θ₂₀ θ₂₁ θ₃₀ θ₃₁ : Scalar)
    (x : Scalar) : Scalar :=
  -- Equation (3.3): three hidden units receive the same input x.
  let h₁ := hiddenUnit activation θ₁₀ θ₁₁ x
  let h₂ := hiddenUnit activation θ₂₀ θ₂₁ x
  let h₃ := hiddenUnit activation θ₃₀ θ₃₁ x
  -- Equation (3.4): combine the hidden activations.
  φ₀ + φ₁ * h₁ + φ₂ * h₂ + φ₃ * h₃

/-- The ten parameters, collected into the book's single parameter object `Φ`.
The φ fields combine the hidden activations; each θ pair defines a hidden unit. -/
structure ShallowNetworkParameters (Scalar : Type) where
  φ₀ : Scalar
  φ₁ : Scalar
  φ₂ : Scalar
  φ₃ : Scalar
  θ₁₀ : Scalar
  θ₁₁ : Scalar
  θ₂₀ : Scalar
  θ₂₁ : Scalar
  θ₃₀ : Scalar
  θ₃₁ : Scalar

/-- The same network written as `fₐ(x; Φ)`, with all ten parameters packaged.
Fixing `activation` gives a function of the input `x` and parameter object `Φ`. -/
def shallowNetworkWithParameters {Scalar : Type} [Add Scalar] [Mul Scalar]
    (activation : ActivationFunction Scalar)
    (x : Scalar) (Φ : ShallowNetworkParameters Scalar) : Scalar :=
  shallowNetwork activation
    Φ.φ₀ Φ.φ₁ Φ.φ₂ Φ.φ₃
    Φ.θ₁₀ Φ.θ₁₁ Φ.θ₂₀ Φ.θ₂₁ Φ.θ₃₀ Φ.θ₃₁
    x

end NeuralNetworks
