module

public import DeepNetwork.Model
public import Mathlib.LinearAlgebra.Matrix.Notation

/-!
# A deep network with two hidden layers of three units

Source: Understanding Deep Learning, Section 4.3, equations (4.7)–(4.10), and
Section 4.4, equations (4.11)–(4.13).

The model is defined in `DeepNetwork.Model`. This module fixes one input, two
hidden layers of three units, and one output, and retains the book's named
parameters:

```
h₁ = a[θ₁₀ + θ₁₁ x]                        (4.7, first layer)
h'₁ = a[ψ₁₀ + ψ₁₁ h₁ + ψ₁₂ h₂ + ψ₁₃ h₃]      (4.8, second layer)
y' = φ'₀ + φ'₁ h'₁ + φ'₂ h'₂ + φ'₃ h'₃      (4.9, output)
```

and similarly for `h₂, h₃, h'₂, h'₃`. There are 22 parameters: six `θ`, twelve
`ψ`, and four `φ'`. `DeepParameters.toDeepNetworkParameters` arranges them into
the bias vectors and weight matrices of equations (4.11)–(4.13), giving
`DeepNetworkParameters Scalar 1 [3, 3] 1`.
-/

@[expose] public section

namespace NeuralNetworks.WidthThree

/-- The 22 named parameters of the two-layer network in equations (4.7)–(4.9). -/
structure DeepParameters (Scalar : Type) where
  θ₁₀ : Scalar
  θ₁₁ : Scalar
  θ₂₀ : Scalar
  θ₂₁ : Scalar
  θ₃₀ : Scalar
  θ₃₁ : Scalar
  ψ₁₀ : Scalar
  ψ₁₁ : Scalar
  ψ₁₂ : Scalar
  ψ₁₃ : Scalar
  ψ₂₀ : Scalar
  ψ₂₁ : Scalar
  ψ₂₂ : Scalar
  ψ₂₃ : Scalar
  ψ₃₀ : Scalar
  ψ₃₁ : Scalar
  ψ₃₂ : Scalar
  ψ₃₃ : Scalar
  φ'₀ : Scalar
  φ'₁ : Scalar
  φ'₂ : Scalar
  φ'₃ : Scalar

/-- Equations (4.11)–(4.13): the named parameters as bias vectors and weight matrices. -/
def DeepParameters.toDeepNetworkParameters {Scalar : Type} (Φ : DeepParameters Scalar) :
    DeepNetworkParameters Scalar 1 [3, 3] 1 :=
  (⟨![Φ.θ₁₀, Φ.θ₂₀, Φ.θ₃₀], !![Φ.θ₁₁; Φ.θ₂₁; Φ.θ₃₁]⟩,
    ⟨![Φ.ψ₁₀, Φ.ψ₂₀, Φ.ψ₃₀],
      !![Φ.ψ₁₁, Φ.ψ₁₂, Φ.ψ₁₃; Φ.ψ₂₁, Φ.ψ₂₂, Φ.ψ₂₃; Φ.ψ₃₁, Φ.ψ₃₂, Φ.ψ₃₃]⟩,
    ⟨![Φ.φ'₀], !![Φ.φ'₁, Φ.φ'₂, Φ.φ'₃]⟩)

/-- The two-layer, width-three network evaluated at a scalar input `x`. -/
def deepNetwork {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar) (x : Scalar) (Φ : DeepParameters Scalar) : Scalar :=
  NeuralNetworks.deepNetwork activation Φ.toDeepNetworkParameters (fun _ => x) 0

end NeuralNetworks.WidthThree
