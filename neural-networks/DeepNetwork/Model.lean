module

public import ActivationFunction
public import Mathlib.Data.Matrix.Mul

/-!
# Deep neural networks

Source: Understanding Deep Learning, Sections 4.3–4.4, equations (4.15)–(4.16).

A deep network with `Dᵢ` inputs, `K` hidden layers of widths `D₁, …, D_K`, and
`Dₒ` outputs computes

```
h₁ = a[β₀ + Ω₀ x]
h_k = a[β_{k-1} + Ω_{k-1} h_{k-1}]     for k = 2, …, K
y = β_K + Ω_K h_K
```

The activation `a` is applied to every component. Each layer stores a bias
vector `β_k` and a weight matrix `Ω_k` (`LayerParameters`). `Ω₀` is `D₁ × Dᵢ`,
`Ω_k` is `D_{k+1} × D_k`, and `Ω_K` is `Dₒ × D_K`.

The network is a composition of layers, as in equation (4.16): each hidden
layer maps `h` to `a[β + Ω h]`, and the output layer maps `h` to `β + Ω h`.
`DeepNetworkParameters Scalar Dᵢ widths Dₒ` holds one `LayerParameters` per
layer, built by recursion on the list of hidden widths. Consecutive layers
therefore have matching sizes by construction.

The hyperparameters `K = widths.length` and `D₁, …, D_K` are part of the type,
while the parameters `β_k` and `Ω_k` are its values. For example,
`DeepNetworkParameters ℝ 1 [3, 3] 1` is the family of networks with one input,
two hidden layers of three units, and one output (Section 4.3). With `widths = []`
there are no hidden layers and the network is a single affine map.
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

/-- The bias vector `β` and weight matrix `Ω` of a layer from `Dᵢ` to `Dₒ` units. -/
structure LayerParameters (Scalar : Type) (Dᵢ Dₒ : ℕ) where
  /-- The bias vector `β`, one entry per unit of the next layer. -/
  bias : Fin Dₒ → Scalar
  /-- The weight matrix `Ω`, of size `Dₒ × Dᵢ`. -/
  weights : Matrix (Fin Dₒ) (Fin Dᵢ) Scalar

/-- The affine map `h ↦ β + Ω h` of a layer. -/
def LayerParameters.apply {Scalar : Type} [NonUnitalNonAssocSemiring Scalar] {Dᵢ Dₒ : ℕ}
    (L : LayerParameters Scalar Dᵢ Dₒ) (h : Fin Dᵢ → Scalar) : Fin Dₒ → Scalar :=
  addBias L.bias (L.weights *ᵥ h)

/-- Parameters of a deep network with `Dᵢ` inputs, hidden widths `widths`, and `Dₒ` outputs:
one layer per hidden width, followed by the output layer. -/
def DeepNetworkParameters (Scalar : Type) : ℕ → List ℕ → ℕ → Type
  | Dᵢ, [], Dₒ => LayerParameters Scalar Dᵢ Dₒ
  | Dᵢ, D :: widths, Dₒ => LayerParameters Scalar Dᵢ D × DeepNetworkParameters Scalar D widths Dₒ

/-- Equation (4.16): apply each hidden layer `h ↦ a[β + Ω h]` in turn, then the
output layer `h ↦ β + Ω h`. -/
def deepNetwork {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]
    (activation : ActivationFunction Scalar) {Dₒ : ℕ} :
    {Dᵢ : ℕ} → {widths : List ℕ} →
      DeepNetworkParameters Scalar Dᵢ widths Dₒ → (Fin Dᵢ → Scalar) → (Fin Dₒ → Scalar)
  | _, [], L, x => L.apply x
  | _, _ :: _, (L, rest), x => deepNetwork activation rest fun d => activation (L.apply x d)

end NeuralNetworks
