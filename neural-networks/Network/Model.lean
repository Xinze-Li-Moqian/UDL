module

public import ActivationFunction
public import Mathlib.Data.Matrix.Mul
public import Mathlib.Algebra.BigOperators.Fin

/-!
# Neural networks in matrix notation

Source: Understanding Deep Learning, Section 4.4, equations (4.15)–(4.16).

This module is the common foundation for the networks in this repository.
Vectors, matrices, and their operations come from Mathlib: a vector of size `D`
is `Fin D → Scalar`, a `D' × D` matrix is `Matrix (Fin D') (Fin D) Scalar`, and
`Ω *ᵥ h` multiplies a matrix by a vector. Only the book's network terms are
defined here. A network with `K` hidden layers computes

```
h₁ = a[β₀ + Ω₀ x]
h_{k+1} = a[β_k + Ω_k h_k]
y = β_K + Ω_K h_K
```

where the activation `a` acts on every component, written `a ∘ v`.

`Architecture` holds the hyperparameters: the depth `K` and the layer sizes
`D₀, D₁, …, D_K, D_{K+1}`, where `D₀ = Dᵢ` is the input size and `D_{K+1} = Dₒ`
the output size. Hidden layers keep the book's numbering from `1` to `K`, so
`A.size k` is `D_k`. `Architecture.ofList Dᵢ [D₁, …, D_K] Dₒ` builds an
architecture from its hidden widths.

`Architecture.parameterCount` counts all scalar entries in the bias vectors and
weight matrices of a fully connected network: `∑ k, D_{k+1} * (D_k + 1)`.

`NetworkParameters Scalar A` holds the parameters `φ = {β_k, Ω_k}` for
`k = 0, …, K`: `β_k` has size `D_{k+1}` and `Ω_k` is `D_{k+1} × D_k`, so the sizes
of consecutive layers always match. `hidden a Φ x k` is the hidden layer `h_k`,
with `h₀ = x` standing for the input, and `network a Φ x` is the output `y`.

Parameters can also be built layer by layer, as in the nested form of equation
(4.16): `single β Ω` is a network with no hidden layers, `y = β + Ω x`, and
`cons β₀ Ω₀ rest` puts the hidden layer `a[β₀ + Ω₀ x]` in front of `rest`.
`network_single` and `network_cons` evaluate networks built this way.

`networkFamily a A` is the family of functions described by the architecture `A`
as the parameters vary (Section 4.3.1). Letting the hyperparameters vary too gives
the book's "family of families".
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

/-- The hyperparameters of a network: the depth `K` and the layer sizes
`D₀ = Dᵢ, D₁, …, D_K, D_{K+1} = Dₒ`. -/
structure Architecture where
  /-- The number of hidden layers `K`. -/
  depth : ℕ
  /-- The layer sizes; `size 0` is the input size and `size (K + 1)` the output size. -/
  size : Fin (depth + 2) → ℕ

namespace Architecture

/-- The input size `Dᵢ = D₀`. -/
def inputs (A : Architecture) : ℕ := A.size 0

/-- The output size `Dₒ = D_{K+1}`. -/
def outputs (A : Architecture) : ℕ := A.size (Fin.last _)

/-- The capacity: the total number of hidden units `D₁ + ⋯ + D_K`. -/
def capacity (A : Architecture) : ℕ := ∑ k : Fin A.depth, A.size k.succ.castSucc

/-- The number of scalar parameters: `β_k` has `D_{k+1}` entries and `Ω_k` has
`D_{k+1} * D_k` entries. All entries are unrestricted in the fully connected model. -/
def parameterCount (A : Architecture) : ℕ :=
  ∑ k : Fin (A.depth + 1), A.size k.succ * (A.size k.castSucc + 1)

/-- The architecture with input size `Dᵢ`, hidden widths `[D₁, …, D_K]`, and output
size `Dₒ`. -/
def ofList (Dᵢ : ℕ) (widths : List ℕ) (Dₒ : ℕ) : Architecture where
  depth := widths.length
  size k := (Dᵢ :: widths ++ [Dₒ]).getD k 0

end Architecture

/-- Equation (4.15): the parameters `φ = {β_k, Ω_k}` for `k = 0, …, K`.
`β_k` has size `D_{k+1}` and `Ω_k` is `D_{k+1} × D_k`. -/
structure NetworkParameters (Scalar : Type) (A : Architecture) where
  /-- The bias vectors `β_k`. -/
  β : (k : Fin (A.depth + 1)) → Fin (A.size k.succ) → Scalar
  /-- The weight matrices `Ω_k`. -/
  Ω : (k : Fin (A.depth + 1)) → Matrix (Fin (A.size k.succ)) (Fin (A.size k.castSucc)) Scalar

section Evaluation

variable {Scalar : Type} [NonUnitalNonAssocSemiring Scalar] {A : Architecture}

/-- The hidden layers `h_k`: `h₀ = x` is the input and `h_{k+1} = a[β_k + Ω_k h_k]`. -/
def hidden (a : ActivationFunction Scalar) (Φ : NetworkParameters Scalar A)
    (x : Fin A.inputs → Scalar) : (k : Fin (A.depth + 1)) → Fin (A.size k.castSucc) → Scalar :=
  Fin.induction x fun k h => a ∘ (Φ.β k.castSucc + Φ.Ω k.castSucc *ᵥ h)

/-- Equation (4.15): the output `y = β_K + Ω_K h_K`. -/
def network (a : ActivationFunction Scalar) (Φ : NetworkParameters Scalar A)
    (x : Fin A.inputs → Scalar) : Fin A.outputs → Scalar :=
  Φ.β (Fin.last _) + Φ.Ω (Fin.last _) *ᵥ hidden a Φ x (Fin.last _)

/-- Section 4.3.1: for fixed hyperparameters `A`, the family of functions the network
describes as its parameters vary. -/
def networkFamily (a : ActivationFunction Scalar) (A : Architecture) :
    Set ((Fin A.inputs → Scalar) → (Fin A.outputs → Scalar)) :=
  Set.range fun Φ : NetworkParameters Scalar A => network a Φ

@[simp] theorem hidden_zero (a : ActivationFunction Scalar) (Φ : NetworkParameters Scalar A)
    (x : Fin A.inputs → Scalar) : hidden a Φ x 0 = x :=
  Fin.induction_zero _ _

@[simp] theorem hidden_succ (a : ActivationFunction Scalar) (Φ : NetworkParameters Scalar A)
    (x : Fin A.inputs → Scalar) (k : Fin A.depth) :
    hidden a Φ x k.succ = a ∘ (Φ.β k.castSucc + Φ.Ω k.castSucc *ᵥ hidden a Φ x k.castSucc) :=
  Fin.induction_succ _ _ _

end Evaluation

namespace NetworkParameters

variable {Scalar : Type}

/-- A network with no hidden layers: the single affine layer `y = β + Ω x`. -/
def single {Dᵢ Dₒ : ℕ} (β : Fin Dₒ → Scalar) (Ω : Matrix (Fin Dₒ) (Fin Dᵢ) Scalar) :
    NetworkParameters Scalar (Architecture.ofList Dᵢ [] Dₒ) where
  β := Fin.cases β fun k => Fin.elim0 k
  Ω := Fin.cases Ω fun k => Fin.elim0 k

/-- Put the hidden layer `a[β₀ + Ω₀ x]` in front of the network `rest`. -/
def cons {Dᵢ D Dₒ : ℕ} {widths : List ℕ} (β₀ : Fin D → Scalar)
    (Ω₀ : Matrix (Fin D) (Fin Dᵢ) Scalar)
    (rest : NetworkParameters Scalar (Architecture.ofList D widths Dₒ)) :
    NetworkParameters Scalar (Architecture.ofList Dᵢ (D :: widths) Dₒ) where
  β := Fin.cases β₀ rest.β
  Ω := Fin.cases Ω₀ rest.Ω

end NetworkParameters

section Layers

variable {Scalar : Type} [NonUnitalNonAssocSemiring Scalar]

open NetworkParameters

/-- A network with no hidden layers is the affine map `β + Ω x`. -/
@[simp] theorem network_single (a : ActivationFunction Scalar) {Dᵢ Dₒ : ℕ}
    (β : Fin Dₒ → Scalar) (Ω : Matrix (Fin Dₒ) (Fin Dᵢ) Scalar) (x : Fin Dᵢ → Scalar) :
    network a (single β Ω) x = β + Ω *ᵥ x :=
  rfl

/-- After the first hidden layer, `cons` continues with the hidden layers of `rest`. -/
theorem hidden_cons (a : ActivationFunction Scalar) {Dᵢ D Dₒ : ℕ} {widths : List ℕ}
    (β₀ : Fin D → Scalar) (Ω₀ : Matrix (Fin D) (Fin Dᵢ) Scalar)
    (rest : NetworkParameters Scalar (Architecture.ofList D widths Dₒ)) (x : Fin Dᵢ → Scalar)
    (k : Fin (widths.length + 1)) :
    hidden a (cons β₀ Ω₀ rest) x k.succ =
      hidden a rest (a ∘ (β₀ + Ω₀ *ᵥ x) : Fin D → Scalar) k := by
  induction k using Fin.induction with
  | zero => rfl
  | succ k ih =>
    show a ∘ (rest.β k.castSucc + rest.Ω k.castSucc *ᵥ hidden a (cons β₀ Ω₀ rest) x k.castSucc.succ)
      = hidden a rest (a ∘ (β₀ + Ω₀ *ᵥ x)) k.succ
    rw [ih]
    exact (hidden_succ a rest _ k).symm

/-- A network built with `cons` applies its first hidden layer, then the rest. -/
@[simp] theorem network_cons (a : ActivationFunction Scalar) {Dᵢ D Dₒ : ℕ} {widths : List ℕ}
    (β₀ : Fin D → Scalar) (Ω₀ : Matrix (Fin D) (Fin Dᵢ) Scalar)
    (rest : NetworkParameters Scalar (Architecture.ofList D widths Dₒ)) (x : Fin Dᵢ → Scalar) :
    network a (cons β₀ Ω₀ rest) x = network a rest (a ∘ (β₀ + Ω₀ *ᵥ x) : Fin D → Scalar) := by
  show rest.β (Fin.last _) + rest.Ω (Fin.last _) *ᵥ hidden a (cons β₀ Ω₀ rest) x (Fin.last _).succ
    = _
  rw [hidden_cons]
  rfl

end Layers

end NeuralNetworks
