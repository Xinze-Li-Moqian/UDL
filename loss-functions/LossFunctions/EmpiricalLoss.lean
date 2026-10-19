module

public import Mathlib.Algebra.BigOperators.Fin

/-!
# Empirical loss

Fix a model `f`, training pairs `(xᵢ, yᵢ)`, and a single-example loss `ℓ`.
The empirical loss is

    L[φ] = ∑ᵢ ℓ(f(xᵢ, φ), yᵢ),    i = 1, …, I.

```lean
open scoped LossFunctions

let L := empiricalLoss ℓ f x y
L[φ]
```

Least squares and negative log-likelihood supply different choices of `ℓ`.
-/

@[expose] public section

namespace LossFunctions

open scoped BigOperators

universe u v w r s

/-- With the model and data fixed, `L[φ]` is the scalar objective to minimize.
`Scalar` is typically `ℝ` or `ℚ`; no nonnegativity or regularity is assumed. -/
abbrev LossFunction (Scalar : Type u) (Parameters : Type v) :=
  Parameters → Scalar

variable {Scalar : Type u} {Parameters : Type v}

/-- With `open scoped LossFunctions`, `L[φ]` evaluates the loss at `φ`. -/
scoped instance :
    GetElem (LossFunction Scalar Parameters) Parameters Scalar (fun _ _ => True) where
  getElem L φ _ := L φ

/-- Bracket notation evaluates the same loss function. -/
@[simp] theorem lossFunction_getElem
    (L : LossFunction Scalar Parameters) (φ : Parameters) : L[φ] = L φ := rfl

variable {X : Type w} {Prediction : Type r} {Y : Type s} {I : Nat}

/-- Sum the single-example losses over the training pairs, without averaging.

    L[φ] = ∑ᵢ ℓ(f(xᵢ, φ), yᵢ),    i = 1, …, I.

Predictions may be values or distribution parameters, so `Prediction` and `Y`
are separate types. -/
def empiricalLoss [AddCommMonoid Scalar] (ℓ : Prediction → Y → Scalar)
    (f : X → Parameters → Prediction) (x : Fin I → X) (y : Fin I → Y) :
    LossFunction Scalar Parameters :=
  fun φ => ∑ i : Fin I, ℓ (f (x i) φ) (y i)

/-- Evaluate losses in index order with right-associated addition.
This also supports floating-point arithmetic, whose addition is not associative.
The order is `ℓ₀ + (ℓ₁ + … + (ℓ_last + 0))`. -/
def orderedEmpiricalLoss [Zero Scalar] [Add Scalar] (ℓ : Prediction → Y → Scalar)
    (f : X → Parameters → Prediction) (x : Fin I → X) (y : Fin I → Y) :
    LossFunction Scalar Parameters :=
  fun φ => (List.ofFn fun i : Fin I => ℓ (f (x i) φ) (y i)).sum

/-- Under associative, commutative addition, ordered evaluation is the same sum. -/
theorem orderedEmpiricalLoss_eq_empiricalLoss [AddCommMonoid Scalar]
    (ℓ : Prediction → Y → Scalar) (f : X → Parameters → Prediction)
    (x : Fin I → X) (y : Fin I → Y) :
    orderedEmpiricalLoss ℓ f x y = empiricalLoss ℓ f x y := by
  funext φ
  simp only [orderedEmpiricalLoss, empiricalLoss, List.sum_ofFn]

end LossFunctions
