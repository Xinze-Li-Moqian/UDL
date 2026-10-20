module

public import FittingModels.GradientDescent

/-!
# Mini-batch gradient updates

Understanding Deep Learning, Section 6.2, equation (6.10).

A batch selects B examples from I training pairs. Its loss is the sum

    L_batch(φ) = ∑ⱼ ℓ(f(x_batch(j), φ), y_batch(j)).

The update uses this loss instead of the full-data loss:

    φ_next = φ − α ∇L_batch(φ).

This file takes the batch as an input. Probability distributions and random
sampling belong to the sampling-specific files. As in the book, the loss is
summed without division by B, reusing the existing `empiricalLoss`.
-/

@[expose] public section

namespace FittingModels.StochasticGradientDescent

open LossFunctions

/-- B sample indices from a dataset of size I, in order. Repeated indices are
allowed; a sampling method may impose additional restrictions. -/
abbrev Batch (I B : ℕ) := Fin B → Fin I

variable {X Prediction Y Parameters : Type*} {I B : ℕ}

/-- Sum the existing empirical loss over the selected training pairs.

    L_batch[φ] = ∑ⱼ ℓ(f(x_batch(j), φ), y_batch(j)).

`NeZero B` excludes empty batches. Repeated indices contribute repeatedly.
-/
noncomputable def batchLoss [NeZero B] (ℓ : Prediction → Y → ℝ)
    (f : X → Parameters → Prediction) (x : Fin I → X) (y : Fin I → Y)
    (batch : Batch I B) : LossFunction ℝ Parameters :=
  empiricalLoss ℓ f (fun j => x (batch j)) (fun j => y (batch j))

section Update

variable [NormedAddCommGroup Parameters] [InnerProductSpace ℝ Parameters]
variable [CompleteSpace Parameters]

/-- One mini-batch update, reusing the mathematical gradient-descent step.

    φ_next = φ − α ∇L_batch(φ) = φ − α ∑ⱼ ∇ℓ_batch(j)(φ).

The gradient-sum interpretation requires differentiability of each sample loss.
This is a mathematical definition, not an executable differentiation routine.
-/
noncomputable def batchStep [NeZero B] (ℓ : Prediction → Y → ℝ)
    (f : X → Parameters → Prediction) (x : Fin I → X) (y : Fin I → Y)
    (batch : Batch I B) (α : ℝ) (φ : Parameters) : Parameters :=
  FittingModels.gradientDescentStep (batchLoss ℓ f x y batch) α φ

end Update

end FittingModels.StochasticGradientDescent
