module

public import FittingModels.Adam.Basic

/-!
# Adam variant: square each sample gradient before summing

This implements the reading of UDL page 90, equation (6.18), where the
exponent is attached to each individual sample gradient:

    g = ∇L_batch(φ) = ∑ᵢ gᵢ,
    q = ∑ᵢ gᵢ².

When (6.18) instead places the square outside the sum, use SquareOfSum.
The standard Adam rule also uses SquareOfSum. The two readings define
distinct updates, not equivalent rewritings.

`batchStep` below selects this method explicitly. Each selected sample's
gradient is squared coordinatewise before summation. For example, gradients
1 and −1 give q = 2 here, whereas SquareOfSum gives q = 0. Both variants
share the summed batch loss, bias correction, and parameter update.
-/

@[expose] public section

namespace FittingModels.Adam.SumOfSquares

open LossFunctions
open Optimization (AdamState)
open StochasticGradientDescent (Batch batchLoss)
open scoped Gradient

variable {X Prediction Y : Type*} {I B N : ℕ}

/-- Square each sample gradient coordinatewise, then sum. -/
noncomputable def secondMomentInput (g : Fin B → EuclideanSpace ℝ (Fin N)) :
    EuclideanSpace ℝ (Fin N) := ∑ j, square (g j)

/-- The sum-of-squares variant, using q = sum of individual squared gradients.
As usual, gradient identities require differentiability of the sample losses. -/
noncomputable def batchStep [NeZero B] (ℓ : Prediction → Y → ℝ)
    (f : X → EuclideanSpace ℝ (Fin N) → Prediction) (x : Fin I → X) (y : Fin I → Y)
    (batch : Batch I B) (α β γ ε : ℝ)
    (state : AdamState (EuclideanSpace ℝ (Fin N))) :
    AdamState (EuclideanSpace ℝ (Fin N)) :=
  let g := ∇ (batchLoss ℓ f x y batch) state.φ
  let sampleGradients := fun j : Fin B =>
    ∇ (fun φ => ℓ (f (x (batch j)) φ) (y (batch j))) state.φ
  stepFromStatistics g (secondMomentInput sampleGradients) α β γ ε state

end FittingModels.Adam.SumOfSquares
