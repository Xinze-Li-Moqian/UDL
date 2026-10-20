module

public import FittingModels.Adam.Basic

/-!
# Adam: aggregate the gradient before squaring

This is the standard Adam second-moment rule (Kingma and Ba, Algorithm 1),
applied to our summed batch loss:

    g = ∇L_batch(φ) = ∑ᵢ gᵢ,
    q = g² = (∑ᵢ gᵢ)².

Source: https://arxiv.org/abs/1412.6980. The locally consulted revised UDL
equation (6.18) also places the square outside the sum.

`batchStep` below selects this method explicitly. It uses the existing
`batchLoss`, with no averaging. The individual-gradient equality requires
differentiable sample losses. This differs from SumOfSquares when a batch
contains multiple examples, since gradient cross terms are included here.
-/

@[expose] public section

namespace FittingModels.Adam.SquareOfSum

open LossFunctions
open Optimization (AdamState)
open StochasticGradientDescent (Batch batchLoss)
open scoped Gradient

variable {X Prediction Y : Type*} {I B N : ℕ}

/-- Aggregate first, then square each coordinate. -/
noncomputable def secondMomentInput (g : Fin B → EuclideanSpace ℝ (Fin N)) :
    EuclideanSpace ℝ (Fin N) := square (∑ j, g j)

/-- Standard Adam on the summed batch loss. The shared update receives
q = square(g); it never substitutes the sum of individual squares. -/
noncomputable def batchStep [NeZero B] (ℓ : Prediction → Y → ℝ)
    (f : X → EuclideanSpace ℝ (Fin N) → Prediction) (x : Fin I → X) (y : Fin I → Y)
    (batch : Batch I B) (α β γ ε : ℝ)
    (state : AdamState (EuclideanSpace ℝ (Fin N))) :
    AdamState (EuclideanSpace ℝ (Fin N)) :=
  let g := ∇ (batchLoss ℓ f x y batch) state.φ
  stepFromStatistics g (square g) α β γ ε state

end FittingModels.Adam.SquareOfSum
