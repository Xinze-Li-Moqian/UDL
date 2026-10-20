module

public import Momentum
public import FittingModels.StochasticGradientDescent.Basic

/-!
# Momentum on a loss

Use the existing loss, gradient, and batch interfaces with equation (6.11):

    g = ∇L(φ),
    m_next = β m + (1 − β) g,
    φ_next = φ − α m_next.

For SGD, L is the current batch's summed loss. Sampling is independent of
the update rule and remains in the SGD sampling modules. These definitions
use the mathematical gradient; its interpretation requires differentiability.
-/

@[expose] public section

namespace FittingModels.Momentum

open LossFunctions
open Optimization (MomentumState)
open StochasticGradientDescent (Batch batchLoss)
open scoped Gradient

variable {Parameters : Type*} [NormedAddCommGroup Parameters]
variable [InnerProductSpace ℝ Parameters] [CompleteSpace Parameters]

/-- Apply momentum to the gradient of a loss at the current parameters. -/
noncomputable def momentumStep (L : LossFunction ℝ Parameters) (α β : ℝ)
    (state : MomentumState Parameters) : MomentumState Parameters :=
  Optimization.momentumStep (∇ L state.φ) α β state

/-- Use the current batch loss in the same momentum update.
This replaces the ordinary batch update rather than following it. -/
noncomputable def batchMomentumStep {X Prediction Y : Type*} {I B : ℕ} [NeZero B]
    (ℓ : Prediction → Y → ℝ) (f : X → Parameters → Prediction)
    (x : Fin I → X) (y : Fin I → Y) (batch : Batch I B)
    (α β : ℝ) (state : MomentumState Parameters) : MomentumState Parameters :=
  momentumStep (batchLoss ℓ f x y batch) α β state

/-- With β = 0, the parameter update reduces to ordinary gradient descent. -/
lemma momentumStep_zero (L : LossFunction ℝ Parameters) (α : ℝ)
    (state : MomentumState Parameters) :
    (momentumStep L α 0 state).φ = FittingModels.gradientDescentStep L α state.φ := by
  simp [momentumStep, Optimization.momentumStep, FittingModels.gradientDescentStep,
    Optimization.gradientDescentStep]

end FittingModels.Momentum
