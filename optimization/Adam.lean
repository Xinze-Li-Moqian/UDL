module

/-!
# Adam state

Understanding Deep Learning, Section 6.4.

Adam stores the model parameters φ and two gradient statistics:

    m: exponentially weighted gradient history,
    v: exponentially weighted squared-gradient history.

Both statistics have the same coordinates as φ. The square used for v is
coordinatewise; v is a second-moment estimate, not a centered variance.
The step count t is used for the bias corrections 1 − βᵗ and 1 − γᵗ.

This file currently defines the state and initialization only.
-/

@[expose] public section

namespace Optimization

universe u

/-- Model parameters, the two gradient statistics, and the number of completed updates. -/
structure AdamState (Parameters : Type u) where
  /-- Current model parameters. -/
  φ : Parameters
  /-- Weighted gradient history (first moment). -/
  m : Parameters
  /-- Weighted coordinatewise squared-gradient history (second raw moment). -/
  v : Parameters
  /-- Number of completed updates; initially zero. -/
  t : Nat

/-- Start with supplied parameters, zero statistics, and no completed updates.

    (φ₀, m₀, v₀, t₀) = (φ₀, 0, 0, 0).
-/
def AdamState.initial {Parameters : Type u} [Zero Parameters]
    (φ : Parameters) : AdamState Parameters := ⟨φ, 0, 0, 0⟩

end Optimization
