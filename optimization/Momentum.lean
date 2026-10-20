module

/-!
# Momentum update

Understanding Deep Learning, Section 6.3, equation (6.11).
The caller supplies the gradient g at the current parameters:

    m_next = β m + (1 − β) g,
    φ_next = φ − α m_next.

The usual choices are α > 0 and 0 ≤ β < 1. The update itself only needs
arithmetic operations, so it also supports executable numeric types.
-/

@[expose] public section

namespace Optimization

universe u v

/-- Current parameters and the accumulated gradient direction. -/
structure MomentumState (Parameters : Type v) where
  φ : Parameters
  m : Parameters

/-- Start with the supplied parameters and zero momentum. -/
def MomentumState.initial {Parameters : Type v} [Zero Parameters]
    (φ : Parameters) : MomentumState Parameters := ⟨φ, 0⟩

/-- Update momentum first, then use the new momentum to update parameters.
`g` must be evaluated at `state.φ`; no previous parameter update is applied. -/
def momentumStep {Scalar : Type u} {Parameters : Type v}
    [Sub Scalar] [OfNat Scalar 1] [SMul Scalar Parameters]
    [Add Parameters] [Sub Parameters]
    (g : Parameters) (α β : Scalar) (state : MomentumState Parameters) :
    MomentumState Parameters :=
  let m_next := β • state.m + (1 - β) • g
  ⟨state.φ - α • m_next, m_next⟩

end Optimization
