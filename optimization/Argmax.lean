module

public import Mathlib.Data.Set.Defs
public import Mathlib.Util.Notation3

/-!
# Argmax

The set of global maximizers of an objective. With `open scoped Optimization`,
write `candidate ∈ argmax φ, objective φ`. No existence or uniqueness is assumed.
-/

@[expose] public section

namespace Optimization

/-- All points at which the objective attains a global maximum.
This set may be empty or contain several points. -/
def argmax {Domain Value : Type*} [LE Value] (objective : Domain → Value) : Set Domain :=
  {candidate | ∀ other, objective other ≤ objective candidate}

/-- Bind the candidate parameter in the objective being maximized. -/
scoped notation3 "argmax " (...) ", " objective:60:(scoped f => argmax f) => objective

@[simp] theorem mem_argmax {Domain Value : Type*} [LE Value]
    (objective : Domain → Value) (candidate : Domain) :
    candidate ∈ Optimization.argmax objective ↔
      ∀ other, objective other ≤ objective candidate := Iff.rfl

end Optimization
