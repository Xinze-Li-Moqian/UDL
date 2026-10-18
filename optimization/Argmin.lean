module

public import Mathlib.Data.Set.Defs
public import Mathlib.Util.Notation3

/-!
# Argmin

A general optimality condition for an objective function.
This module is independent of supervised learning and any particular model.
With `open scoped Optimization`, write `candidate ∈ argmin φ, objective φ`.
-/

@[expose] public section

namespace Optimization

/-- `candidate` belongs to the argmin of `objective`: its value is no greater
than the value at any other point. This specifies global optimality, without
assuming that a minimizer exists or is unique, or supplying a solver. -/
def IsArgmin {Domain Value : Type*} [LE Value]
    (objective : Domain → Value) (candidate : Domain) : Prop :=
  ∀ other : Domain, objective candidate ≤ objective other

/-- The set of global minimizers, described by the existing optimality predicate. -/
def argmin {Domain Value : Type*} [LE Value] (objective : Domain → Value) : Set Domain :=
  {candidate | IsArgmin objective candidate}

/-- Bind the candidate parameter in the objective being minimized. -/
scoped notation3 "argmin " (...) ", " objective:60:(scoped f => argmin f) => objective

@[simp] theorem mem_argmin {Domain Value : Type*} [LE Value]
    (objective : Domain → Value) (candidate : Domain) :
    candidate ∈ Optimization.argmin objective ↔ IsArgmin objective candidate := Iff.rfl

end Optimization
