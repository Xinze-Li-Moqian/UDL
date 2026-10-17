module

/-!
# Argmin

A general optimality condition for an objective function.
This module is independent of supervised learning and any particular model.
-/

@[expose] public section

namespace Optimization

/-- `candidate` belongs to the argmin of `objective`: its value is no greater
than the value at any other point. This specifies global optimality, without
assuming that a minimizer exists or is unique, or supplying a solver. -/
def IsArgmin {Domain Value : Type} [LE Value]
    (objective : Domain → Value) (candidate : Domain) : Prop :=
  ∀ other : Domain, objective candidate ≤ objective other

end Optimization
