module

public import FittingModels.Gradient
public import Mathlib.Analysis.Calculus.FDeriv.Prod
public import Mathlib.Analysis.Calculus.FDeriv.Pow
public import Mathlib.Tactic.Linarith

/-!
# Local minima and a saddle point

Understanding Deep Learning, Section 6.1.3.

Use Mathlib's existing predicates:

* `IsMinOn L Set.univ φ`: L(φ) ≤ L(ψ) for every parameter ψ.
* `IsLocalMin L φ`: the same inequality holds in some neighborhood of φ.

The example S(a, b) = a² − b² has zero derivative at the origin, yet every
neighborhood contains both higher and lower values. It is a saddle point.
This is a simple objective illustrating the concept, not the Gabor loss.
-/

@[expose] public section

namespace FittingModels.Examples.SaddlePoint

open scoped Topology

/-- Every global minimum is a local minimum. -/
theorem localMin_of_globalMin {P : Type*} [TopologicalSpace P]
    {L : P → ℝ} {φ : P} (h : IsMinOn L Set.univ φ) : IsLocalMin L φ :=
  h.isLocalMin (by simp)

/-- S(a, b) = a² − b². -/
def objective (p : ℝ × ℝ) : ℝ := p.1 ^ 2 - p.2 ^ 2

/-- The full derivative is zero, not just one coordinate derivative. -/
theorem stationary : HasFDerivAt (𝕜 := ℝ) objective 0 (0, 0) := by
  have h := ((hasFDerivAt_fst (𝕜 := ℝ) (p := ((0 : ℝ), (0 : ℝ)))).pow 2).sub
    ((hasFDerivAt_snd (𝕜 := ℝ) (p := ((0 : ℝ), (0 : ℝ)))).pow 2)
  convert! h using 1
  simp

/-- For every positive radius r, there are nearby higher and lower values.

    S(r/2, 0) > S(0, 0) > S(0, r/2).
-/
theorem nearby_values {r : ℝ} (hr : 0 < r) :
    ∃ up down : ℝ × ℝ,
      dist up (0, 0) < r ∧ dist down (0, 0) < r ∧
      objective down < objective (0, 0) ∧ objective (0, 0) < objective up := by
  refine ⟨(r / 2, 0), (0, r / 2), ?_, ?_, ?_, ?_⟩
  · simpa [Prod.dist_eq, Real.dist_eq, abs_of_pos hr] using And.intro (half_lt_self hr) hr
  · simpa [Prod.dist_eq, Real.dist_eq, abs_of_pos hr] using And.intro hr (half_lt_self hr)
  · dsimp [objective]
    nlinarith [sq_pos_of_pos (half_pos hr)]
  · dsimp [objective]
    nlinarith [sq_pos_of_pos (half_pos hr)]

/-- A zero derivative does not imply a local minimum. -/
theorem not_localMin : ¬ IsLocalMin objective (0, 0) := by
  intro h
  obtain ⟨r, hr, hball⟩ := Metric.mem_nhds_iff.mp h
  obtain ⟨up, down, _, hd, hlt, _⟩ := nearby_values hr
  exact (not_le_of_gt hlt) (hball hd)

/-- The origin is not a local maximum either. -/
theorem not_localMax : ¬ IsLocalMax objective (0, 0) := by
  intro h
  obtain ⟨r, hr, hball⟩ := Metric.mem_nhds_iff.mp h
  obtain ⟨up, down, hu, _, _, hgt⟩ := nearby_values hr
  exact (not_le_of_gt hgt) (hball hu)

end FittingModels.Examples.SaddlePoint
