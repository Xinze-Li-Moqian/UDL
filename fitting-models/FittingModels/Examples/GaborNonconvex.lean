module

public import FittingModels.Examples.Gabor
public import Mathlib.Analysis.Convex.Function
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.Ring

/-!
# A nonconvex Gabor loss

Use a single training pair (x, y) = (0, 0). Fixing scale to 1 gives:

    S(a, 1) = (sin(a) exp(−a²/32))².

At phases 0 and π the loss is zero, while at their midpoint π/2 it is
positive. This violates the convexity inequality. The proof uses Mathlib's
`ConvexOn` and the existing model and squared-error loss constructors.
-/

@[expose] public section

namespace FittingModels.Examples.Gabor

open SupervisedLearning

/-- The explicit dataset for this counterexample. -/
def zeroSample : TrainingDataset ℝ 1 1 1 where
  x _ := #v[0]
  y _ := #v[0]

/-- Express the model's parameters as coordinates so Mathlib can form midpoints. -/
noncomputable def lossOnCoordinates (p : ℝ × ℝ) : ℝ :=
  leastSquaresLoss zeroSample ⟨p.1, p.2⟩

/-- Only the phase varies along this line; scale stays positive and equal to 1. -/
noncomputable def phaseLoss (a : ℝ) : ℝ := lossOnCoordinates (a, 1)

theorem phaseLoss_eq (a : ℝ) :
    phaseLoss a = (Real.sin a * Real.exp (-(a ^ 2) / 32)) ^ 2 := by
  simp [phaseLoss, lossOnCoordinates, leastSquaresLoss,
    LossFunctions.UnivariateRegression.leastSquaresLoss, LossFunctions.empiricalLoss,
    prediction, SupervisedLearning.Discriminative.Gabor.gaborModel, zeroSample]

/-- First endpoint: phase 0 fits the sample exactly. -/
theorem phaseLoss_zero : phaseLoss 0 = 0 := by
  rw [phaseLoss_eq]
  simp

/-- Second endpoint: phase π also fits the sample exactly. -/
theorem phaseLoss_pi : phaseLoss Real.pi = 0 := by
  rw [phaseLoss_eq]
  simp

/-- The midpoint has strictly positive loss, although both endpoints have loss zero. -/
theorem phaseLoss_midpoint_pos : 0 < phaseLoss (Real.pi / 2) := by
  rw [phaseLoss_eq, Real.sin_pi_div_two, one_mul]
  exact sq_pos_of_pos (Real.exp_pos _)

/-- The squared-error loss is nonconvex on the book's positive-scale domain.

    S((0, 1) + (π, 1) divided by 2) > (S(0, 1) + S(π, 1)) / 2.
-/
theorem leastSquaresLoss_not_convex :
    ¬ ConvexOn ℝ {p : ℝ × ℝ | 0 < p.2} lossOnCoordinates := by
  intro hconvex
  have hineq := hconvex.2
  have hmid := hineq
    (x := ((0 : ℝ), (1 : ℝ))) (by norm_num)
    (y := (Real.pi, (1 : ℝ))) (by norm_num)
    (a := (1 / 2 : ℝ)) (b := (1 / 2 : ℝ))
    (by norm_num) (by norm_num) (by norm_num)
  have hpoint :
      (1 / 2 : ℝ) • ((0 : ℝ), (1 : ℝ)) +
        (1 / 2 : ℝ) • (Real.pi, (1 : ℝ)) = (Real.pi / 2, 1) := by
    ext <;> simp [smul_eq_mul] <;> ring
  rw [hpoint] at hmid
  change phaseLoss (Real.pi / 2) ≤
    (1 / 2 : ℝ) * phaseLoss 0 + (1 / 2 : ℝ) * phaseLoss Real.pi at hmid
  simp only [phaseLoss_zero, phaseLoss_pi, mul_zero, add_zero] at hmid
  exact (not_le_of_gt phaseLoss_midpoint_pos) hmid

end FittingModels.Examples.Gabor
