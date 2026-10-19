module

public import FittingModels.Examples.Gabor
public import Mathlib.Topology.Order.Compact
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.NormNum
public import Mathlib.Tactic.FunProp

/-!
# A Gabor local minimum that is not global

Use the single training pair (x, y) = (0, 1). The existing squared loss becomes

    S(a, b) = (1 − sin(a) exp(−a²/32))².

On the second positive lobe [2π, 3π], continuity gives a minimum. Both
endpoints have loss 1, while an interior point has smaller loss, so this
minimum is local in the full parameter space. The Gaussian envelope makes
the first lobe higher: phase π/2 gives a strictly smaller loss.

This is an existence proof for explicit toy data, not the book's 28 samples.
Scale b is unused at x = 0, so the local minimum is not strict.
-/

@[expose] public section

namespace FittingModels.Examples.Gabor.LocalMinimum

open SupervisedLearning
open scoped Topology

/-- One observation with target 1, rather than the earlier target 0. -/
def data : TrainingDataset ℝ 1 1 1 where
  x _ := #v[0]
  y _ := #v[1]

/-- The actual model loss, with phase and scale as two real coordinates. -/
noncomputable def loss (p : ℝ × ℝ) : ℝ :=
  leastSquaresLoss data ⟨p.1, p.2⟩

noncomputable def profile (a : ℝ) : ℝ := Real.sin a * Real.exp (-(a ^ 2) / 32)

noncomputable def phaseLoss (a : ℝ) : ℝ := (1 - profile a) ^ 2

lemma loss_eq (p : ℝ × ℝ) : loss p = phaseLoss p.1 := by
  simp [loss, leastSquaresLoss, LossFunctions.UnivariateRegression.leastSquaresLoss,
    LossFunctions.empiricalLoss, prediction,
    SupervisedLearning.Discriminative.Gabor.gaborModel, data, phaseLoss, profile]

lemma phaseLoss_continuous : Continuous phaseLoss := by
  unfold phaseLoss profile
  fun_prop

/-- The Gaussian envelope bounds the height of every oscillation. -/
lemma profile_le_envelope (a : ℝ) : profile a ≤ Real.exp (-(a ^ 2) / 32) := by
  exact mul_le_of_le_one_left (le_of_lt (Real.exp_pos _)) (Real.sin_le_one a)

lemma profile_lt_one {a : ℝ} (ha : 0 < a) : profile a < 1 := by
  apply lt_of_le_of_lt (profile_le_envelope a)
  rw [← Real.exp_zero]
  apply Real.exp_lt_exp.mpr
  nlinarith [sq_pos_of_pos ha]

/-- In the second lobe, even the envelope lies below the first-lobe test point. -/
lemma profile_lt_first {a : ℝ} (ha : 2 * Real.pi ≤ a) :
    profile a < profile (Real.pi / 2) := by
  have hp := Real.pi_pos
  have hsq : (Real.pi / 2) ^ 2 < a ^ 2 := by
    nlinarith [sq_nonneg (a - 2 * Real.pi), sq_pos_of_pos hp]
  calc
    profile a ≤ Real.exp (-(a ^ 2) / 32) := profile_le_envelope a
    _ < Real.exp (-((Real.pi / 2) ^ 2) / 32) := Real.exp_lt_exp.mpr (by linarith)
    _ = profile (Real.pi / 2) := by simp [profile]

lemma first_has_lower_loss {a : ℝ} (ha : 2 * Real.pi ≤ a) :
    phaseLoss (Real.pi / 2) < phaseLoss a := by
  have hlt := profile_lt_first ha
  have hfirst := profile_lt_one (half_pos Real.pi_pos)
  dsimp [phaseLoss]
  nlinarith [sq_nonneg (profile (Real.pi / 2) - profile a)]

/-- Both ends of the lobe have loss 1. -/
lemma endpoint_losses : phaseLoss (2 * Real.pi) = 1 ∧ phaseLoss (3 * Real.pi) = 1 := by
  have hthree := Real.sin_nat_mul_pi 3
  norm_num at hthree
  simp [phaseLoss, profile, Real.sin_two_pi, hthree]

/-- An interior point has positive prediction below 1, hence loss below 1. -/
lemma interior_loss_lt_one : phaseLoss (Real.pi / 2 + 2 * Real.pi) < 1 := by
  have hs : Real.sin (Real.pi / 2 + 2 * Real.pi) = 1 := by
    simp [Real.sin_add]
  have hpos : 0 < profile (Real.pi / 2 + 2 * Real.pi) := by
    simp only [profile, hs, one_mul]
    exact Real.exp_pos _
  have hlt := profile_lt_one (a := Real.pi / 2 + 2 * Real.pi) (by linarith [Real.pi_pos])
  dsimp [phaseLoss]
  nlinarith

/-- There is a local minimum in the second lobe, but phase π/2 is better. -/
lemma exists_bad_phase :
    ∃ a ∈ Set.Ioo (2 * Real.pi) (3 * Real.pi),
      IsLocalMin phaseLoss a ∧ phaseLoss (Real.pi / 2) < phaseLoss a := by
  have hp := Real.pi_pos
  obtain ⟨a, ha, hmin⟩ := isCompact_Icc.exists_isMinOn
    (s := Set.Icc (2 * Real.pi) (3 * Real.pi))
    ⟨2 * Real.pi, by constructor <;> linarith⟩ phaseLoss_continuous.continuousOn
  have hsmall : phaseLoss a < 1 := lt_of_le_of_lt
    (hmin (by constructor <;> linarith : Real.pi / 2 + 2 * Real.pi ∈
      Set.Icc (2 * Real.pi) (3 * Real.pi))) interior_loss_lt_one
  have hleft : 2 * Real.pi < a := lt_of_le_of_ne ha.1 (by
    intro heq
    rw [← heq, endpoint_losses.1] at hsmall
    exact (lt_irrefl _) hsmall)
  have hright : a < 3 * Real.pi := lt_of_le_of_ne ha.2 (by
    intro heq
    rw [heq, endpoint_losses.2] at hsmall
    exact (lt_irrefl _) hsmall)
  exact ⟨a, ⟨hleft, hright⟩, hmin.isLocalMin (Icc_mem_nhds hleft hright),
    first_has_lower_loss ha.1⟩

/-- A local minimum of the full two-parameter loss, with a strictly better point.
Both parameter choices have positive scale, equal to 1. -/
theorem exists_localMin_not_global :
    ∃ p : ℝ × ℝ, 0 < p.2 ∧ IsLocalMin loss p ∧ loss (Real.pi / 2, 1) < loss p := by
  obtain ⟨a, _, hlocal, hbetter⟩ := exists_bad_phase
  refine ⟨(a, 1), by norm_num, ?_, ?_⟩
  · have h := hlocal.comp_continuous (g := fun p : ℝ × ℝ => p.1)
      (b := (a, 1)) continuous_fst.continuousAt
    simpa only [Function.comp_def, ← loss_eq] using h
  · simpa only [loss_eq] using hbetter

end FittingModels.Examples.Gabor.LocalMinimum
