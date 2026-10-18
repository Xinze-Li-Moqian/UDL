module

public import ProbabilityModels.Density
public import Mathlib.Probability.Distributions.Bernoulli
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Bernoulli distributions on binary labels

Labels are `Fin 2`, so their values are exactly 0 and 1. The parameter
`p : unitInterval` is the probability of label 1. This module packages
Mathlib's Bernoulli measure as a family with a density relative to counting
measure; here density values are point probabilities.

    P(Y = 1 | p) = p,    P(Y = 0 | p) = 1 − p.

The family includes p = 0 and p = 1. Strict positivity of both masses requires
0 < p < 1, as supplied by sigmoid-based models when taking logarithms.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

/-- A Bernoulli family with probability `p` assigned to label 1. -/
noncomputable def bernoulliFamily : Family unitInterval (Fin 2) :=
  fun p => (bernoulliMeasure (1 : Fin 2) 0 p).toProbabilityMeasure

@[simp] theorem coe_bernoulliFamily (p : unitInterval) :
    (bernoulliFamily p : Measure (Fin 2)) = bernoulliMeasure 1 0 p := rfl

/-- Equation (5.16): the two probability masses, certified against the family.

    p(y | λ) = λ when y = 1, and 1 − λ when y = 0.
-/
noncomputable def bernoulliMass : bernoulliFamily.Density Measure.count where
  toFun p y := if y = 1 then (p : ℝ) else 1 - (p : ℝ)
  nonneg p y := by
    split_ifs
    · exact p.2.1
    · exact sub_nonneg.mpr p.2.2
  measurable p := measurable_of_finite _
  withDensity_eq p := by
    apply Measure.ext_of_singleton
    intro y
    fin_cases y <;>
      simp [withDensity_apply, bernoulliMeasure_def]
    · rw [← unitInterval.coe_symm_eq p,
        ← unitInterval.coe_toNNReal (unitInterval.symm p), ENNReal.ofReal_coe_nnreal]
    · rw [← unitInterval.coe_toNNReal p, ENNReal.ofReal_coe_nnreal]

@[simp] theorem bernoulliMass_one (p : unitInterval) :
    bernoulliMass p 1 = (p : ℝ) := rfl

@[simp] theorem bernoulliMass_zero (p : unitInterval) :
    bernoulliMass p 0 = 1 - (p : ℝ) := rfl

/-- Equation (5.17): combine the two cases into one expression.

    p(y | λ) = (1 − λ)^(1 − y) λ^y,    y ∈ {0, 1}.
-/
theorem bernoulliMass_eq (p : unitInterval) (y : Fin 2) :
    bernoulliMass p y = (1 - (p : ℝ)) ^ (1 - y.val) * (p : ℝ) ^ y.val := by
  fin_cases y <;> simp

/-- Interior parameters give strictly positive probability to both labels. -/
theorem bernoulliMass_pos (p : unitInterval)
    (hp : 0 < (p : ℝ)) (hp1 : (p : ℝ) < 1) (y : Fin 2) :
    0 < bernoulliMass p y := by
  fin_cases y
  · simpa using sub_pos.mpr hp1
  · simpa using hp

/-- The negative log mass is the binary cross-entropy term.

    −log p(y | λ) = −(1 − y) log(1 − λ) − y log λ.

At boundary parameters this is an identity of Lean's totalized real logarithm;
likelihood applications use interior parameters to avoid zero probabilities.
-/
theorem neg_log_bernoulliMass (p : unitInterval) (y : Fin 2) :
    -Real.log (bernoulliMass p y) =
      -(1 - (y.val : ℝ)) * Real.log (1 - (p : ℝ)) -
        (y.val : ℝ) * Real.log (p : ℝ) := by
  fin_cases y <;> simp

end ProbabilityModels
