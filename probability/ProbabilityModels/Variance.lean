module

public import ProbabilityModels.Expectation
public import Mathlib.Probability.Moments.Variance

/-!
# Variance

Use `ProbabilityTheory.variance X μ`, or `Var[X; μ]` in the
`ProbabilityTheory` scope. `ProbabilityTheory.evariance X μ` retains infinite
values in `ℝ≥0∞`; the real-valued `variance` is its `toReal` conversion.

For probability measures, `MeasureTheory.MemLp X 2 μ` supplies a finite second
moment. Under this assumption `evariance` is finite, and the usual variance
identities apply with the hypotheses given in Mathlib. Without it, an infinite
`evariance` becomes zero in the real-valued `variance`.

The formulas below use the shared expectation notation and retain finite
second-moment hypotheses. Mathlib also provides `variance_nonneg`,
`variance_const_mul`, and `evariance_lt_top` directly. Independent-sum rules
are collected in `ProbabilityModels.Independence`.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}
variable {P : Measure Ω} {X : Ω → ℝ}

/-- Variance is the expected squared deviation from the mean. -/
theorem variance_eq_expectation_centered_sq (hX : MemLp X 2 P) :
    Var[X; P] = 𝔼[(fun ω => (X ω - 𝔼[X; P]) ^ 2); P] :=
  variance_eq_integral hX.aemeasurable

/-- Variance is the second moment minus the square of the mean. -/
theorem variance_eq_secondMoment_sub_sq [IsProbabilityMeasure P] (hX : MemLp X 2 P) :
    Var[X; P] = 𝔼[(fun ω => X ω ^ 2); P] - 𝔼[X; P] ^ 2 := by
  simpa only [Pi.pow_apply] using! variance_eq_sub hX

/-- A constant random variable has variance zero. -/
@[simp] theorem variance_const [IsProbabilityMeasure P] (c : ℝ) :
    Var[(fun _ : Ω => c); P] = 0 := by
  rw [variance_eq_integral measurable_const.aemeasurable]
  simp

/-- Shifting a random variable does not change its variance. -/
theorem variance_shift [IsProbabilityMeasure P] (b : ℝ) (hX : MemLp X 2 P) :
    Var[(fun ω => X ω + b); P] = Var[X; P] :=
  variance_add_const hX.aestronglyMeasurable b

/-- An affine transformation scales variance by the square of its slope. -/
theorem variance_affine [IsProbabilityMeasure P] (a b : ℝ) (hX : MemLp X 2 P) :
    Var[(fun ω => a * X ω + b); P] = a ^ 2 * Var[X; P] := by
  rw [variance_shift b (hX.const_mul a), variance_const_mul]

/-- With a finite second moment, variance is zero exactly for variables
almost surely equal to their mean. -/
theorem variance_eq_zero_iff [IsProbabilityMeasure P] (hX : MemLp X 2 P) :
    Var[X; P] = 0 ↔ X =ᵐ[P] (fun _ => 𝔼[X; P]) := by
  constructor
  · exact ae_eq_integral_of_variance_eq_zero hX
  · intro h
    rw [variance_congr h, variance_const]

end ProbabilityModels
