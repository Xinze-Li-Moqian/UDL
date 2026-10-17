module

public import ProbabilityModels.Variance
public import Mathlib.Analysis.Real.Sqrt

/-!
# Standard deviation

Standard deviation is the nonnegative square root of Mathlib's real-valued
variance. It inherits the totalization convention of `variance`: an infinite
extended variance yields zero. For probabilistic applications under a
probability measure `μ`, assume `MeasureTheory.MemLp X 2 μ` to ensure a finite
second moment.

The notation `SD[X; P]` is scoped to `ProbabilityModels`. The main affine rule
is `SD[aX + b; P] = |a| * SD[X; P]`: shifting does not change the standard
deviation, and a negative scale contributes its absolute value.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

/-- The square root of the real-valued variance. For finite-variance semantics
under a probability measure, use this with `MemLp X 2 μ`. -/
noncomputable def standardDeviation {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : ℝ :=
  Real.sqrt (variance X μ)

/-- Standard deviation under an explicit measure. -/
scoped notation "SD[" X "; " P "]" => ProbabilityModels.standardDeviation X P

theorem standardDeviation_nonneg {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : 0 ≤ standardDeviation X μ :=
  Real.sqrt_nonneg _

/-- Squaring the standard deviation recovers the real-valued variance. -/
@[simp] theorem standardDeviation_sq {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : standardDeviation X μ ^ 2 = variance X μ :=
  Real.sq_sqrt (variance_nonneg X μ)

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}
variable {P : Measure Ω} {X Y : Ω → ℝ}

/-- Almost-everywhere equal variables have the same standard deviation. -/
theorem standardDeviation_congr (h : X =ᵐ[P] Y) : SD[X; P] = SD[Y; P] := by
  rw [standardDeviation, standardDeviation, variance_congr h]

/-- A constant random variable has standard deviation zero. -/
@[simp] theorem standardDeviation_const [IsProbabilityMeasure P] (c : ℝ) :
    SD[(fun _ : Ω => c); P] = 0 := by
  simp [standardDeviation]

/-- Scaling multiplies the standard deviation by the absolute value of the scale. -/
theorem standardDeviation_const_mul (a : ℝ) (X : Ω → ℝ) :
    SD[(fun ω => a * X ω); P] = |a| * SD[X; P] := by
  rw [standardDeviation, variance_const_mul, Real.sqrt_mul (sq_nonneg a),
    Real.sqrt_sq_eq_abs]
  rfl

/-- Shifting does not change the standard deviation. -/
theorem standardDeviation_shift [IsProbabilityMeasure P] (b : ℝ) (hX : MemLp X 2 P) :
    SD[(fun ω => X ω + b); P] = SD[X; P] := by
  rw [standardDeviation, variance_shift b hX]
  rfl

/-- The standard deviation of an affine transformation. -/
theorem standardDeviation_affine [IsProbabilityMeasure P] (a b : ℝ) (hX : MemLp X 2 P) :
    SD[(fun ω => a * X ω + b); P] = |a| * SD[X; P] := by
  rw [standardDeviation_shift b (hX.const_mul a), standardDeviation_const_mul]

/-- With a finite second moment, zero standard deviation means the variable
is almost surely equal to its mean. -/
theorem standardDeviation_eq_zero_iff [IsProbabilityMeasure P] (hX : MemLp X 2 P) :
    SD[X; P] = 0 ↔ X =ᵐ[P] (fun _ => 𝔼[X; P]) := by
  rw [standardDeviation, Real.sqrt_eq_zero (variance_nonneg X P), variance_eq_zero_iff hX]

end ProbabilityModels
