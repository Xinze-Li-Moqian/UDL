module

public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.Probability.Notation

/-!
# Expectation

For a real random variable `X : Ω → ℝ` under a probability measure `P`, its
expectation is the integral `∫ ω, X ω ∂P`. The scoped notation `𝔼[X; P]`
expands directly to Mathlib's integral; use `open scoped ProbabilityModels`.
For a bundled probability measure, write `(P : MeasureTheory.Measure Ω)`
when needed for type inference.

`MeasureTheory.Integrable X P` supplies almost-everywhere measurability and
integrability of the absolute value, so that the expectation is finite.
Mathlib's Bochner integral is total and takes the value zero on nonintegrable
functions. The linearity and order results below keep the required
integrability hypotheses explicit.

The main formulas are expectation of a constant, linearity, the affine
transformation rule, nonnegativity, and monotonicity. The constant and affine
rules require `IsProbabilityMeasure P`; the other integral identities also
hold for general measures. All proofs reuse Mathlib's integral theory.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory

/-- Expectation under an explicit measure, defined by Mathlib's integral. -/
scoped notation "𝔼[" X "; " P "]" => MeasureTheory.integral P X

variable {Ω : Type*} {mΩ : MeasurableSpace Ω}
variable {P : Measure Ω} {X Y : Ω → ℝ}

/-- The expectation of a constant is that constant under a probability measure. -/
@[simp] theorem expectation_const [IsProbabilityMeasure P] (c : ℝ) :
    𝔼[(fun _ : Ω => c); P] = c := by
  simp

/-- Expectation preserves addition for integrable random variables. -/
theorem expectation_add (hX : Integrable X P) (hY : Integrable Y P) :
    𝔼[(fun ω => X ω + Y ω); P] = 𝔼[X; P] + 𝔼[Y; P] :=
  integral_add hX hY

/-- A constant scalar can be taken outside the expectation. -/
theorem expectation_const_mul (a : ℝ) (X : Ω → ℝ) :
    𝔼[(fun ω => a * X ω); P] = a * 𝔼[X; P] :=
  integral_const_mul a X

/-- Linearity of expectation. Independence of `X` and `Y` is not required. -/
theorem expectation_linearCombination (a b : ℝ)
    (hX : Integrable X P) (hY : Integrable Y P) :
    𝔼[(fun ω => a * X ω + b * Y ω); P] = a * 𝔼[X; P] + b * 𝔼[Y; P] := by
  rw [expectation_add (hX.const_mul a) (hY.const_mul b),
    expectation_const_mul, expectation_const_mul]

/-- An affine transformation sends the mean to the same affine transformation
of the mean, provided the original random variable is integrable. -/
theorem expectation_affine [IsProbabilityMeasure P] (a b : ℝ)
    (hX : Integrable X P) :
    𝔼[(fun ω => a * X ω + b); P] = a * 𝔼[X; P] + b := by
  rw [expectation_add (hX.const_mul a) (integrable_const b),
    expectation_const_mul, expectation_const]

/-- An almost-everywhere nonnegative random variable has nonnegative expectation. -/
theorem expectation_nonneg (hX : 0 ≤ᵐ[P] X) : 0 ≤ 𝔼[X; P] :=
  integral_nonneg_of_ae hX

/-- Expectation preserves an almost-everywhere inequality between integrable
random variables. -/
theorem expectation_mono (hX : Integrable X P) (hY : Integrable Y P)
    (hXY : X ≤ᵐ[P] Y) : 𝔼[X; P] ≤ 𝔼[Y; P] :=
  integral_mono_ae hX hY hXY

end ProbabilityModels
