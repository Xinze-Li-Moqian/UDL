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
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

/-- The square root of the real-valued variance. For finite-variance semantics
under a probability measure, use this with `MemLp X 2 μ`. -/
noncomputable def standardDeviation {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : ℝ :=
  Real.sqrt (variance X μ)

theorem standardDeviation_nonneg {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : 0 ≤ standardDeviation X μ :=
  Real.sqrt_nonneg _

/-- Squaring the standard deviation recovers the real-valued variance. -/
@[simp] theorem standardDeviation_sq {Ω : Type*} {mΩ : MeasurableSpace Ω}
    (X : Ω → ℝ) (μ : Measure Ω) : standardDeviation X μ ^ 2 = variance X μ :=
  Real.sq_sqrt (variance_nonneg X μ)

end ProbabilityModels
