module

public import ProbabilityModels.Notation
public import Mathlib.Probability.Distributions.Bernoulli

/-!
# Reading expectation formulas

These examples connect the expectation notation to a discrete weighted mean
and a continuous Gaussian mean, then apply the affine expectation rule.
The identity function `id` is the random variable when the measure already
lives on the space of possible numerical outcomes.
-/

@[expose] public section

namespace ProbabilityModels.Examples

open MeasureTheory ProbabilityTheory
open scoped ProbabilityModels NNReal

/-- A distribution that chooses `x` with probability `p` and `y` with
probability `1 - p` has the corresponding weighted mean. -/
theorem expectation_two_values (x y : ℝ) (p : unitInterval) :
    𝔼[id; Ber(x, y, p)] = (p : ℝ) * x + (1 - (p : ℝ)) * y := by
  simpa only [id_eq, smul_eq_mul] using! integral_bernoulliMeasure x y p (id : ℝ → ℝ)

/-- The mean of a Gaussian is its parameter `m`. -/
theorem expectation_gaussian (m : ℝ) (v : ℝ≥0) :
    𝔼[id; 𝒩(m, v)] = m :=
  integral_id_gaussianReal

/-- Rescaling and shifting a Gaussian variable rescales and shifts its mean. -/
theorem expectation_affine_gaussian (a b m : ℝ) (v : ℝ≥0) :
    𝔼[(fun y => a * y + b); 𝒩(m, v)] = a * m + b := by
  have hId : Integrable (id : ℝ → ℝ) (𝒩(m, v)) :=
    (memLp_id_gaussianReal (μ := m) (v := v) 1).integrable (by simp)
  calc
    _ = a * 𝔼[id; 𝒩(m, v)] + b := by
      simpa only [id_eq] using! expectation_affine a b hId
    _ = a * m + b := by rw [expectation_gaussian]

end ProbabilityModels.Examples
