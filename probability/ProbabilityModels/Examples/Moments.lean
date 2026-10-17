module

public import ProbabilityModels.Examples.Expectation

/-!
# Variance and standard deviation in examples

The two-outcome variance is `p * (1 - p) * (x - y)^2`. For a Gaussian,
rescaling by `a` multiplies the variance by `a^2` and the standard deviation
by `|a|`; adding an offset changes neither measure of spread.
-/

@[expose] public section

namespace ProbabilityModels.Examples

open MeasureTheory ProbabilityTheory
open scoped ProbabilityModels NNReal

/-- The variance of a distribution with two possible numerical outcomes. -/
theorem variance_two_values (x y : ℝ) (p : unitInterval) :
    Var[id; Ber(x, y, p)] = (p : ℝ) * (1 - (p : ℝ)) * (x - y) ^ 2 := by
  rw [variance_eq_integral measurable_id.aemeasurable]
  calc
    _ = (p : ℝ) * (x - 𝔼[id; Ber(x, y, p)]) ^ 2 +
        (1 - (p : ℝ)) * (y - 𝔼[id; Ber(x, y, p)]) ^ 2 := by
      simpa only [id_eq, smul_eq_mul] using!
        integral_bernoulliMeasure x y p (fun z => (z - 𝔼[id; Ber(x, y, p)]) ^ 2)
    _ = _ := by rw [expectation_two_values]; ring

/-- The variance of an affine transformation of a Gaussian variable. -/
theorem variance_affine_gaussian (a b m : ℝ) (v : ℝ≥0) :
    Var[(fun y => a * y + b); 𝒩(m, v)] = a ^ 2 * (v : ℝ) := by
  have hId : MemLp (id : ℝ → ℝ) 2 (𝒩(m, v)) :=
    memLp_id_gaussianReal 2
  calc
    _ = a ^ 2 * Var[id; 𝒩(m, v)] := by
      simpa only [id_eq] using! variance_affine a b hId
    _ = _ := by rw [variance_id_gaussianReal]

/-- The absolute value matters when the scale is negative. -/
theorem standardDeviation_affine_gaussian (a b m : ℝ) (v : ℝ≥0) :
    SD[(fun y => a * y + b); 𝒩(m, v)] = |a| * Real.sqrt (v : ℝ) := by
  have hId : MemLp (id : ℝ → ℝ) 2 (𝒩(m, v)) :=
    memLp_id_gaussianReal 2
  calc
    _ = |a| * SD[id; 𝒩(m, v)] := by
      simpa only [id_eq] using! standardDeviation_affine a b hId
    _ = _ := by rw [standardDeviation_gaussianReal]

end ProbabilityModels.Examples
