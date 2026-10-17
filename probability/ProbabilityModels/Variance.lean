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
-/
