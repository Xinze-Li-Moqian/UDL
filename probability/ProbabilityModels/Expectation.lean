module

public import Mathlib.MeasureTheory.Integral.Bochner.Basic
public import Mathlib.Probability.Notation

/-!
# Expectation

Use Mathlib's `MeasureTheory.integral` for expectation. With
`open scoped ProbabilityTheory`, `μ[X]` denotes `∫ ω, X ω ∂μ`.
For a bundled probability measure `P`, pass `(P : MeasureTheory.Measure Ω)`.

`MeasureTheory.Integrable X μ` expresses the existence of a finite expectation;
linearity is supplied by `integral_add` and `integral_const_mul`. Mathlib's
Bochner integral is total and takes the value zero on nonintegrable functions,
so applications must retain the integrability hypotheses of these results.
-/
