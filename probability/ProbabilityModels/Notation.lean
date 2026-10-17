module

public import ProbabilityModels.Expectation
public import ProbabilityModels.Variance
public import ProbabilityModels.StandardDeviation
public import ProbabilityModels.Gaussian
public import ProbabilityModels.ConditionalProbability
public import ProbabilityModels.ConditionalExpectation

/-!
# Probability notation

Import this module and use
`open scoped ProbabilityTheory ProbabilityModels`.

Mathlib provides `P[X]`, `Var[X; P]`, `eVar[X; P]`, and the event-conditional
probability `P[A | B]`. This module collects notation with an explicit measure:

- `𝔼[X; P]`: expectation of `X` under `P`.
- `𝔼[X | m; P]`: conditional expectation given a sub-sigma-algebra `m`.
- `SD[X; P]`: standard deviation under `P`.
- `𝒩(m, v)`: the real Gaussian measure with mean `m` and variance `v`.

For example, the mean of a Gaussian can be written `𝔼[id; 𝒩(m, v)] = m`,
and its variance as `Var[id; 𝒩(m, v)] = (v : ℝ)`. Here `m : ℝ` and
`v : ℝ≥0` (the latter notation uses `open scoped NNReal`).

Use explicit measures in model formulas: Mathlib's `𝔼[X]` defaults to the
ambient `volume` measure, which on `ℝ` is Lebesgue measure. For a bundled
probability measure `P`, write `(P : MeasureTheory.Measure Ω)` when its domain
cannot be inferred, for example when integrating `id`.
The notation leaves all integrability, finite-moment, and conditioning
assumptions to the underlying definitions and theorems.

The expectation notation is declared in `Expectation`, so its basic formulas
can use it without importing Gaussian distributions or conditioning. This
module collects it together with the additional notations below.
-/

public section

namespace ProbabilityModels

/-- Conditional expectation under an explicit measure, given a sub-sigma-algebra. -/
scoped notation "𝔼[" X " | " m "; " P "]" => MeasureTheory.condExp m P X

/-- Standard deviation under an explicit measure. -/
scoped notation "SD[" X "; " P "]" => ProbabilityModels.standardDeviation X P

/-- The Gaussian measure with mean `m` and variance `v : ℝ≥0`. -/
scoped notation "𝒩(" m ", " v ")" => ProbabilityTheory.gaussianReal m v

end ProbabilityModels
