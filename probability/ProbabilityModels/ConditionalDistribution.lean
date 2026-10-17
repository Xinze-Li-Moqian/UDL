module

public import Mathlib.Probability.Kernel.CondDistrib

/-!
# Conditional distributions

Use `ProbabilityTheory.Kernel X Y` with `IsMarkovKernel` for a measurable
family of probability measures on `Y`, indexed by inputs in `X`.

`ProbabilityTheory.condDistrib Y X μ` constructs a regular conditional law
from random variables and an underlying measure, under Mathlib's standard
Borel and finiteness hypotheses. Its defining relation to conditional
expectation holds almost everywhere.

A specified predictive kernel is a model for a conditional law. The pointwise
interface in `ProbabilityModels.Family` additionally needs measurability to
produce such a kernel; an identification with the conditional law of a given
joint distribution requires a separate argument.
-/
