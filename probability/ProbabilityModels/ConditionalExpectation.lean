module

public import ProbabilityModels.Expectation
public import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
public import Mathlib.Probability.ConditionalExpectation

/-!
# Conditional expectation

Use `MeasureTheory.condExp m μ X`, written `μ[X | m]`, to condition on the
information in a sub-sigma-algebra `m`. Conditioning on a random variable `Z`
uses the sigma-algebra `MeasurableSpace.comap Z` of its codomain sigma-algebra.

Mathlib supplies integrability, measurability, linearity, the tower property
(`condExp_condExp_of_le`), and the identity under independence
(`condExp_indep_eq`). These identities carry their integrability and
sigma-algebra hypotheses and generally assert almost-everywhere equality.
-/
