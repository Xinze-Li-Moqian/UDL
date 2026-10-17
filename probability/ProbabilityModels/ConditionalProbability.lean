module

public import Mathlib.Probability.ConditionalProbability

/-!
# Probability conditioned on an event

Use `ProbabilityTheory.cond μ B`, or `μ[|B]` in the `ProbabilityTheory` scope,
for the measure obtained by conditioning on an event `B`. Its evaluation on
an event `A` is written `μ[A | B]`.

For a probability measure and a measurable event of positive probability,
this is the normalized restriction to `B` and gives the usual ratio
`μ (B ∩ A) / μ B`. Mathlib supplies the corresponding Bayes identities.

Conditioning on a random variable, including continuous observations whose
singletons can have probability zero, is covered by `ConditionalDistribution`.
-/
