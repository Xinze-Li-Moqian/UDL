module

public import ProbabilityModels.Family
public import ProbabilityModels.Expectation
public import ProbabilityModels.Variance
public import ProbabilityModels.StandardDeviation
public import ProbabilityModels.Gaussian
public import ProbabilityModels.Bernoulli
public import ProbabilityModels.ConditionalProbability
public import ProbabilityModels.ConditionalDistribution
public import ProbabilityModels.ConditionalExpectation
public import ProbabilityModels.Density
public import ProbabilityModels.Independence
public import ProbabilityModels.Notation
public import ProbabilityModels.Examples.Expectation
public import ProbabilityModels.Examples.Conditioning
public import ProbabilityModels.Examples.Density
public import ProbabilityModels.Examples.Moments
public import ProbabilityModels.Examples.Independence

/-!
# Probability models

Reusable probability interfaces built on Mathlib, organized by concept.

The modules expose expectation, variance, standard deviation, Gaussian and Bernoulli
distributions, conditioning, densities, and independence. `Family` connects
parameterized distributions to predictors, and `Gaussian` adapts Mathlib's
real Gaussian measures to that interface. `Bernoulli` supplies binary labels
with a certified mass function relative to counting measure.

Each topic can also be imported individually, for example
`ProbabilityModels.Expectation` or `ProbabilityModels.Gaussian`.
`ProbabilityModels.Notation` provides a shared notation entry point, with
explicit measures for expectation and standard deviation.
`ProbabilityModels.Examples.Expectation` demonstrates discrete and Gaussian means.
`ProbabilityModels.Examples.Conditioning` builds a Gaussian predictive kernel.
`ProbabilityModels.Examples.Density` supplies its normalized, pointwise density.
`ProbabilityModels.Examples.Moments` computes discrete and Gaussian measures of spread.
`ProbabilityModels.Examples.Independence` derives the variance of a sample mean.
-/
