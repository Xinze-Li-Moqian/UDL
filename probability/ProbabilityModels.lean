module

public import ProbabilityModels.Family
public import ProbabilityModels.Expectation
public import ProbabilityModels.Variance
public import ProbabilityModels.StandardDeviation
public import ProbabilityModels.Gaussian
public import ProbabilityModels.ConditionalProbability
public import ProbabilityModels.ConditionalDistribution
public import ProbabilityModels.ConditionalExpectation
public import ProbabilityModels.Density
public import ProbabilityModels.Independence

/-!
# Probability models

Reusable probability interfaces built on Mathlib, organized by concept.

The modules expose expectation, variance, standard deviation, Gaussian
distributions, conditioning, densities, and independence. `Family` connects
parameterized distributions to predictors, and `Gaussian` adapts Mathlib's
real Gaussian measures to that interface.

Each topic can also be imported individually, for example
`ProbabilityModels.Expectation` or `ProbabilityModels.Gaussian`.
-/
