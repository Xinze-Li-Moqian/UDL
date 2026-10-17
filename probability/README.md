# Probability foundations

This module holds reusable probability interfaces for UDL. It depends on
Mathlib and is independent of neural-network architectures and training losses.
Import `ProbabilityModels` for the full foundation, or import one topic such as
`ProbabilityModels.Expectation` or `ProbabilityModels.Gaussian`.

Each concept has its own file in `ProbabilityModels/`:

| File | Contents |
| --- | --- |
| `Family.lean` | Parameterized probability measures and predictive distributions |
| `Expectation.lean` | Mathlib's integral and expectation notation |
| `Variance.lean` | Real and extended variance, with finite-moment requirements |
| `StandardDeviation.lean` | Square root of variance, nonnegativity, and the square identity |
| `Gaussian.lean` | Mathlib's Gaussian law and density, a `Family` adapter, and standard deviation |
| `ConditionalProbability.lean` | Conditioning a measure on an event |
| `ConditionalDistribution.lean` | Markov kernels and regular conditional distributions |
| `ConditionalExpectation.lean` | Conditional expectation and its standard identities |
| `Density.lean` | Densities relative to a reference measure |
| `Independence.lean` | Independence and conditional independence |

Most files are documented public imports of Mathlib's existing theory.
`StandardDeviation.lean` adds `ProbabilityModels.standardDeviation`, and
`Gaussian.lean` provides `gaussianFamily : Family (ℝ × ℝ≥0) ℝ`. Its two
parameters are the mean and variance. Positive variance is required when
using the Gaussian density; zero variance gives a Dirac mass.

Expectation and variance retain Mathlib's integrability assumptions and
totalization conventions. In particular, use `MemLp X 2 μ` for a finite second
moment under a probability measure. Conditional-expectation and
conditional-distribution identities generally hold almost everywhere.

`ProbabilityModels/Family.lean` provides the first interface for Section 5.1.1:

- `ProbabilityModels.Family Θ Y` is a family `Θ → ProbabilityMeasure Y`.
  Every member is a probability measure, including its normalization proof.
- `family.predictiveDistribution f x φ` is `family (f x φ)`: a model predicts
  distribution parameters `θ = f x φ`, then the family supplies the output
  distribution. Distribution parameters `θ` and model parameters `φ` have
  separate types.

Use Mathlib's measures, distributions, and probability theorems directly.
Project-specific adapters and additional lemmas belong here, so they remain
versioned independently of the downloaded Mathlib dependency. Add further
infrastructure as concrete developments need it.

The `Family` interface specifies a probability distribution at each input.
Connecting it to a kernel additionally needs measurability in the input.
Connecting it to a likelihood needs specified probability masses or densities
relative to a fixed reference measure. These connections are not yet
implemented. Likelihood and negative-log-likelihood constructions will have
their own files in `../loss-functions/` as Chapter 5 develops.
