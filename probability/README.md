# Probability foundations

This module holds reusable probability interfaces for UDL. It depends on
Mathlib and is independent of neural-network architectures and training losses.
Import `ProbabilityModels` for the full foundation, or import one topic such as
`ProbabilityModels.Expectation` or `ProbabilityModels.Gaussian`.

Each concept has its own file in `ProbabilityModels/`:

| File | Contents |
| --- | --- |
| `Family.lean` | Parameterized probability measures and predictive distributions |
| `Expectation.lean` | Expectation notation, linearity, affine transformations, and order properties |
| `Variance.lean` | Real and extended variance, with finite-moment requirements |
| `StandardDeviation.lean` | Square root of variance, nonnegativity, and the square identity |
| `Gaussian.lean` | Mathlib's Gaussian law and density, a `Family` adapter, and standard deviation |
| `ConditionalProbability.lean` | Conditioning a measure on an event |
| `ConditionalDistribution.lean` | Markov kernels and regular conditional distributions |
| `ConditionalExpectation.lean` | Conditional expectation and its standard identities |
| `Density.lean` | Densities relative to a reference measure |
| `Independence.lean` | Independence and conditional independence |
| `Notation.lean` | Shared notation, including explicit measures for expectations |
| `Examples/Expectation.lean` | Discrete weighted means and Gaussian expectation formulas |

Most files are documented public imports of Mathlib's existing theory.
`StandardDeviation.lean` adds `ProbabilityModels.standardDeviation`, and
`Gaussian.lean` provides `gaussianFamily : Family (ℝ × ℝ≥0) ℝ`. Its two
parameters are the mean and variance. Positive variance is required when
using the Gaussian density; zero variance gives a Dirac mass.

Expectation and variance retain Mathlib's integrability assumptions and
totalization conventions. In particular, use `MemLp X 2 μ` for a finite second
moment under a probability measure. Conditional-expectation and
conditional-distribution identities generally hold almost everywhere.

## Expectation

Start with `Expectation.lean`: `𝔼[X; P]` means `∫ ω, X ω ∂P`, directly using
Mathlib's integral. The notation is available after importing this file and
opening `ProbabilityModels`; the shared `Notation.lean` entry also exports it.

`Integrable X P` is the finite-expectation hypothesis. The basic formulas are:

- `expectation_const`: `𝔼[c; P] = c` for a constant random variable under a
  probability measure.
- `expectation_add`, `expectation_const_mul`, and
  `expectation_linearCombination`: linearity, with integrability assumptions
  for addition. Independence is not required.
- `expectation_affine`: `𝔼[aX + b; P] = a 𝔼[X; P] + b`, assuming a probability
  measure and an integrable `X`.
- `expectation_nonneg` and `expectation_mono`: positivity and preservation of
  almost-everywhere inequalities, with integrability for monotonicity.

The formulas in this list use mathematical shorthand for constant functions
and pointwise operations. The Lean theorem statements spell out those functions.

`Examples/Expectation.lean` shows `p * x + (1 - p) * y` for a two-outcome
distribution, the Gaussian mean `m`, and the transformed Gaussian mean
`a * m + b`. These examples are included in `lake build`.

## Notation

```lean
import ProbabilityModels.Notation

open scoped ProbabilityTheory ProbabilityModels NNReal
```

| Notation | Meaning | Source |
| --- | --- | --- |
| `𝔼[X; P]` | Expectation under measure `P` | This project |
| `Var[X; P]` | Real-valued variance | Mathlib |
| `eVar[X; P]` | Extended variance, allowing infinity | Mathlib |
| `SD[X; P]` | Standard deviation | This project |
| `𝒩(m, v)` | Gaussian measure with mean `m` and variance `v` | This project |
| `P[A \| B]` | Probability of event `A` conditioned on event `B` | Mathlib |
| `𝔼[X \| m; P]` | Conditional expectation given a sub-sigma-algebra `m` | This project |

For `m : ℝ` and `v : ℝ≥0`, Gaussian moments read
`𝔼[id; 𝒩(m, v)] = m`, `Var[id; 𝒩(m, v)] = (v : ℝ)`, and
`SD[id; 𝒩(m, v)] = Real.sqrt (v : ℝ)`.

For bundled probability measures, give the measure type when needed for
inference, for example
`𝔼[id; (gaussianFamily (m, v) : MeasureTheory.Measure ℝ)]`.
Keep the measure explicit in model formulas: Mathlib's `𝔼[X]` defaults to
ambient `volume`, which on `ℝ` is Lebesgue measure.
The additional notation is scoped to `ProbabilityModels`; existing Mathlib
notation remains in `ProbabilityTheory`.

## Distribution families and likelihoods

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
