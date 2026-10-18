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
| `Variance.lean` | Centered and second-moment formulas, affine transformations, and zero variance |
| `StandardDeviation.lean` | Square-root definition, affine transformations, and zero standard deviation |
| `Gaussian.lean` | Gaussian families, a positive-variance density adapter, its maximum at the mean, and standard deviation |
| `ConditionalProbability.lean` | Event conditioning, normalization, ratio and Bayes rules |
| `ConditionalDistribution.lean` | Predictive kernels and the links between conditional laws and expectations |
| `ConditionalExpectation.lean` | Defining integral identity, measurability, averaging, and the tower property |
| `Density.lean` | Chosen family densities, normalization, event probabilities, and predictive densities |
| `Independence.lean` | Factorization of events, joint laws, and expectations; variances of independent sums |
| `Notation.lean` | Shared notation, including explicit measures for expectations |
| `Examples/Expectation.lean` | Discrete weighted means and Gaussian expectation formulas |
| `Examples/Conditioning.lean` | A Gaussian predictive kernel with mean equal to the model output |
| `Examples/Density.lean` | A normalized Gaussian predictive density, consistent with that kernel |
| `Examples/Moments.lean` | Two-outcome variance and affine Gaussian variance and standard deviation |
| `Examples/Independence.lean` | Variance of a sample mean under pairwise independence |

The files reuse Mathlib's probability theory and add interfaces for models.
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

## Variance and standard deviation

`Variance.lean` uses Mathlib's `Var[X; P]` and `eVar[X; P]` directly. For a
probability measure and `MemLp X 2 P`, its main formulas are:

- `variance_eq_expectation_centered_sq`: `Var[X; P] = 𝔼[(X - 𝔼[X; P])²; P]`.
- `variance_eq_secondMoment_sub_sq`: `Var[X; P] = 𝔼[X²; P] - 𝔼[X; P]²`.
- `variance_const`, `variance_shift`, and `variance_affine`: constants have
  zero variance, shifts preserve it, and `Var[aX + b; P] = a² * Var[X; P]`.
- `variance_eq_zero_iff`: zero variance is equivalent to equality with the
  mean almost everywhere. The finite second moment is essential for this
  interpretation of real-valued variance.

Mathlib's `variance_nonneg`, `variance_const_mul`, and `evariance_lt_top` are
also exported. The last theorem certifies finite extended variance from
`MemLp X 2 P` under a finite measure.

`StandardDeviation.lean` defines `SD[X; P] = Real.sqrt (Var[X; P])`. In addition
to nonnegativity and the square identity, it provides almost-everywhere
congruence, constants, shifts, scaling, and the affine rule
`SD[aX + b; P] = |a| * SD[X; P]`. With a finite second moment, zero standard
deviation is equivalent to equality with the mean almost everywhere.
The `SD` notation is available from this concept file and the shared
`Notation.lean` entry point.

`Examples/Moments.lean` proves the two-outcome variance
`p * (1 - p) * (x - y)²`, and the Gaussian affine formulas `a² * v` for
variance and `|a| * sqrt v` for standard deviation. The Gaussian examples
include zero variance and arbitrary real scales.

## Independence

Use Mathlib's `IndepFun X Y P` (notation `X ⟂ᵢ[P] Y`) for two variables, and
`iIndepFun X P` for a mutually independent family. `Independence.lean` connects
these assumptions to the operations used in probability models:

- `independent_probability_inter` and `independent_probability_iInter`
  factor probabilities of simultaneous output events.
- `independent_probability_forall` writes the finite-family rule as
  `P {ω | ∀ i, Z i ω ∈ events i} = ∏ i, P {ω | Z i ω ∈ events i}`.
  Taking the index type to be `Fin I` covers `I` observations.
- `independent_iff_jointLaw_eq_prod` and `independent_iff_jointLaw_eq_pi`
  characterize independence by a joint law equal to the product of its
  marginal laws, assuming measurable variables under a probability measure.
- `integrable_mul_of_independent` and `integrable_prod_of_independent`
  certify integrability of the products. The corresponding
  `expectation_mul_of_independent` and `expectation_prod_of_independent`
  express their expectations as products of expectations.
- `variance_add_of_independent` and `variance_sum_of_pairwise_independent`
  add variances, assuming finite second moments. For the finite sum, pairwise
  independence is enough; the finite product rules require mutual independence.

These results do not require identical marginal distributions. Conditional
independence remains available through the Mathlib public import.
`Examples/Independence.lean` proves that the average of `n > 0` pairwise
independent variables, each of variance `v`, has variance `v / n`.

## Conditioning

The three conditioning modules keep the different kinds of condition explicit:

- `ConditionalProbability.lean` conditions on an event `B`. It supplies the
  ratio and multiplication formulas, total probability over `B` and `Bᶜ`,
  Bayes' rule, and the indicator-expectation identity. A proof that `P B ≠ 0`
  ensures `P[|B]` is a probability measure. On null events, Mathlib's convention
  gives the zero measure; the algebraic identities retain that convention.
- `ConditionalExpectation.lean` conditions on a sub-sigma-algebra `s ≤ mΩ`.
  The result `𝔼[X | s; P]` is a random variable. The defining integral identity
  holds on every `s`-measurable event for integrable `X`. The module also gives
  constants, already-measurable variables, additivity, total expectation,
  the tower property, and conditioning on the trivial sigma-algebra.
- `ConditionalDistribution.lean` uses `condDistrib Y X P`, a Markov kernel
  giving a distribution on outputs for each input. The output space is
  nonempty and standard Borel. The joint-law theorem characterizes its relation
  to the law of `(X,Y)`; at inputs of nonzero probability it recovers the ratio
  formula. Event probabilities are conditional expectations of indicators,
  and conditional means are integrals against the conditional distribution.
  The latter identities hold almost everywhere, with measurability and
  integrability assumptions as stated in the Lean theorems.

For models, `Family.toKernel` converts a measurable distribution family to a
Markov kernel. `Family.predictiveKernel` composes it with a predictor measurable
in its input at fixed model parameters. Its `IsMarkovKernel` instance certifies
normalization at every input. This constructs a model; identifying it with the
conditional law of a given data distribution is a separate claim.

`Examples/Conditioning.lean` applies this construction to `gaussianFamily`:
the kernel at `x` is `𝒩(f x φ, v)`, and its expectation is `f x φ`.

## Densities

`Family.Density family ν` specifies a real-valued function `p θ y` for every
parameter, relative to one common reference measure `ν`. Its fields certify
nonnegativity, measurability in `y`, and the measure identity
`ν.withDensity (fun y => ENNReal.ofReal (p θ y)) = family θ`.
Normalization is a theorem derived from this identity, rather than a second
independent assumption.

- `probability_eq_lintegral` and `probability_eq_integral` compute event
  probabilities by integrating the density over the event.
- `lintegral_eq_one`, `integral_eq_one`, and `integrable` certify normalization
  and finite integrability.
- `expectation_eq_integral` computes an expectation as the integral of the
  observable multiplied by the density.
- `hasPDF` connects the certificate to Mathlib's `HasPDF` for the identity
  random variable under the family member.
- `p.predictiveDensity f x φ y` evaluates `p (f x φ) y` at an observation.
  `withDensity_predictiveDensity` and `integral_predictiveDensity_eq_one`
  certify its agreement with the predictive distribution and its normalization.

This interface chooses finite density values at every observation, including
points in null sets; an almost-everywhere equivalence class alone does not
specify those values. Densities may be zero, and continuous density values may
exceed one. Counting measure gives probability masses; Lebesgue measure gives
continuous densities. The latter are integrated over events to obtain
probabilities, rather than interpreted as singleton probabilities.

`PositiveGaussianParameters` is a mean paired with a strictly positive
variance. `positiveGaussianFamily` restricts the existing Gaussian family to
these parameters, and `positiveGaussianDensity` certifies its usual density
relative to Lebesgue measure. The density is strictly positive everywhere;
`positiveGaussianDensity_le_mean` proves that it attains its maximum at the mean.
The original `gaussianFamily` still includes variance zero, which gives a
Dirac measure and has no Lebesgue density.

`Examples/Density.lean` specializes to a fixed positive variance and a predicted
mean `f x φ`. It proves the familiar Gaussian formula, normalization, and
agreement with `gaussianPredictionKernel`.

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
| `Pr[· \| θ; family]` | The entire distribution `family θ`, with parameters `θ` | This project |
| `Pr[· \| x; model, φ]` | The prediction `model φ x` at input `x` with model parameters `φ` | This project |
| `Pr[y \| θ; p]` | The chosen probability mass or density `p θ y` at output `y` | This project |
| `P[A \| B]` | Probability of event `A` conditioned on event `B` | Mathlib |
| `𝔼[X \| m; P]` | Conditional expectation given a sub-sigma-algebra `m` | This project |
| `X ⟂ᵢ[P] Y` | Independence of two random variables under `P` | Mathlib |

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

The `Family` interface specifies a probability distribution at each input;
`ConditionalDistribution.lean` connects it to kernels, and `Density.lean`
supplies chosen densities relative to a fixed reference measure. Likelihood
and negative-log-likelihood constructions will have their own files in
`../loss-functions/` as Chapter 5 develops. Products over observations still
need a sampling model; logarithmic identities need the appropriate positivity
assumptions on observed density values.
