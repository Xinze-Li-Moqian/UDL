# Loss functions

[`EmpiricalLoss.lean`](LossFunctions/EmpiricalLoss.lean) defines the shared type:

```lean
abbrev LossFunction (Scalar : Type u) (Parameters : Type v) := Parameters → Scalar
```

A concrete loss fixes the model and data, then maps parameters to a scalar
value to minimize. Chapter 2's `SupervisedLearning.LossFunction` is an alias
of this type, so least squares and negative log-likelihood share the same
interface. Optimization can accept any such function.

The same module defines `empiricalLoss ℓ f x y`, which constructs
`L[φ] = ∑ᵢ ℓ(f(xᵢ, φ), yᵢ)` from a single-example loss, a model, and paired
training data. The prediction and target types may differ: a prediction can
be a distribution parameter `θ`, while the target is an observed output `y`.
This constructor sums the terms without dividing by the sample count.
Its definition uses `∑ i : Fin I, ℓ (f (x i) φ) (y i)` directly. The scalar
type has an `AddCommMonoid` structure, as both `ℝ` and `ℚ` do.

Both existing losses now use this construction. Chapter 2 supplies squared
error for `ℓ`; negative log-likelihood supplies `ℓ(θ, y) = -log p(y | θ)`.
`negativeLogLikelihood_eq_neg_logLikelihood` verifies that this sum equals
minus the log-likelihood from Section 5.1.3.

[`BinaryCrossEntropy.lean`](LossFunctions/BinaryCrossEntropy.lean) defines the
reusable single-example loss `binaryCrossEntropy p y` as the negative log
Bernoulli mass. Its formula theorem gives
`ℓ(p, y) = −(1 − y) log(1 − p) − y log p`, and simplification rules handle
labels 0 and 1. The usual finite interpretation assumes `0 < p < 1`;
endpoint behavior follows the totalized real logarithm used by the shared
negative log-likelihood. The classification example sums this loss using
`empiricalLoss` and proves agreement with the common negative log-likelihood.

[`MulticlassCrossEntropy.lean`](LossFunctions/MulticlassCrossEntropy.lean)
defines the corresponding single-example categorical loss
`multiclassCrossEntropy p y = −log(p.weights y)`. Here `p` is a probability
vector and `y` selects the observed label. This loss is independent of the
model and softmax; its usual finite interpretation requires positive mass
at the observed label.

`EmpiricalLoss.lean` also enables the book's `L[φ]` notation with
`open scoped LossFunctions`. Here `L` is the actual loss function
bound in the surrounding expression; `L[φ]` means `L φ`.

Section 5.1.1 defines
[`networkPredictiveDistribution`](LossFunctions/PredictiveDistribution.lean).
It composes the existing network with a conversion to distribution parameters
and a chosen probability family:

```text
x → network a φ x → θ → Pr[· | θ; family]
```

The `Pr` notation comes from `ProbabilityModels.Family`:
`Pr[· | θ; family]` expands to `family θ`, and `Pr[· | x; model, φ]`
expands to `model φ x` after the family and parameter conversion are fixed.
The observed output still belongs to `Y`; the prediction is a probability
distribution on `Y`. Run `lake build LossFunctions` to check the module.

`networkPredictiveDistribution_eq` proves that the prediction at `x` equals
the family member with parameters `toParameters (network a φ x)`.
The general version, `Family.predictiveDistribution_eq`, writes these parameters
as `f x φ`. Both are definitional equalities, proved by `rfl`.

Section 5.1.2 adds
[`MaximumLikelihood.lean`](LossFunctions/MaximumLikelihood.lean):

- `likelihood p f x y φ` multiplies the chosen predictive densities
  `p (f (x i) φ) (y i)` across the observations.
- `IsMaximumLikelihoodEstimator p f x y «φ̂»` means that this likelihood
  at `φ̂` is at least its value at every other parameter choice.

With `open scoped ProbabilityModels Optimization`, the definition reads:

```lean
«φ̂» ∈ argmax φ, ∏ i : Fin I, Pr[y i | f (x i) φ; p]
```

`φ` ranges over candidate parameters; `«φ̂»` is Lean's quoted identifier for
the estimate `φ̂`. `argmax` is the set of all maximizers, so the formula uses
membership (`∈`). `Pr[y | θ; p]` evaluates the chosen density at `y`, while
`Pr[· | θ; family]` denotes the entire distribution. Expanding these notations
gives the original optimality condition, as proved by
`isMaximumLikelihoodEstimator_iff`.

The densities come from `ProbabilityModels.Family.Density`, with a fixed
reference measure. The product uses the conditional independence assumption
in equation (5.2); independence is not inferred from sharing a family.
The maximum-likelihood criterion specifies an optimum without assuming one
exists or providing a solver. `likelihood_nonneg` proves nonnegativity.

The event-probability factorization behind equation (5.2) is provided by
`ProbabilityModels.independent_probability_forall` in
[`ProbabilityModels/Independence.lean`](../probability/ProbabilityModels/Independence.lean).
It is a general probability theorem, reusable for a joint output model after
fixing inputs and parameters. A joint-density formula is a later step.

Sections 5.1.3 and 5.1.4 have separate files alongside `MaximumLikelihood.lean`:

- [`LogLikelihood.lean`](LossFunctions/LogLikelihood.lean) defines
  `logLikelihood` as a sum of log densities and its maximum-likelihood criterion.
  It proves that the sum is the log of the likelihood, that candidate ordering
  is preserved, and that the two criteria have the same maximizers.
- [`NegativeLogLikelihood.lean`](LossFunctions/NegativeLogLikelihood.lean)
  defines `negativeLogLikelihood` and its minimization criterion. It proves
  that negation turns log-likelihood maximization into loss minimization,
  then connects this criterion to the original maximum likelihood.

The logarithmic equivalences assume strictly positive densities at every
observation for the compared candidates (all candidates for global optimality).
The definitions use `Real.log`, whose totalized value at zero is `0`; these
results do not cover the extended-log treatment of zero density. Negation alone
needs no positivity assumption. None of these equivalences assumes an optimum
exists or that it is unique.

The minimization criterion reuses `Optimization.IsArgmin` through set notation:

```lean
open scoped LossFunctions Optimization

let L : LossFunction ℝ Parameters := negativeLogLikelihood p f x y
«φ̂» ∈ argmin φ, L[φ]
```

`negativeLogLikelihood_apply` states the book's formula directly as
`L[φ] = -∑ i : Fin I, Real.log Pr[y i | f (x i) φ; p]`.

Section 5.1.5 adds [`Inference.lean`](LossFunctions/Inference.lean).
`IsModePrediction` fixes the input and model parameters, then optimizes the
output rather than the parameters:

```lean
ŷ ∈ argmax y, Pr[y | f x «φ̂»; p]
```

This describes a mode of the chosen predictive density, without asserting
existence or uniqueness for arbitrary families. `gaussian_mean_isModePrediction`
proves that the predicted mean satisfies this criterion for a positive-variance
Gaussian. The underlying density inequality, `positiveGaussianDensity_le_mean`,
lives in `ProbabilityModels.Gaussian`.

Section 5.2 assembles these existing definitions into the book's recipe:

1. Choose a `Family` and its `Family.Density`.
2. Use the network and `toParameters` to predict distribution parameters.
3. Specify fitted parameters with `IsNegativeLogLikelihoodMinimizer`.
4. For a new input, return the predictive distribution or an output satisfying
   `IsModePrediction`.

Step 3 specifies the training objective; an algorithm for fitting general
deep networks is still to be added.

Univariate regression has its own
[`UnivariateRegression/`](LossFunctions/UnivariateRegression/README.md) directory,
with [`UnivariateRegression.lean`](LossFunctions/UnivariateRegression.lean)
as the import entry point. Here "univariate" means a single real-valued output;
the input and distribution-parameter types are arbitrary.

[`Model.lean`](LossFunctions/UnivariateRegression/Model.lean) defines the shared
`Model` interface from an existing distribution family, its certified density,
and a predictor of distribution parameters. `model.predictiveDistribution`
returns the full output distribution; `model.loss` reuses `negativeLogLikelihood`.
The distribution-independent `leastSquaresLoss` also lives here.

Sections 5.3–5.3.3 are implemented in
[`Gaussian.lean`](LossFunctions/UnivariateRegression/Gaussian.lean).
`gaussianModel f v` instantiates the shared interface using
`ProbabilityModels.Gaussian`. `gaussianParameters` pairs the predicted mean
`f x φ` with a fixed positive variance `v = σ²`; `gaussianLoss` uses this
model's shared loss operation. Other distributions can add their own modules.

`gaussianLoss_eq` proves the exact identity
`L[φ] = (I / 2) * log (2 * π * v) + S[φ] / (2 * v)`.
The constant and scale are independent of `φ`, and the scale is positive.
`mem_argmin_gaussianLoss_iff` therefore proves that the two losses have the
same minimizers, without asserting existence or uniqueness.
`mean_isModePrediction` reuses the Gaussian mode theorem for inference.

For Section 5.3.3, `gaussianModelWithLearnedVariance` extends the model parameters
to a pair `(φ, v)` with `v > 0`. `jointGaussianLoss` uses the same shared loss
operation, and `IsGaussianJointEstimator` states joint optimality. Its exact
loss expansion retains both variance-dependent terms. This defines the fitting
criterion; it does not compute an estimate or assert that a minimizer exists.
`learnedVariance_mean_isModePrediction` proves that the predicted mean is still
a mode with this parameterization.

Section 5.3.4 is implemented in
[`Heteroscedastic/Gaussian.lean`](LossFunctions/UnivariateRegression/Heteroscedastic/Gaussian.lean),
under the same univariate interface. Its Gaussian constructor accepts mean
and positive-variance predictors depending on both input and model parameters.
The previous constant-variance constructors specialize this more general one.
The module proves the loss expansion and mode inference, and implements the
book's squared second output with an explicit nonzero-output hypothesis.
See [the directory guide](LossFunctions/UnivariateRegression/Heteroscedastic/README.md).

Section 5.4 is an application of the shared interfaces in
[`Examples/BinaryClassification/`](LossFunctions/Examples/BinaryClassification/README.md),
imported through `LossFunctions.Examples.BinaryClassification`. Labels use `Fin 2`.
Its Bernoulli model uses the shared `ActivationFunction.Sigmoid` interface
(backed by Mathlib's sigmoid) on a real model score and reuses
`ProbabilityModels.bernoulliFamily` and its certified probability masses.
`crossEntropyLoss` reuses the common negative log-likelihood, and its expansion
theorem gives equation (5.20). Strictly positive predicted masses supply the
likelihood-equivalence hypotheses. The probability threshold rule, selecting
label 1 above one half and label 0 otherwise, is proved to select a mode.

Figure 5.9 begins the next example in
[`Examples/MulticlassClassification/`](LossFunctions/Examples/MulticlassClassification/README.md).
Its categorical probabilities are specified as cat 0.7, dog 0.2, bird 0.1.
The example verifies the probabilities, their sum, and that cat is a mode,
using `ProbabilityModels.Categorical`.
`Examples/MulticlassClassification/Softmax.lean` connects a vector-valued
predictor to the same categorical family using `ActivationFunction.Softmax`.
It proves equation (5.23), positivity of predicted masses, and equivalence
between probability modes and maximal raw scores.
`Examples/MulticlassClassification/CrossEntropy.lean` sums the reusable
single-example loss using `empiricalLoss`, proves both lines of equation
(5.24), and reuses maximum-likelihood equivalence for the resulting objective.

Section 5.6 is implemented in
[`MultipleOutputs/`](LossFunctions/MultipleOutputs/README.md), imported through
`LossFunctions.MultipleOutputs`. It combines per-target parameter predictors
with the product probability families and certified densities from
`ProbabilityModels.Product`. Output types and distribution families may vary
by target. `predictiveDensity_eq` proves (5.25); `loss_eq_sum` proves the double
sum in (5.26) under positive observed component densities.
`loss_single` recovers an original single-output negative log-likelihood.
[`Examples/MultipleOutputs.lean`](LossFunctions/Examples/MultipleOutputs.lean)
checks this reduction against the existing Gaussian, binary, and multiclass
loss definitions.

[`Examples/Data/ToyRegression.lean`](LossFunctions/Examples/Data/ToyRegression.lean)
stores ten synthetic `(x, y)` pairs near `y = 1 + 2x`, with small fixed
perturbations, for future training examples.

Deep-network training examples will follow once general back-propagation
is available.
