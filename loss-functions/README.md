# Loss functions

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

[`Examples/Data/ToyRegression.lean`](LossFunctions/Examples/Data/ToyRegression.lean)
stores ten synthetic `(x, y)` pairs near `y = 1 + 2x`, with small fixed
perturbations, for future training examples.

Log-likelihoods and loss functions are the next steps; deep-network training
examples will follow once general back-propagation is available.
