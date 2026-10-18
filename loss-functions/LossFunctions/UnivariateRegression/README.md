# Univariate regression

The target `y : ℝ` is one-dimensional. Inputs and predicted distribution
parameters may have any dimension. Predicting a mean is a choice made by
the Gaussian model, rather than a requirement of the shared interface.

- `Model.lean` defines `Model X Parameters Θ`, packaging an existing `Family`,
  its reference measure and certified `Family.Density`, and a predictor
  `predictParameters : X → Parameters → Θ`.
- `Gaussian.lean` supplies `gaussianModel f v` using the existing positive-variance
  Gaussian family. It proves the fixed-variance loss identity, equality of
  minimizers with least squares, and the Gaussian mode prediction. Section 5.3.3
  adds `gaussianModelWithLearnedVariance`, whose parameter pair includes one
  shared positive variance, with a joint loss and an optimality criterion.
- `../UnivariateRegression.lean` imports the interface and implementations.

All implementations use the same prediction and loss operations:

```lean
open scoped LossFunctions ProbabilityModels Optimization

let model := gaussianModel f v
let L := model.loss x y
«φ̂» ∈ argmin φ, L[φ]
```

For a new input `xNew`, return the full predictive distribution with:

```lean
Pr[· | xNew; model.predictiveDistribution, «φ̂»]
```

When fitting the variance as well, use `gaussianModelWithLearnedVariance f`
and optimize over `ψ = (φ, v)`. `jointGaussianLoss_eq` proves
`L[φ, v] = (I / 2) log(2πv) + S[φ] / (2v)`; the log term depends on `v`
and must remain in the joint objective. A fitted pair gives the predictive
distribution `𝒩(f(x, φ̂), v̂)`, whose mean is a mode. A fitting algorithm and
a closed-form estimate of the variance are not supplied here.

`model.loss` calls the existing `negativeLogLikelihood`; its positivity
requirements for likelihood equivalences and its treatment of zero density
are documented there and in `LogLikelihood.lean`.

Future distributions, such as Laplace, can supply their own module and use
these same operations. Their probability definitions belong in
`ProbabilityModels`; their regression choices, loss derivations, and examples
belong here. Each point-prediction rule needs justification for that family.
Only the Gaussian implementation is present so far.
