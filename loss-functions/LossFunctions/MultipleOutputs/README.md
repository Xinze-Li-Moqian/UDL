# Multiple outputs

Source: Understanding Deep Learning, Section 5.6, equations (5.25)–(5.26).

One input now has several target values:

    y = (y₁, …, yᴰ).

Each target has its own distribution parameters, obtained by grouping the
existing network's output coordinates. For example, D categorical targets
with K classes use D groups of K scores, with softmax applied to each group.
Targets may also have different types and distribution families.

For the book's factorized model, the targets are assumed conditionally
independent given the input and model parameters:

    p(y | x, φ) = ∏ᵈ pᵈ(yᵈ | θᵈ(x, φ)).

The negative log-likelihood then sums over samples and targets:

    L[φ] = −∑ᵢ ∑ᵈ log pᵈ(yᵢᵈ | θᵈ(xᵢ, φ)).

## Implementation

`../MultipleOutputs.lean` is the import entry point. `Model.lean` uses a finite
index type `ι` for targets; take `ι = Fin D` for D outputs. Input and model
parameter types remain arbitrary. The types `Y d` and `Θ d` may depend on the
target, allowing different distributions and target domains in the same model.

`ProbabilityModels.Product` supplies the reusable probability construction:

- `Family.pi` is the finite product of component probability families.
- `Family.pi_map_eval` recovers each component law by projecting its coordinate.
- `Family.pi_independent` proves mutual independence under the constructed law.
- `Family.Density.pi` certifies the product density against the product law,
  using sigma-finite reference measures. Lebesgue and finite counting measures
  satisfy this assumption.

This is a chosen independent model at each fixed input and parameter value.
The independence follows from its product construction.

In `Model.lean`, `predictiveDistribution families f` combines the component
predictors `f d : X → Parameters → Θ d`. Each predictor interprets a group of
network outputs. `predictiveDensity_eq` proves (5.25):

```lean
Pr[y | (fun d => f d x φ); Family.Density.pi p] =
  ∏ d, Pr[y d | f d x φ; p d]
```

`loss p f x y` reuses `negativeLogLikelihood` for this joint family.
`loss_eq_sum` proves (5.26) under positive observed component densities.

## Single-output recovery

Take `ι = Fin 1`. `Family.Density.pi_single` reduces the joint density to the
original density, and `Family.pi_single_map_zero` recovers the original law
by selecting the sole coordinate. The tuple-valued output and the original
output have different types; the projection explicitly identifies their laws.

`loss_single` recovers any original single-output negative log-likelihood.
[`Examples/MultipleOutputs.lean`](../Examples/MultipleOutputs.lean) checks this with the existing Gaussian,
Bernoulli, and categorical model and loss interfaces.

The target count is distinct from input dimension and network output width.
A categorical predictor may have K scores while still predicting one label:
as a multiple-output model, that case has D = 1.
