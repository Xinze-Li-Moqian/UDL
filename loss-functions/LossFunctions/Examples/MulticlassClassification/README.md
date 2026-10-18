# Example: multiclass classification

`Categorical.lean` starts with the distribution in Figure 5.9. Its probabilities
are specified by hand: cat 0.7, dog 0.2, bird 0.1. The example proves:

- each label has the specified probability;
- the corresponding probability measure gives the same singleton probability;
- the masses sum to one;
- cat is a most likely label.

The generic distribution and mass function live in `ProbabilityModels.Categorical`.
Its `CategoricalParameters K` reuses Mathlib's standard simplex to encode
nonnegative probabilities summing to one. `PMF` supplies the probability measure.

The examples use the shared notation `Pr[k | animalProbabilities; categoricalMass]`.
Lean labels are indexed by `Fin K` from 0 through K − 1. The book indexes
them from 1 through K; this changes only the names of the labels.

`Softmax.lean` then constructs parameters from a predictor with K real outputs:

    x, φ → z = f(x, φ) → λ = softmax(z) → Categorical(λ).

The vector transformation lives in `ActivationFunction.Softmax`. It proves
positive entries, normalization, and preservation of label ordering within
one score vector. Softmax is a function; the probability distribution remains
the categorical family from the probability foundation.

`predictedParameters` packages the softmax vector as `CategoricalParameters K`.
`predictiveDistribution` returns the full distribution, while `predictiveMass_eq`
proves equation (5.23). The inference theorem proves that the most probable
labels are exactly the labels with maximal raw scores. The illustrative
scores `(log 7, log 2, 0)` recover the earlier probabilities `(0.7, 0.2, 0.1)`.

With a predictor `f`, the shared notation reads:

```lean
Pr[k | predictedParameters f x φ; categoricalMass]
Pr[· | x; predictiveDistribution f, φ]
```

The label count must be positive, encoded by `[NeZero K]`. Import
`LossFunctions.Examples.MulticlassClassification` for both examples.

`CrossEntropy.lean` adds the training objective in equation (5.24), using the
reusable `LossFunctions.multiclassCrossEntropy` single-example loss:

    ℓ(λ, y) = −log λᵧ,
    L[φ] = ∑ᵢ ℓ(softmax(f(xᵢ, φ)), yᵢ).

`crossEntropyLoss` uses `empiricalLoss`, and
`crossEntropyLoss_eq_negativeLogLikelihood` proves agreement with the existing
categorical negative log-likelihood. `crossEntropyLoss_eq` proves the first
line of (5.24); `crossEntropyLoss_eq_logSumExp` proves the second:

    L[φ] = −∑ᵢ [fᵧᵢ(xᵢ, φ) − log(∑ₖ exp(fₖ(xᵢ, φ)))].

The single-example identity `NeuralNetworks.neg_log_softmax` supplies this
expansion. Positive predicted masses justify the reused maximum-likelihood
equivalence. The earlier mode theorem supplies the book's inference criterion.
These definitions and proofs do not run a training algorithm.
