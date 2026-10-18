# Example: binary classification

Section 5.4 models a label `y : Fin 2`, whose value is either 0 or 1.
The input type and network parameters are unrestricted. The construction is:

    x, φ → f(x, φ) → λ = sig(f(x, φ)) → Bernoulli(λ).

This example instantiates the common likelihood and inference interfaces.
`Bernoulli.lean` supplies the model, loss, and inference rule, in the namespace
`LossFunctions.Examples.BinaryClassification`.
`../BinaryClassification.lean` is the import entry point. The underlying
distribution lives in `ProbabilityModels.Bernoulli` and reuses Mathlib's
Bernoulli measure. Its mass function is a density relative to counting measure,
so the same likelihood interface works for these discrete labels.

- `predictedProbability` produces the probability of label 1; label 0 has
  probability `1 - λ`. `ActivationFunction.Sigmoid` exposes the underlying
  Mathlib function as the shared `NeuralNetworks.sigmoid` activation.
- `predictiveDistribution` returns the full two-label probability distribution.
- `predictiveMass_eq` proves the substituted probability formula (5.19).
- `crossEntropyLoss` uses `empiricalLoss binaryCrossEntropy` to sum the
  reusable single-example loss. `crossEntropyLoss_eq_negativeLogLikelihood`
  proves it is the shared negative log-likelihood, and `crossEntropyLoss_eq`
  proves the expansion (5.20):

      L[φ] = ∑ᵢ [−(1 − yᵢ) log(1 − λᵢ) − yᵢ log λᵢ],
      λᵢ = sig(f(xᵢ, φ)).

- `isMaximumLikelihoodEstimator_iff_mem_argmin` reuses the equivalence
  between maximum likelihood and minimizing this loss.
- `predictLabel` selects 1 if `λ > 1/2` and 0 otherwise;
  `predictLabel_isModePrediction` proves this is a mode, including the tie case.
  `predictLabel_eq_score_rule` proves this is equivalent to selecting 1 for
  a positive raw score and 0 otherwise.

For example, with the existing probability notation:

```lean
open ProbabilityModels LossFunctions.Examples.BinaryClassification
open scoped ProbabilityModels

Pr[1 | predictedProbability f x φ; bernoulliMass]
Pr[· | x; predictiveDistribution f, φ]
```

The Bernoulli family includes endpoint probabilities 0 and 1. The sigmoid
model always gives `0 < λ < 1` for finite real scores, so neither observed
label has zero probability in these loss and likelihood theorems.
The model and objective are formalized; no classification training algorithm
is supplied in this step.
