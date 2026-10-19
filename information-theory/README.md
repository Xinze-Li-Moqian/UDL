# Information theory

This directory sits alongside `probability/` and `loss-functions/`.
Probability supplies distributions and expectations; information theory studies
entropy and divergence; learning losses use these concepts to define objectives.

Import `InformationTheoryModels` for this layer, or import
`InformationTheoryModels.KullbackLeibler` individually.

| File | Contents |
| --- | --- |
| `InformationTheoryModels/KullbackLeibler.lean` | Mathlib's KL divergence adapted to bundled probability measures |

`klDivergence q p` reuses `InformationTheory.klDiv` directly. Its value lies in
`ℝ≥0∞`: nonnegative real values together with infinity. The module proves
self-divergence is zero, zero divergence characterizes equal distributions,
and failure of absolute continuity gives infinite divergence.

Entropy and the KL/cross-entropy decomposition are not yet implemented here.
The current empirical cross-entropy/NLL connection remains in
[`LossFunctions.CrossEntropy`](../loss-functions/LossFunctions/CrossEntropy.lean).
Future general entropy identities belong in this layer; model and dataset loss
constructions remain in `loss-functions/`.
