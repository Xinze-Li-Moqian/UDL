module

public import ShallowNetwork.LinearRegions.OneInput
public import ShallowNetwork.LinearRegions.MultiInput

/-!
# Linear regions of shallow ReLU networks

Source: Understanding Deep Learning, Sections 3.1–3.3.

A shallow ReLU network is piecewise affine. These modules bound how many affine
pieces a network of a given width can have.

- `OneInput`: with `D` hidden units, the real line splits into at most `D + 1`
  intervals on which the network is affine (`linearRegionBound : LinearRegionBound`).
- `MultiInput`: with `Dᵢ` inputs, `ℝ^Dᵢ` splits into at most
  `∑ j ≤ Dᵢ, (D choose j)` activation patterns, on each of which the network is
  affine (`MultiInput.linearRegionBound : MultiInput.LinearRegionBound`).

Supporting lemmas:

- `ActivationPattern`: the network is affine on each activation pattern, and no
  `Dᵢ + 1` units realize all their on/off combinations.
-/
