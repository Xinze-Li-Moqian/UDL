module

public import ShallowNetwork.LinearRegions.OneInput

/-!
# Linear regions of shallow ReLU networks

Source: Understanding Deep Learning, Sections 3.1–3.3.

A shallow ReLU network is piecewise affine. These modules bound how many affine
pieces a network of a given width can have.

- `OneInput`: with `D` hidden units, the real line splits into at most `D + 1`
  intervals on which the network is affine (`linearRegionBound : LinearRegionBound`).
-/
