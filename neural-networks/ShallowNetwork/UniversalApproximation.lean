module

public import ShallowNetwork.UniversalApproximation.OneInput
public import ShallowNetwork.UniversalApproximation.MultiInput

/-!
# Universal approximation by shallow ReLU networks

Source: Understanding Deep Learning, Sections 3.2–3.4.

- `OneInput`: every continuous function on a compact subset of the real line is
  uniformly approximated by a one-input shallow ReLU network
  (`universalApproximation : UniversalApproximation`).
- `MultiInput`: every continuous function on a compact subset of `ℝ^Dᵢ` is
  uniformly approximated by a shallow ReLU network with `Dᵢ` inputs
  (`MultiInput.universalApproximation : MultiInput.UniversalApproximation`).
  With two inputs, this approximates surfaces.

Supporting lemmas, imported by the theorem files:

- `Representable`: functions computed exactly by shallow ReLU networks and their
  closure under constants, sums, and scalar multiples.
- `Interpolation`: ramps from two ReLU units, piecewise-linear interpolation,
  and its error bound.
- `Ridge`: approximation of ridge functions via the one-input theorem, and the
  algebra of exponential ridges used with Stone–Weierstrass.
-/
