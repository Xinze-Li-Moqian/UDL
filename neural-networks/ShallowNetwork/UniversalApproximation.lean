module

public import ShallowNetwork.UniversalApproximation.OneInput

/-!
# Universal approximation by shallow ReLU networks

Source: Understanding Deep Learning, Section 3.2.

- `OneInput`: every continuous function on a compact subset of the real line is
  uniformly approximated by a one-input shallow ReLU network
  (`universalApproximation : UniversalApproximation`).

Supporting lemmas, imported by the theorem files:

- `Representable`: functions computed exactly by shallow ReLU networks and their
  closure under constants, sums, and scalar multiples.
- `Interpolation`: ramps from two ReLU units, piecewise-linear interpolation,
  and its error bound.

The multi-input theorem, approximating surfaces and higher-dimensional graphs,
is not yet formalized.
-/
