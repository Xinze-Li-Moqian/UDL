module

public import DeepNetwork.LinearRegions.UpperBound
public import DeepNetwork.LinearRegions.Attainability

/-!
# Linear regions of one-input deep ReLU networks

Source: Understanding Deep Learning, Section 4.5.2.

`deepLinearRegionBound` bounds arbitrary hidden widths by `∏ k, (D_k+1)`.
`deepLinearRegionBound_uniform` specializes this to `(D+1)^K`.
`deepLinearRegionBound_attained` constructs an equal-width network attaining
the bound when `D ≥ 3`. `HasExactlyRegions` rules out redundant subdivisions.
-/
