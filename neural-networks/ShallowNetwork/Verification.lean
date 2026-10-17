module

public import ShallowNetwork.Verification.Model
public import ShallowNetwork.Verification.WidthThree
public import ShallowNetwork.Verification.MultiInput

/-!
# Shallow network verification

These checks confirm that the definitions mean what they claim: edge cases,
agreement with the book's formulas, and agreement between equivalent
formulations. They establish no new mathematical results; those live in
topic modules such as `ShallowNetwork.UniversalApproximation`.

- `Model`: the one-input network at zero width.
- `WidthThree`: the book's width-three formula, equation (3.4).
- `MultiInput`: the multi-input network at zero width, and its agreement with
  the one-input model when `Dᵢ = 1`.
-/
