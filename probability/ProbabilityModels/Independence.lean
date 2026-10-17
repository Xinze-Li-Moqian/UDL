module

public import Mathlib.Probability.Independence.Basic
public import Mathlib.Probability.Independence.Conditional

/-!
# Independence

Use `ProbabilityTheory.IndepFun` for two independent random variables and
`ProbabilityTheory.iIndepFun` for an independent family. Mathlib also supplies
conditional independence in `Probability.Independence.Conditional`.

These are the probabilistic assumptions used to justify product laws for
observations. The construction of a sample likelihood from conditional
densities belongs to `loss-functions/`.
-/
