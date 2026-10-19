module

public import FittingModels.FloatingPoint
public meta import FittingModels.FloatingPoint

/-!
# FloatLib integration

Use certified binary32 operations, independently of the rational training
examples. This first check connects executable addition to its FloatLib
specification and displays the exact rational value stored in the result.
It does not yet migrate training or certify a trained model's loss bound.

Build explicitly with `lake build FittingModels.Examples.FloatArithmetic`.
-/

@[expose] public section

namespace FittingModels.Examples.FloatArithmetic

open FloatLib.Floats

/-- IEEE binary32: one sign bit, eight exponent bits, and 23 stored fraction bits. -/
abbrev Binary32 := FittingModels.Binary32

/-- The executable addition agrees with the library's certified specification. -/
theorem add_eq_spec (a b : Binary32) : a + b = ExecFloat.Spec.add a b :=
  ExecFloat.Proof.add_notation_eq_spec a b

-- Exact decoding returns `none` for exceptional values such as NaN or infinity.
#eval ExecFloat.Binary.toRat? ((1.5 : Binary32) + 2.25)
#eval ExecFloat.Binary.toRat? ((0.1 : Binary32) + 0.2)

end FittingModels.Examples.FloatArithmetic
