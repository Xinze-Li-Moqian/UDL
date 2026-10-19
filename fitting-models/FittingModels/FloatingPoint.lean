module

public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Instances
public import FloatLib.Floats.Formats.BinaryInterchange.Configured.Plan.Instances
public import FloatLib.Floats.ExecFloat.Proof.Arithmetic
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured.Instances
public meta import FloatLib.Floats.Formats.BinaryInterchange.Configured.Plan.Instances

/-!
# Floating-point execution

Use FloatLib's certified binary32 arithmetic. Convert the original rational
data directly with one nearest-even rounding, and decode finite results exactly
for later checks. Decoding returns `none` for NaN and infinity.
-/

@[expose] public section

namespace FittingModels

open FloatLib.Floats
open FloatLib.Floats.Formats.BinaryInterchange

/-- IEEE binary32 with 24 significant binary digits for normal numbers. -/
abbrev Binary32 := ExecFloat.Binary (exponentBits := 8) (fractionBits := 23)

/-- Round an exact rational once into binary32, using nearest, ties to even. -/
def binary32OfRat (q : ℚ) : Binary32 :=
  ExecFloat.Binary.ofModel (Model.roundRatQ FloatFormat.binary32 q)

/-- Recover the exact rational represented by a finite binary32 value. -/
def binary32ToRat? (x : Binary32) : Option ℚ := ExecFloat.Binary.toRat? x

end FittingModels
