module

public import Discriminative.Model
public import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
public import Mathlib.Analysis.SpecialFunctions.Exp

@[expose] public section

namespace SupervisedLearning.Discriminative.Gabor

/-- The two parameters in equation (6.8). -/
structure Parameters where
  /-- The phase parameter φ₀. -/
  φ₀ : ℝ
  /-- The horizontal scaling parameter φ₁; the book uses positive values. -/
  φ₁ : ℝ

/-- Understanding Deep Learning, Section 6.1.2, equation (6.8).

    z = φ₀ + 0.06 φ₁ x,
    f[x, φ] = sin(z) exp(−z² / 32).

Use the shared one-input, one-output model interface. `x[0]` is the scalar
input, and `#v[...]` packages the scalar prediction as a one-element vector.
-/
noncomputable def gaborModel : Model ℝ Parameters 1 1 :=
  fun x φ =>
    let z := φ.φ₀ + (3 / 50) * φ.φ₁ * x[0]
    #v[Real.sin z * Real.exp (-(z ^ 2) / 32)]

end SupervisedLearning.Discriminative.Gabor
