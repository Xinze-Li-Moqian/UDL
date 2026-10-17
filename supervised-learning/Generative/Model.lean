module

public import SupervisedLearning

/-!
# Generative models

The deterministic generative approach in Chapter 2 of Understanding Deep Learning
models the measurements as `x = g[y, φ]`.

The training pairs keep their original roles: `x` is the observed input and `y`
is the desired output. The model maps from `y` to `x`; predicting `y` from a new
`x` requires solving this relation. A general model need not have an inverse.
-/

@[expose] public section

namespace SupervisedLearning.Generative

/-- A generative model `g`: the book's `g[y, φ]` is written `g y φ` in Lean. -/
abbrev Model (Scalar Parameters : Type) (inputSize outputSize : Nat) :=
  Output Scalar outputSize → Parameters → Input Scalar inputSize

end SupervisedLearning.Generative
