module

public import SupervisedLearning

/-!
# Discriminative models

The discriminative approach directly predicts the output from the input:
`y = f[x, φ]`. The training pairs retain their original `(x, y)` roles.
-/

@[expose] public section

namespace SupervisedLearning.Discriminative

/-- A discriminative model `f`: the book's `f[x, φ]` is written `f x φ` in Lean. -/
abbrev Model (Scalar Parameters : Type) (inputSize outputSize : Nat) :=
  Input Scalar inputSize → Parameters → Output Scalar outputSize

/-- Inference: apply the fitted model to an input to predict its output. -/
def inference {Scalar Parameters : Type} {inputSize outputSize : Nat}
    (f : Model Scalar Parameters inputSize outputSize)
    (x : Input Scalar inputSize) (φ : Parameters) : Output Scalar outputSize :=
  f x φ

end SupervisedLearning.Discriminative
