module

/-!
# Supervised learning

Terminology from Simon J. D. Prince, Understanding Deep Learning
(February 8, 2026), Section 2.1, equations (2.1)-(2.2).

`Scalar` is the type of numeric values (typically real numbers).
`Parameters` is the type of model parameters `φ`.
Both are left abstract so these definitions use only Lean's core library.
-/

@[expose] public section

namespace SupervisedLearning

/-- An input `x`: a vector of a predetermined size, with a fixed coordinate order. -/
abbrev Input (Scalar : Type) (inputSize : Nat) :=
  Vector Scalar inputSize

/-- An output `y`: a vector of a predetermined size, with a fixed coordinate order. -/
abbrev Output (Scalar : Type) (outputSize : Nat) :=
  Vector Scalar outputSize

/-- A model `f`: its parameters `φ` select the mapping from input to output.
The book's `f[x, φ]` is written `f x φ` in Lean. -/
abbrev Model (Scalar Parameters : Type) (inputSize outputSize : Nat) :=
  Input Scalar inputSize → Parameters → Output Scalar outputSize

/-- A training dataset of `I` pairs `(xᵢ, yᵢ)`.
For each index `i : Fin I`, `x i` and `y i` form one input/output pair. -/
structure TrainingDataset (Scalar : Type) (inputSize outputSize I : Nat) where
  x : Fin I → Input Scalar inputSize
  y : Fin I → Output Scalar outputSize

variable {Scalar Parameters : Type} {inputSize outputSize : Nat}

/-- Inference: compute the prediction `y = f[x, φ]` from an input `x`. -/
def inference (f : Model Scalar Parameters inputSize outputSize)
    (x : Input Scalar inputSize) (φ : Parameters) : Output Scalar outputSize :=
  f x φ

/-- With the model and training dataset fixed, the loss `L[φ]` is a scalar
that measures mismatch between predictions and training outputs.
This gives its type; the choice of loss formula is left open. -/
abbrev LossFunction (Scalar Parameters : Type) :=
  Parameters → Scalar

end SupervisedLearning
