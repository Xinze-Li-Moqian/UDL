module

public import LossFunctions.EmpiricalLoss

/-!
# Supervised learning

Shared terminology from Simon J. D. Prince, Understanding Deep Learning
(February 8, 2026), Chapter 2. Model directions are defined separately in
`Discriminative/Model.lean` and `Generative/Model.lean`.

`Scalar` is the type of numeric values (typically real numbers).
`Parameters` is the type of model parameters `φ`.
Both are left abstract; the shared loss construction comes from `LossFunctions.EmpiricalLoss`.
-/

@[expose] public section

namespace SupervisedLearning

/-- An input `x`: a vector of a predetermined size, with a fixed coordinate order. -/
abbrev Input (Scalar : Type) (inputSize : Nat) :=
  Vector Scalar inputSize

/-- An output `y`: a vector of a predetermined size, with a fixed coordinate order. -/
abbrev Output (Scalar : Type) (outputSize : Nat) :=
  Vector Scalar outputSize

/-- A training dataset of `I` pairs `(xᵢ, yᵢ)`.
For each index `i : Fin I`, `x i` and `y i` form one input/output pair. -/
structure TrainingDataset (Scalar : Type) (inputSize outputSize I : Nat) where
  x : Fin I → Input Scalar inputSize
  y : Fin I → Output Scalar outputSize

variable {Scalar Parameters : Type}

/-- The shared loss-function type, retained here for Chapter 2 code. -/
abbrev LossFunction (Scalar Parameters : Type) :=
  LossFunctions.LossFunction Scalar Parameters

end SupervisedLearning
