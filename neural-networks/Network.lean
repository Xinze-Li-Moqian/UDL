module

public import Network.Model
public import Network.Verification

/-!
# Neural networks in matrix notation

Source: Understanding Deep Learning, Sections 4.4 and 4.5.2.

The common foundation for every network in this repository. Vectors and matrices
come from Mathlib; this folder defines the book's network terms on top of them.

- `Model`: architectures (depth, layer sizes, capacity, parameter count), parameters
  `φ = {β_k, Ω_k}`, hidden layers `h_k`, and the output `y`, with `single` and
  `cons` for building networks layer by layer.
- `Verification`: `Architecture.ofList` has the expected depth and capacity;
  parameter counts agree with the shallow and deep formulas of Section 4.5.2.
-/
