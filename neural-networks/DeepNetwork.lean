module

public import DeepNetwork.Model
public import DeepNetwork.WidthThree
public import DeepNetwork.Verification

/-!
# Deep neural networks

Source: Understanding Deep Learning, Sections 4.3–4.4.

- `Model`: deep networks in the matrix notation of Section 4.4, as a composition
  of layers.
- `WidthThree`: the book's network with two hidden layers of three units and
  named parameters, equations (4.7)–(4.9).
- `Verification`: a deep network with one hidden layer is a shallow network, and
  the width-three network expands to equation (4.10).
-/
