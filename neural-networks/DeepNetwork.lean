module

public import Network
public import DeepNetwork.WidthThree
public import DeepNetwork.Verification

/-!
# Deep neural networks

Source: Understanding Deep Learning, Sections 4.3–4.4.

The general deep network of equation (4.15) is the foundation in `Network`.
This folder holds the book's specific deep networks and their checks.

- `WidthThree`: the book's network with two hidden layers of three units and
  named parameters, equations (4.7)–(4.9).
- `Verification`: the width-three network expands to equation (4.10).

Shallow networks are the foundation's networks with one hidden layer; see
`ShallowNetwork.Model` and `ShallowNetwork.MultiInput`.
-/
