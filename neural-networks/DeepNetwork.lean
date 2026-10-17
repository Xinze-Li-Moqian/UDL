module

public import Network
public import DeepNetwork.FromComposition
public import DeepNetwork.WidthThree
public import DeepNetwork.Verification

/-!
# Deep neural networks

Source: Understanding Deep Learning, Sections 4.2–4.4.

The general deep network of equation (4.15) is the foundation in `Network`.
This folder holds the book's specific deep networks, their checks, and the link
from composed networks.

- `FromComposition`: composing shallow networks of widths `[D₁, …, D_K]` gives a
  deep network with those hidden widths (Section 4.2, equations 4.5–4.6).
- `WidthThree`: the book's network with two hidden layers of three units and
  named parameters, equations (4.7)–(4.9).
- `Verification`: the width-three network expands to equation (4.10).

Shallow networks are the foundation's networks with one hidden layer; see
`ShallowNetwork.Model` and `ShallowNetwork.MultiInput`.
-/
