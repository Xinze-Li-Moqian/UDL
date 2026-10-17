module

public import ComposedNetwork.Model
public import ComposedNetwork.WidthThree
public import ComposedNetwork.Verification

/-!
# Composed shallow networks

Source: Understanding Deep Learning, Section 4.1.

- `Model`: shallow networks applied in order, each output feeding the next.
- `WidthThree`: the book's composition of two width-three networks.
- `Verification`: the width-three composition is a special case of the general one.
-/
