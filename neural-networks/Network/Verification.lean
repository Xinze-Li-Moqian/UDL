module

public import Network.Model

/-!
# Network verification

`Architecture.ofList Dᵢ widths Dₒ` describes the architecture the book means:
its depth is the number of hidden widths and its capacity is their sum
(`Architecture.ofList_depth_capacity`). The two-layer network of Section 4.3,
with one input, two layers of three units, and one output, has depth two,
layer sizes `1, 3, 3, 1`, and capacity six.
-/

@[expose] public section

namespace NeuralNetworks.Architecture

/-- `ofList` has depth `K` and capacity `D₁ + ⋯ + D_K`. -/
theorem ofList_depth_capacity (Dᵢ : ℕ) (widths : List ℕ) (Dₒ : ℕ) :
    (ofList Dᵢ widths Dₒ).depth = widths.length ∧
      (ofList Dᵢ widths Dₒ).capacity = widths.sum := by
  refine ⟨rfl, ?_⟩
  rw [← Fin.sum_univ_getElem widths]
  refine Finset.sum_congr rfl fun k _ => ?_
  have hk : (k : ℕ) < widths.length := k.2
  show (widths ++ [Dₒ])[(k : ℕ)]?.getD 0 = widths[(k : ℕ)]
  rw [List.getElem?_append_left hk, List.getElem?_eq_getElem hk]
  rfl

/-- The two-layer network of Section 4.3. -/
example : (ofList 1 [3, 3] 1).depth = 2 ∧
    ((ofList 1 [3, 3] 1).size 0, (ofList 1 [3, 3] 1).size 1, (ofList 1 [3, 3] 1).size 2,
      (ofList 1 [3, 3] 1).size 3) = (1, 3, 3, 1) ∧
    (ofList 1 [3, 3] 1).capacity = 6 := by
  decide

end NeuralNetworks.Architecture
