module

public import Network.Model

/-!
# Network verification

`Architecture.ofList Dᵢ widths Dₒ` describes the architecture the book means:
its depth is the number of hidden widths and its capacity is their sum
(`Architecture.ofList_depth_capacity`). The two-layer network of Section 4.3,
with one input, two layers of three units, and one output, has depth two,
layer sizes `1, 3, 3, 1`, capacity six, and 22 parameters.

Section 4.5.2 counts the scalar parameters of fully connected networks. With one
input and one output, a shallow network of width `D` has `3 * D + 1` parameters
(`parameterCount_shallow`), and `K ≥ 1` hidden layers of width `D` have
`3 * D + 1 + (K - 1) * D * (D + 1)` parameters (`parameterCount_deep`). These counts
hold for every width, without the restriction `D > 2` used in the book's
discussion of linear regions.
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

/-- With no hidden layers, the output layer has `Dₒ` biases and `Dₒ * Dᵢ` weights. -/
@[simp] theorem parameterCount_ofList_nil (Dᵢ Dₒ : ℕ) :
    (ofList Dᵢ [] Dₒ).parameterCount = Dₒ * (Dᵢ + 1) :=
  rfl

/-- Adding a first hidden layer adds its biases and weights to the parameter count. -/
@[simp] theorem parameterCount_ofList_cons (Dᵢ D Dₒ : ℕ) (widths : List ℕ) :
    (ofList Dᵢ (D :: widths) Dₒ).parameterCount =
      D * (Dᵢ + 1) + (ofList D widths Dₒ).parameterCount := by
  unfold parameterCount
  rw [Fin.sum_univ_succ]
  rfl

/-- Section 4.5.2: a one-input, one-output shallow network of width `D` has
`3 * D + 1` parameters. -/
theorem parameterCount_shallow (D : ℕ) :
    (ofList 1 [D] 1).parameterCount = 3 * D + 1 := by
  simp only [parameterCount_ofList_cons, parameterCount_ofList_nil]
  omega

private theorem parameterCount_ofList_replicate_self (D Dₒ K : ℕ) :
    (ofList D (List.replicate K D) Dₒ).parameterCount =
      K * D * (D + 1) + Dₒ * (D + 1) := by
  induction K with
  | zero => simp
  | succ K ih =>
    simp only [List.replicate_succ, parameterCount_ofList_cons, ih, Nat.add_mul,
      Nat.one_mul]
    omega

/-- Section 4.5.2: a one-input, one-output deep network with `K ≥ 1` hidden layers
of width `D` has `3 * D + 1 + (K - 1) * D * (D + 1)` parameters. -/
theorem parameterCount_deep (D K : ℕ) (hK : 1 ≤ K) :
    (ofList 1 (List.replicate K D) 1).parameterCount =
      3 * D + 1 + (K - 1) * D * (D + 1) := by
  obtain ⟨K, rfl⟩ := Nat.exists_eq_succ_of_ne_zero (by omega : K ≠ 0)
  simp only [List.replicate_succ, parameterCount_ofList_cons,
    parameterCount_ofList_replicate_self, Nat.succ_sub_one, Nat.one_mul]
  omega

/-- The two-layer network of Section 4.3. -/
example : (ofList 1 [3, 3] 1).depth = 2 ∧
    ((ofList 1 [3, 3] 1).size 0, (ofList 1 [3, 3] 1).size 1, (ofList 1 [3, 3] 1).size 2,
      (ofList 1 [3, 3] 1).size 3) = (1, 3, 3, 1) ∧
    (ofList 1 [3, 3] 1).capacity = 6 ∧
    (ofList 1 [3, 3] 1).parameterCount = 22 := by
  decide

end NeuralNetworks.Architecture
