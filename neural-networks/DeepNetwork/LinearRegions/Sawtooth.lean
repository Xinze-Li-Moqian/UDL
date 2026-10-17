module

public import DeepNetwork.LinearRegions.Branches
public import Mathlib.LinearAlgebra.Matrix.Notation
public import Mathlib.Tactic.FunProp

/-!
# A width-`D` sawtooth with `D+1` full branches

The first three units use one left-facing ReLU. Further units add alternating
hinges to the right. Before rescaling, the knots are `1,3,5,…,2D-1`: the first
branch has length one and the others length two. Rescaling by `2D+1` puts all
`D+1` branches inside `[0,1]`.

This implements the folding construction behind Section 4.5.2; see also
Serra, Tjandraatmadja, and Ramalingam (2018), Theorem 7 and Appendix D.
-/

@[expose] public section

namespace NeuralNetworks.Sawtooth

open Matrix Set

/-- The unscaled sawtooth, with `n+3` units. -/
noncomputable def raw : ℕ → ℝ → ℝ
  | 0, x => 5 - relu (5 - x) - (3 / 2) * relu (x - 1) + relu (x - 3)
  | n + 1, x => raw n x + (-1 : ℝ)^n * relu (x - (2 * n + 7))

/-- Parameters realizing the unscaled sawtooth with exactly `n+3` units. -/
noncomputable def rawParameters : (n : ℕ) → ShallowNetworkParameters ℝ (n + 3)
  | 0 => ⟨5, ![-3 / 2, 1, -1], ![-1, -3, 5], ![1, 1, -1]⟩
  | n + 1 =>
    { outputBias := (rawParameters n).outputBias
      outputWeights := Fin.cons ((-1 : ℝ)^n) (rawParameters n).outputWeights
      hiddenBiases := Fin.cons (-(2 * n + 7)) (rawParameters n).hiddenBiases
      hiddenWeights := Fin.cons 1 (rawParameters n).hiddenWeights }

theorem rawParameters_eval (n : ℕ) (x : ℝ) :
    shallowNetwork relu x (rawParameters n) = raw n x := by
  induction n with
  | zero =>
    simp [rawParameters, shallowNetwork_eq, hiddenUnit, addBias, Fin.sum_univ_three,
      raw, sub_eq_add_neg, add_comm]
    ring
  | succ n ih =>
    rw [raw, ← ih]
    simp only [shallowNetwork_eq, rawParameters, addBias, Fin.sum_univ_succ,
      Fin.cons_zero, Fin.cons_succ, hiddenUnit, one_mul]
    rw [show -(2 * (n : ℝ) + 7) + x = x - (2 * n + 7) by ring]
    ring

/-- The first affine branch extends to the left. -/
theorem raw_left (n : ℕ) {x : ℝ} (hx : x ≤ 1) : raw n x = x := by
  induction n with
  | zero =>
    rw [raw, relu_eq_max, relu_eq_max, relu_eq_max,
      max_eq_left (by linarith), max_eq_right (by linarith), max_eq_right (by linarith)]
    ring
  | succ n ih =>
    rw [raw, ih, relu_eq_max, max_eq_right (by have := Nat.cast_nonneg (α := ℝ) n; linarith)]
    ring

/-- The remaining branches alternate slope; the last extends to the right. -/
theorem raw_piece (n j : ℕ) (hj : 1 ≤ j) (hjn : j ≤ n + 3) {x : ℝ}
    (hx : 2 * (j : ℝ) - 1 ≤ x) (hupper : j < n + 3 → x ≤ 2 * (j : ℝ) + 1) :
    raw n x = (1 - (-1 : ℝ)^j) / 2 + (-1 : ℝ)^j / 2 * (x - (2 * j - 1)) := by
  induction n generalizing j with
  | zero =>
    have : j = 1 ∨ j = 2 ∨ j = 3 := by omega
    rcases this with rfl | rfl | rfl
    · have hu := hupper (by omega)
      norm_num at hx hu ⊢
      rw [raw, relu_eq_max, relu_eq_max, relu_eq_max,
        max_eq_left (by linarith), max_eq_left (by linarith), max_eq_right (by linarith)]
      ring
    · have hu := hupper (by omega)
      norm_num at hx hu ⊢
      rw [raw, relu_eq_max, relu_eq_max, relu_eq_max,
        max_eq_left (by linarith), max_eq_left (by linarith), max_eq_left (by linarith)]
      ring
    · norm_num at hx ⊢
      rw [raw, relu_eq_max, relu_eq_max, relu_eq_max,
        max_eq_right (by linarith), max_eq_left (by linarith), max_eq_left (by linarith)]
      ring
  | succ n ih =>
    by_cases h : j ≤ n + 3
    · have hu : x ≤ 2 * (n : ℝ) + 7 := by
        have := hupper (by omega)
        have hjr : (j : ℝ) ≤ n + 3 := by exact_mod_cast h
        linarith
      rw [raw, ih j hj h hx (fun hlt => hupper (by omega)), relu_eq_max,
        max_eq_right (by linarith)]
      ring
    · have heq : j = (n + 3) + 1 := by omega
      subst j
      have hx' : 2 * ((n + 3 : ℕ) : ℝ) - 1 ≤ x := by push_cast at hx ⊢; linarith
      have hp := ih (n + 3) (by omega) le_rfl hx' (by omega)
      rw [raw, hp, relu_eq_max, max_eq_left (by push_cast at hx; linarith)]
      simp only [pow_succ, Nat.cast_add, Nat.cast_ofNat]
      ring

/-- The inverse of branch `j` before rescaling the input. -/
noncomputable def rawInv (j : ℕ) (t : ℝ) : ℝ :=
  if j = 0 then t else 2 * j + (-1 : ℝ)^j * (2 * t - 1)

theorem rawInv_zero (t : ℝ) : rawInv 0 t = t := by simp [rawInv]

theorem rawInv_bounds {j : ℕ} (hj : 1 ≤ j) {t : ℝ} (ht : t ∈ Ioo 0 1) :
    2 * (j : ℝ) - 1 < rawInv j t ∧ rawInv j t < 2 * j + 1 := by
  rw [rawInv, ite_eq_right (by omega)]
  rcases neg_one_pow_eq_or ℝ j with h | h <;> rw [h] <;> constructor <;> nlinarith [ht.1, ht.2]

theorem rawInv_eval (n j : ℕ) (hj : j ≤ n + 3) {t : ℝ} (ht : t ∈ Ioo 0 1) :
    raw n (rawInv j t) = t := by
  by_cases hz : j = 0
  · subst j
    rw [rawInv_zero, raw_left n ht.2.le]
  · have hb := rawInv_bounds (by omega : 1 ≤ j) ht
    rw [raw_piece n j (by omega) hj hb.1.le (fun _ => hb.2.le)]
    rw [rawInv, ite_eq_right hz]
    rcases neg_one_pow_eq_or ℝ j with h | h <;> rw [h] <;> ring

theorem rawInv_order {i j : ℕ} (hij : i < j) {s t : ℝ}
    (hs : s ∈ Ioo 0 1) (ht : t ∈ Ioo 0 1) : rawInv i s < rawInv j t := by
  have hj := (rawInv_bounds (by omega : 1 ≤ j) ht).1
  by_cases hi : i = 0
  · subst i
    rw [rawInv_zero]
    have hjr : (1 : ℝ) ≤ j := by exact_mod_cast (show 1 ≤ j by omega)
    linarith [hs.2]
  · have hi' := (rawInv_bounds (by omega : 1 ≤ i) hs).2
    have hijr : (i : ℝ) + 1 ≤ j := by exact_mod_cast hij
    linarith

/-- The sawtooth rescaled to have `n+4` full branches inside `[0,1]`. -/
noncomputable def parameters (n : ℕ) : ShallowNetworkParameters ℝ (n + 3) :=
  { rawParameters n with
    hiddenWeights := fun d => (rawParameters n).hiddenWeights d * (2 * n + 7) }

theorem parameters_eval (n : ℕ) (x : ℝ) :
    shallowNetwork relu x (parameters n) = raw n ((2 * n + 7) * x) := by
  rw [← rawParameters_eval]
  simp [parameters, shallowNetwork_eq, hiddenUnit, addBias, mul_assoc]

/-- Every one of the `D+1` branches covers `(0,1)`; different branches are disjoint. -/
noncomputable def branches (n : ℕ) :
    FullBranches (fun x => shallowNetwork relu x (parameters n)) (Fin (n + 4)) where
  inv j t := rawInv j t / (2 * n + 7)
  continuous j := by unfold rawInv; split_ifs <;> fun_prop
  mapsTo j t ht := by
    have hL : (0 : ℝ) < 2 * n + 7 := by positivity
    constructor
    · apply div_pos _ hL
      by_cases hj : (j : ℕ) = 0
      · simpa [hj, rawInv_zero] using ht.1
      · have hb := (rawInv_bounds (by omega : 1 ≤ (j : ℕ)) ht).1
        have hjr : (1 : ℝ) ≤ j := by exact_mod_cast (show 1 ≤ (j : ℕ) by omega)
        linarith
    · apply (div_lt_one hL).mpr
      by_cases hj : (j : ℕ) = 0
      · rw [hj, rawInv_zero]
        have := Nat.cast_nonneg (α := ℝ) n
        linarith [ht.2]
      · have hb := (rawInv_bounds (by omega : 1 ≤ (j : ℕ)) ht).2
        have hjr : (j : ℝ) ≤ n + 3 := by exact_mod_cast (show (j : ℕ) ≤ n + 3 by omega)
        linarith
  right_inv j t ht := by
    rw [parameters_eval, mul_div_cancel₀ _ (by positivity : (2 * (n : ℝ) + 7) ≠ 0)]
    exact rawInv_eval n j (by omega) ht
  disjoint i j hij s hs t ht := by
    intro heq
    have hL : (2 * (n : ℝ) + 7) ≠ 0 := by positivity
    have heq' := (div_left_inj' hL).mp heq
    rcases lt_or_gt_of_ne (show (i : ℕ) ≠ j from fun h => hij (Fin.ext h)) with h | h
    · exact (ne_of_lt (rawInv_order h hs ht)) heq'
    · exact (ne_of_gt (rawInv_order h ht hs)) heq'

end NeuralNetworks.Sawtooth
