module

public import ShallowNetwork.LinearRegions.OneInput
public import Mathlib.Data.Finset.Card

/-!
# Affine pieces on the real line

A finite set of joints certifies an upper bound on the number of affine pieces.
An exact count additionally requires that no certificate uses fewer joints.
The closed cells below include their boundary joints and are indexed by the
number of joints to their left; this avoids choosing an ordering of the finset.
-/

@[expose] public section

namespace NeuralNetworks

/-- `f` is affine on every closed interval with no joint in its interior. -/
def AffinePieces (f : ℝ → ℝ) (S : Finset ℝ) : Prop :=
  ∀ a b : ℝ, (∀ s ∈ S, s ∉ Set.Ioo a b) →
    ∃ m c : ℝ, ∀ x ∈ Set.Icc a b, f x = m * x + c

/-- The least number of affine pieces is `N`; redundant subdivision cannot
increase this count. -/
def HasExactlyRegions (f : ℝ → ℝ) (N : ℕ) : Prop :=
  (∃ S : Finset ℝ, AffinePieces f S ∧ S.card + 1 = N) ∧
  ∀ S : Finset ℝ, AffinePieces f S → N ≤ S.card + 1

namespace AffinePieces

open Finset

/-- The closed cell numbered `i`, counting joints from the left. -/
noncomputable def cell (S : Finset ℝ) (i : ℕ) : Set ℝ :=
  {x | (S.filter (· < x)).card ≤ i ∧ i ≤ (S.filter (· ≤ x)).card}

theorem cell_no_joint {S : Finset ℝ} {i : ℕ} {a b : ℝ}
    (ha : a ∈ cell S i) (hb : b ∈ cell S i) :
    ∀ s ∈ S, s ∉ Set.Ioo a b := by
  classical
  intro s hs h
  have hsub : S.filter (· ≤ a) ⊂ S.filter (· < b) := by
    refine Finset.ssubset_iff_subset_ne.mpr ⟨?_, ?_⟩
    · intro t ht
      simp only [mem_filter] at ht ⊢
      exact ⟨ht.1, lt_of_le_of_lt ht.2 (lt_trans h.1 h.2)⟩
    · intro heq
      have : s ∈ S.filter (· ≤ a) := heq ▸ (mem_filter.mpr ⟨hs, h.2⟩)
      exact (not_le_of_gt h.1) (mem_filter.mp this).2
  have := Finset.card_lt_card hsub
  exact (not_lt_of_ge (le_trans hb.1 ha.2)) this

theorem interval_in_cell (S : Finset ℝ) (a b : ℝ)
    (h : ∀ s ∈ S, s ∉ Set.Ioo a b) :
    ∃ i : Fin (S.card + 1), Set.Icc a b ⊆ cell S i := by
  classical
  refine ⟨⟨(S.filter (· ≤ a)).card, Nat.lt_succ_of_le (card_filter_le _ _)⟩, ?_⟩
  intro x hx
  constructor
  · apply card_le_card
    intro s hs
    obtain ⟨hs, hsx⟩ := mem_filter.mp hs
    refine mem_filter.mpr ⟨hs, ?_⟩
    by_contra hn
    exact h s hs ⟨lt_of_not_ge hn, lt_of_lt_of_le hsx hx.2⟩
  · apply card_le_card
    intro s hs
    obtain ⟨hs, hsa⟩ := mem_filter.mp hs
    exact mem_filter.mpr ⟨hs, le_trans hsa hx.1⟩

/-- Local affine formulas on intervals give one formula on each entire cell. -/
theorem on_cell {f : ℝ → ℝ} {S : Finset ℝ} (hf : AffinePieces f S) (i : ℕ) :
    ∃ m c : ℝ, ∀ x ∈ cell S i, f x = m * x + c := by
  classical
  by_cases hab : ∃ a ∈ cell S i, ∃ b ∈ cell S i, a < b
  · obtain ⟨a, ha, b, hb, hab⟩ := hab
    obtain ⟨m, c, hmc⟩ := hf a b (cell_no_joint ha hb)
    refine ⟨m, c, fun x hx => ?_⟩
    have hlo : min a x ∈ cell S i := by
      rcases le_total a x with h | h
      · simpa [min_eq_left h] using ha
      · simpa [min_eq_right h] using hx
    have hhi : max b x ∈ cell S i := by
      rcases le_total b x with h | h
      · simpa [max_eq_right h] using hx
      · simpa [max_eq_left h] using hb
    obtain ⟨m', c', hmc'⟩ := hf (min a x) (max b x) (cell_no_joint hlo hhi)
    have hma := hmc a ⟨le_rfl, hab.le⟩
    have hmb := hmc b ⟨hab.le, le_rfl⟩
    have hma' := hmc' a ⟨min_le_left _ _, le_trans hab.le (le_max_left _ _)⟩
    have hmb' := hmc' b ⟨le_trans (min_le_left _ _) hab.le, le_max_left _ _⟩
    have hmm : m' = m := by nlinarith
    have hcc : c' = c := by rw [hmm] at hma'; linarith
    simpa [hmm, hcc] using hmc' x ⟨min_le_right _ _, le_max_right _ _⟩
  · by_cases hn : (cell S i).Nonempty
    · obtain ⟨a, ha⟩ := hn
      refine ⟨0, f a, fun x hx => ?_⟩
      have hxa : x = a := by
        by_contra hne
        rcases lt_or_gt_of_ne hne with h | h
        · exact hab ⟨x, hx, a, ha, h⟩
        · exact hab ⟨a, ha, x, hx, h⟩
      simp [hxa]
    · exact ⟨0, 0, fun x hx => (hn ⟨x, hx⟩).elim⟩

/-- Affine combinations preserve a common partition. -/
theorem affine_combination {n : ℕ} {f : ℝ → Fin n → ℝ} {S : Finset ℝ}
    (hf : ∀ d, AffinePieces (fun x => f x d) S) (b : ℝ) (w : Fin n → ℝ) :
    AffinePieces (fun x => b + ∑ d, w d * f x d) S := by
  intro a z hS
  choose m c h using fun d => hf d a z hS
  refine ⟨∑ d, w d * m d, b + ∑ d, w d * c d, fun x hx => ?_⟩
  simp only [h _ x hx, mul_add, Finset.sum_add_distrib, Finset.sum_mul, mul_assoc]
  ring

/-- `D` ReLUs add at most `D` possible joints per old cell. -/
theorem relu_refine {D : ℕ} {f : ℝ → Fin D → ℝ} {S : Finset ℝ}
    (hf : ∀ d, AffinePieces (fun x => f x d) S) :
    ∃ T : Finset ℝ, T.card + 1 ≤ (S.card + 1) * (D + 1) ∧
      ∀ d, AffinePieces (fun x => relu (f x d)) T := by
  classical
  choose m c hmc using fun (i : Fin (S.card + 1)) (d : Fin D) => (hf d).on_cell i
  let R : Finset ℝ := Finset.univ.image
    (fun p : Fin (S.card + 1) × Fin D => -c p.1 p.2 / m p.1 p.2)
  refine ⟨S ∪ R, ?_, ?_⟩
  · have hR : R.card ≤ (S.card + 1) * D := by
      calc R.card ≤ (Finset.univ : Finset (Fin (S.card + 1) × Fin D)).card :=
          Finset.card_image_le
        _ = (S.card + 1) * D := by simp
    have hu := Finset.card_union_le S R
    nlinarith
  · intro d a b hT
    obtain ⟨i, hi⟩ := interval_in_cell S a b
      (fun s hs => hT s (Finset.mem_union_left R hs))
    obtain ⟨u, v, huv⟩ := hiddenUnit_affine (c i d) (m i d) a b (fun _ =>
      hT _ (Finset.mem_union_right S (Finset.mem_image.mpr ⟨(i, d), Finset.mem_univ _, rfl⟩)))
    refine ⟨u, v, fun x hx => ?_⟩
    change relu (f x d) = u * x + v
    rw [hmc i d x (hi hx)]
    simpa [hiddenUnit, addBias, add_comm] using huv x hx

end AffinePieces

end NeuralNetworks
