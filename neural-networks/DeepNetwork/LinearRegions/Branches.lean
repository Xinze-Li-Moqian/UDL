module

public import DeepNetwork.LinearRegions.Pieces
public import Mathlib.Topology.Instances.Real.Lemmas
public import Mathlib.Order.Interval.Set.Infinite

/-!
# Full inverse branches and a lower bound on affine pieces

Disjoint continuous inverse branches over `(0,1)` give a lower bound on the
number of affine pieces. Such branches compose by taking pairs of indices.
The lower bound concerns the output function, not its activation patterns.
-/

@[expose] public section

namespace NeuralNetworks

open Set Filter
open scoped Topology

/-- Disjoint continuous inverse branches, each lying in `(0,1)` and covering
the entire output interval `(0,1)`. -/
structure FullBranches (f : ℝ → ℝ) (ι : Type) where
  inv : ι → ℝ → ℝ
  continuous : ∀ i, Continuous (inv i)
  mapsTo : ∀ i, MapsTo (inv i) (Ioo 0 1) (Ioo 0 1)
  right_inv : ∀ i t, t ∈ Ioo 0 1 → f (inv i t) = t
  disjoint : ∀ i j, i ≠ j → ∀ s ∈ Ioo 0 1, ∀ t ∈ Ioo 0 1, inv i s ≠ inv j t

namespace FullBranches

/-- Relabel a family of branches by an equivalent finite index type. -/
def reindex {f : ℝ → ℝ} {ι κ : Type} (F : FullBranches f ι) (e : κ ≃ ι) :
    FullBranches f κ where
  inv i := F.inv (e i)
  continuous i := F.continuous (e i)
  mapsTo i := F.mapsTo (e i)
  right_inv i := F.right_inv (e i)
  disjoint i j h := F.disjoint (e i) (e j) (fun he => h (e.injective he))

/-- Inverse branches multiply under composition. -/
def comp {f g : ℝ → ℝ} {ι κ : Type} (F : FullBranches f ι) (G : FullBranches g κ) :
    FullBranches (g ∘ f) (ι × κ) where
  inv p := F.inv p.1 ∘ G.inv p.2
  continuous p := (F.continuous p.1).comp (G.continuous p.2)
  mapsTo p := (F.mapsTo p.1).comp (G.mapsTo p.2)
  right_inv p t ht := by
    dsimp
    rw [F.right_inv _ _ (G.mapsTo _ ht), G.right_inv _ _ ht]
  disjoint p q hpq s hs t ht := by
    dsimp
    by_cases h : p.1 = q.1
    · have h' : p.2 ≠ q.2 := fun h' => hpq (Prod.ext h h')
      intro heq
      have := congrArg f heq
      rw [F.right_inv _ _ (G.mapsTo _ hs), F.right_inv _ _ (G.mapsTo _ ht)] at this
      exact G.disjoint _ _ h' s hs t ht this
    · exact F.disjoint _ _ h _ (G.mapsTo _ hs) _ (G.mapsTo _ ht)

private theorem cell_mem_nhds (S : Finset ℝ) (x : ℝ) (hx : x ∉ S) :
    AffinePieces.cell S (S.filter (· < x)).card ∈ 𝓝 x := by
  classical
  have hn : ∀ s ∈ S, ∀ᶠ z in 𝓝 x, (s < x ↔ s < z) ∧ (s ≤ x ↔ s ≤ z) := by
    intro s hs
    have hne : s ≠ x := fun h => hx (h ▸ hs)
    rcases lt_or_gt_of_ne hne with h | h
    · filter_upwards [Ioi_mem_nhds h] with z hz
      exact ⟨iff_of_true h hz, iff_of_true h.le hz.le⟩
    · filter_upwards [Iio_mem_nhds h] with z hz
      exact ⟨iff_of_false (not_lt_of_ge h.le) (not_lt_of_ge hz.le),
        iff_of_false (not_le_of_gt h) (not_le_of_gt hz)⟩
  filter_upwards [(Filter.eventually_all_finset S).mpr hn] with z hz
  have hlt : S.filter (· < z) = S.filter (· < x) := by
    ext s
    simpa only [Finset.mem_filter] using and_congr_right (fun hs => (hz s hs).1.symm)
  have hle : S.filter (· ≤ z) = S.filter (· ≤ x) := by
    ext s
    simpa only [Finset.mem_filter] using and_congr_right (fun hs => (hz s hs).2.symm)
  change (S.filter (· < z)).card ≤ _ ∧ _ ≤ (S.filter (· ≤ z)).card
  rw [hlt, hle]
  refine ⟨le_rfl, Finset.card_le_card ?_⟩
  intro s hs
  exact Finset.mem_filter.mpr ⟨(Finset.mem_filter.mp hs).1, (Finset.mem_filter.mp hs).2.le⟩

/-- Every affine partition needs at least as many pieces as full branches. -/
theorem card_le {f : ℝ → ℝ} {ι : Type} [Fintype ι] (F : FullBranches f ι)
    (S : Finset ℝ) (hf : AffinePieces f S) : Fintype.card ι ≤ S.card + 1 := by
  classical
  obtain ⟨y, hy, hyS⟩ := (Set.Ioo_infinite (show (0 : ℝ) < 1 by norm_num)).exists_notMem_finset
    (S.image f)
  have hnot : ∀ i, F.inv i y ∉ S := by
    intro i hi
    apply hyS
    exact Finset.mem_image.mpr ⟨F.inv i y, hi, F.right_inv i y hy⟩
  let rank : ι → Fin (S.card + 1) := fun i =>
    ⟨(S.filter (· < F.inv i y)).card, Nat.lt_succ_of_le (Finset.card_filter_le _ _)⟩
  have hmem : ∀ i, F.inv i y ∈ AffinePieces.cell S (rank i) :=
    fun i => mem_of_mem_nhds (cell_mem_nhds S _ (hnot i))
  have hinj : Function.Injective rank := by
    intro i j hij
    obtain ⟨m, c, hmc⟩ := hf.on_cell (rank i)
    have hi := hmc _ (hmem i)
    rw [F.right_inv i y hy] at hi
    have hm : m ≠ 0 := by
      intro hm
      have hn : {t | F.inv i t ∈ AffinePieces.cell S (rank i)} ∈ 𝓝 y :=
        (F.continuous i).continuousAt.preimage_mem_nhds (cell_mem_nhds S _ (hnot i))
      obtain ⟨a, b, hab, hsub⟩ := mem_nhds_iff_exists_Ioo_subset.mp
        (Filter.inter_mem hn (Ioo_mem_nhds hy.1 hy.2))
      obtain ⟨t, hyt, htb⟩ := exists_between hab.2
      have ht := hsub ⟨lt_trans hab.1 hyt, htb⟩
      have hmt := hmc _ ht.1
      rw [F.right_inv i t ht.2] at hmt
      simp only [hm, zero_mul, zero_add] at hi hmt
      linarith
    have hj := hmc _ (hij ▸ hmem j)
    rw [F.right_inv j y hy] at hj
    have heq : F.inv i y = F.inv j y := by
      apply mul_left_cancel₀ hm
      linarith
    by_contra hne
    exact F.disjoint i j hne y hy y hy heq
  simpa using Fintype.card_le_of_injective rank hinj

end FullBranches

end NeuralNetworks
