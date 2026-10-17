module

public import DeepNetwork.LinearRegions.UpperBound
public import DeepNetwork.LinearRegions.Sawtooth
public import DeepNetwork.FromComposition

/-!
# Attaining the one-dimensional region bound

Source: Understanding Deep Learning, Section 4.5.2.

Compose `K` copies of the width-`D` sawtooth. Its `D+1` inverse branches
multiply under composition, so any affine partition needs at least `(D+1)^K`
pieces. Section 4.2 converts the composition to a deep network of precisely
the requested architecture. The upper bound then proves the count is exact.
-/

@[expose] public section

namespace NeuralNetworks

namespace Sawtooth

/-- `K` copies of the width-`n+3` sawtooth have `(n+4)^K` disjoint full branches. -/
noncomputable def iteratedBranches (n K : ℕ) :
    FullBranches
      (composedNetwork relu (List.replicate K ⟨n + 3, parameters n⟩))
      (Fin ((n + 4)^K)) := by
  induction K with
  | zero =>
    change FullBranches (fun x : ℝ => x) (Fin 1)
    refine {
      inv := fun _ x => x
      continuous := fun _ => continuous_id
      mapsTo := fun _ _ hx => hx
      right_inv := fun _ _ _ => rfl
      disjoint := ?_ }
    intro i j h
    exact (h (Subsingleton.elim i j)).elim
  | succ K ih =>
    exact ((branches n).comp ih).reindex (Fintype.equivOfCardEq (by
      simp [pow_succ, Nat.mul_comm]))

end Sawtooth

/-- **Section 4.5.2, attainability:** for every width `D ≥ 3` and depth `K`,
some one-input, one-output ReLU network has exactly `(D+1)^K` affine regions.
This also includes depth zero, when the witness is the identity. -/
theorem deepLinearRegionBound_attained (D K : ℕ) (hD : 3 ≤ D) :
    ∃ Φ : NetworkParameters ℝ (Architecture.ofList 1 (List.replicate K D) 1),
      ∀ j, HasExactlyRegions (fun x => network relu Φ (fun _ : Fin 1 => x) j)
        ((D + 1)^K) := by
  obtain ⟨n, rfl⟩ := Nat.exists_eq_add_of_le hD
  let Ns : List (Σ D, ShallowNetworkParameters ℝ D) :=
    List.replicate K ⟨n + 3, Sawtooth.parameters n⟩
  have hfamily : composedNetwork relu Ns ∈ composedFamily relu (List.replicate K (3 + n)) := by
    refine ⟨Ns, ?_, rfl⟩
    simp [Ns, Nat.add_comm]
  obtain ⟨g, ⟨Φ, rfl⟩, hg⟩ := ComposedNetwork.composedFamily_subset relu _ _ hfamily
  refine ⟨Φ, fun j => ?_⟩
  have heq : (fun x => network relu Φ (fun _ : Fin 1 => x) j) = composedNetwork relu Ns :=
    funext (fun x => hg x j)
  have hlower : ∀ T, AffinePieces (fun x => network relu Φ (fun _ : Fin 1 => x) j) T →
      (3 + n + 1)^K ≤ T.card + 1 := by
    intro T hT
    rw [heq] at hT
    have := (Sawtooth.iteratedBranches n K).card_le T hT
    simpa only [Fintype.card_fin, show n + 4 = 3 + n + 1 by omega] using this
  obtain ⟨S, hS, hf⟩ := deepLinearRegionBound_uniform (3 + n) K Φ
  exact ⟨⟨S, hf j, Nat.le_antisymm hS (hlower S (hf j))⟩, hlower⟩

end NeuralNetworks
