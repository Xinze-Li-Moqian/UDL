module

public import ShallowNetwork.LinearRegions.ActivationPattern
public import Mathlib.Combinatorics.SetFamily.Shatter

/-!
# Linear regions with multiple inputs

Source: Understanding Deep Learning, Section 3.3.

With `Dᵢ` inputs, each hidden unit switches on or off across a hyperplane
`θ_d0 + ∑ i, θ_di * x_i = 0`. A network of width `D` has at most
`∑ j ≤ Dᵢ, (D choose j)` linear regions: for `Dᵢ = 1` this is `D + 1`, for
`Dᵢ = 2` it is `1 + D + D * (D - 1) / 2`, and it grows like `D ^ Dᵢ` for large `D`.

`LinearRegionBound` makes "region" precise through activation patterns: the
inputs at which the same hidden units are active. Every input has a pattern in a
family `𝒜` of at most that many patterns, and the network is affine on each
one. Each pattern's inputs form an intersection of half-spaces, hence a convex
set, so `ℝ^Dᵢ` splits into at most that many convex pieces with the network
affine on each.

The proof does not count cells of the hyperplane arrangement directly. Instead:

1. The realized patterns form a family of subsets of the `D` hidden units.
2. No `Dᵢ + 1` units realize all their on/off combinations
   (`not_all_patterns`, in `ActivationPattern`), so the family's VC dimension is at
   most `Dᵢ`.
3. The Sauer–Shelah lemma bounds a family of VC dimension at most `Dᵢ` by
   `∑ j ≤ Dᵢ, (D choose j)`.
-/

@[expose] public section

namespace NeuralNetworks.MultiInput

/-- The linear region bound for shallow ReLU networks with `Dᵢ` inputs. -/
def LinearRegionBound : Prop :=
  ∀ (Dᵢ D : ℕ) (Φ : Parameters ℝ Dᵢ D),
    ∃ 𝒜 : Finset (Finset (Fin D)),
      𝒜.card ≤ ∑ j ∈ Finset.range (Dᵢ + 1), D.choose j ∧
      (∀ x, activationPattern Φ x ∈ 𝒜) ∧
      ∀ P ∈ 𝒜, ∃ (m : Fin Dᵢ → ℝ) (c : ℝ), ∀ x, activationPattern Φ x = P →
        network relu x Φ = ∑ i, m i * x i + c

/-- **Linear region bound** for shallow ReLU networks with `Dᵢ` inputs. -/
theorem linearRegionBound : LinearRegionBound := by
  intro Dᵢ D Φ
  classical
  set 𝒜 := Finset.univ.filter fun P => ∃ x, activationPattern Φ x = P
  have hvc : 𝒜.vcDim ≤ Dᵢ := by
    apply Finset.sup_le
    intro T hT
    rw [Finset.mem_shatterer] at hT
    by_contra hlt
    obtain ⟨T', hT'T, hcard⟩ := Finset.exists_subset_card_eq (s := T) (n := Dᵢ + 1) (by omega)
    apply not_all_patterns Φ T' hcard
    intro U hU
    obtain ⟨u, hu, hTu⟩ := hT.mono_right hT'T hU
    obtain ⟨x, rfl⟩ := (Finset.mem_filter.mp hu).2
    exact ⟨x, hTu⟩
  refine ⟨𝒜, ?_, fun x => by simp [𝒜], fun P _ => affine_on_pattern Φ P⟩
  calc 𝒜.card ≤ 𝒜.shatterer.card := Finset.card_le_card_shatterer _
    _ ≤ ∑ k ∈ Finset.Iic 𝒜.vcDim, (Fintype.card (Fin D)).choose k :=
      Finset.card_shatterer_le_sum_vcDim
    _ ≤ ∑ j ∈ Finset.range (Dᵢ + 1), D.choose j := by
      rw [Fintype.card_fin]
      apply Finset.sum_le_sum_of_subset
      intro k hk
      simp only [Finset.mem_Iic, Finset.mem_range] at hk ⊢
      omega

end NeuralNetworks.MultiInput
