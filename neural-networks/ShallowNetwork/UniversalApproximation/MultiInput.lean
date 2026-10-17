module

public import ShallowNetwork.UniversalApproximation.Ridge

/-!
# Universal approximation with multiple inputs

Source: Understanding Deep Learning, Sections 3.2–3.4.

Every continuous real-valued function on a compact subset of `ℝ^Dᵢ` can be
approximated uniformly by a shallow ReLU network with `Dᵢ` inputs. With
`Dᵢ = 2`, the target's graph is a surface and the network's graph is a surface
made of flat polygonal pieces.

The one-input proof interpolates on a grid, but that idea does not carry over:
each hidden unit is a ridge function, constant along hyperplanes, so a shallow
network cannot build a localized bump. The proof instead reduces to one input:

1. Extend `g` from `K` to a continuous function `G` on all of `ℝ^Dᵢ` (Tietze).
2. By Stone–Weierstrass, `G` is within `ε / 2` on `K` of a linear combination of
   exponential ridges `exp (∑ i, w i * x i)` (`Ridge`).
3. Each exponential ridge is `exp` composed with a linear map. The one-input
   theorem approximates `exp` on the compact image of `K`, giving a network.
4. Summing those networks gives a network within `ε / 2` of the combination,
   hence within `ε` of `g` on `K`.
-/

@[expose] public section

namespace NeuralNetworks.MultiInput

/-- The universal approximation statement for shallow ReLU networks with `Dᵢ` inputs. -/
def UniversalApproximation : Prop :=
  ∀ (Dᵢ : ℕ) (K : Set (Fin Dᵢ → ℝ)), IsCompact K →
    ∀ (g : (Fin Dᵢ → ℝ) → ℝ), ContinuousOn g K →
      ∀ (ε : ℝ), 0 < ε →
        ∃ (D : ℕ) (Φ : Parameters ℝ Dᵢ D),
          ∀ x ∈ K, |network relu x Φ - g x| < ε

/-- **Universal approximation** for shallow ReLU networks with `Dᵢ` inputs. -/
theorem universalApproximation : UniversalApproximation := by
  intro Dᵢ K hK g hg ε hε
  -- Extend `g` to a continuous function on all of `ℝ^Dᵢ`.
  obtain ⟨G, hG⟩ := ContinuousMap.exists_restrict_eq hK.isClosed
    ⟨fun y : K => g y, continuousOn_iff_continuous_domRestrict.mp hg⟩
  have hGK : ∀ x ∈ K, G x = g x := fun x hx =>
    congrArg (fun f : C(K, ℝ) => f ⟨x, hx⟩) hG
  -- Stone–Weierstrass: approximate `G` by the exponential ridge algebra.
  obtain ⟨f, hf, hfG⟩ :=
    ContinuousMap.exists_mem_subalgebra_near_continuous_of_isCompact_of_separatesPoints
      (expRidgeAlgebra_separatesPoints Dᵢ) G hK (half_pos hε)
  -- Approximate that element by a network.
  obtain ⟨F, ⟨D, Φ, hΦ⟩, hFf⟩ := approximable_of_mem hK hf (ε / 2) (half_pos hε)
  refine ⟨D, Φ, fun x hx => ?_⟩
  have h₁ := hFf x hx
  have h₂ := hfG x hx
  rw [Real.norm_eq_abs, hGK x hx] at h₂
  rw [hΦ]
  rw [abs_lt] at *
  constructor <;> linarith [h₁.1, h₁.2, h₂.1, h₂.2]

end NeuralNetworks.MultiInput
