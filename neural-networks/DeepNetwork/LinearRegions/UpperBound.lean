module

public import DeepNetwork.LinearRegions.Pieces

/-!
# The one-dimensional deep-network region bound

Source: Understanding Deep Learning, Section 4.5.2.

For arbitrary hidden widths, a ReLU network has at most `∏ k, (D_k + 1)`
affine pieces. The induction tracks a common partition for every coordinate of
each hidden layer. The final affine output introduces no additional joints.
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

private theorem parameters_cons {Dᵢ D Dₒ : ℕ} {widths : List ℕ}
    (Φ : NetworkParameters ℝ (Architecture.ofList Dᵢ (D :: widths) Dₒ)) :
    Φ = NetworkParameters.cons (Φ.β 0) (Φ.Ω 0)
      { β := fun k => Φ.β k.succ, Ω := fun k => Φ.Ω k.succ } := by
  cases Φ with
  | mk β Ω =>
    congr 1 <;> funext k <;> refine Fin.cases ?_ (fun j => ?_) k <;> rfl

/-- Applying a network to a vector of functions with a common partition
multiplies the piece bound by the product of the hidden widths plus one. -/
theorem network_affinePieces (widths : List ℕ) {Dᵢ Dₒ : ℕ}
    (Φ : NetworkParameters ℝ (Architecture.ofList Dᵢ widths Dₒ))
    (f : ℝ → Fin Dᵢ → ℝ) (S : Finset ℝ)
    (hf : ∀ d, AffinePieces (fun x => f x d) S) :
    ∃ T : Finset ℝ, T.card + 1 ≤ (S.card + 1) * (widths.map (· + 1)).prod ∧
      ∀ j, AffinePieces (fun x => network relu Φ (f x) j) T := by
  induction widths generalizing Dᵢ S with
  | nil =>
    refine ⟨S, by simp, fun j => ?_⟩
    exact AffinePieces.affine_combination hf (Φ.β 0 j) (Φ.Ω 0 j)
  | cons D widths ih =>
    let rest : NetworkParameters ℝ (Architecture.ofList D widths Dₒ) :=
      { β := fun k => Φ.β k.succ, Ω := fun k => Φ.Ω k.succ }
    let g : ℝ → Fin D → ℝ := fun x => Φ.β 0 + Φ.Ω 0 *ᵥ f x
    have hg : ∀ d, AffinePieces (fun x => g x d) S :=
      fun d => AffinePieces.affine_combination hf (Φ.β 0 d) (Φ.Ω 0 d)
    obtain ⟨T, hT, hrelu⟩ := AffinePieces.relu_refine hg
    obtain ⟨U, hU, hout⟩ := ih rest (fun x => relu ∘ g x) T hrelu
    refine ⟨U, ?_, ?_⟩
    · calc U.card + 1 ≤ (T.card + 1) * (widths.map (· + 1)).prod := hU
        _ ≤ ((S.card + 1) * (D + 1)) * (widths.map (· + 1)).prod :=
          Nat.mul_le_mul_right _ hT
        _ = (S.card + 1) * ((D :: widths).map (· + 1)).prod := by
          simp [Nat.mul_assoc]
    · intro j
      have heq := parameters_cons Φ
      change Φ = NetworkParameters.cons (Φ.β 0) (Φ.Ω 0) rest at heq
      have heval : ∀ x, network relu Φ (f x) j = network relu rest (relu ∘ g x) j := by
        intro x
        conv_lhs => rw [heq]
        exact congrFun (network_cons relu (Dᵢ := Dᵢ) (D := D) (Dₒ := Dₒ)
          (Φ.β 0) (Φ.Ω 0) rest (f x)) j
      intro a b h
      obtain ⟨m, c, hmc⟩ := hout j a b h
      exact ⟨m, c, fun x hx => (heval x).trans (hmc x hx)⟩

/-- **Section 4.5.2, upper bound:** arbitrary one-input ReLU networks admit a
partition with at most `∏ D ∈ widths, (D + 1)` pieces, shared by all outputs. -/
theorem deepLinearRegionBound (widths : List ℕ)
    (Φ : NetworkParameters ℝ (Architecture.ofList 1 widths 1)) :
    ∃ S : Finset ℝ, S.card + 1 ≤ (widths.map (· + 1)).prod ∧
      ∀ j, AffinePieces (fun x => network relu Φ (fun _ : Fin 1 => x) j) S := by
  obtain ⟨S, hS, hf⟩ := network_affinePieces widths Φ (fun x _ => x) ∅
    (fun _ a b _ => ⟨1, 0, fun x _ => by simp⟩)
  exact ⟨S, by simpa using hS, hf⟩

/-- With `K` hidden layers of width `D`, the bound is `(D + 1)^K`. -/
theorem deepLinearRegionBound_uniform (D K : ℕ)
    (Φ : NetworkParameters ℝ (Architecture.ofList 1 (List.replicate K D) 1)) :
    ∃ S : Finset ℝ, S.card + 1 ≤ (D + 1) ^ K ∧
      ∀ j, AffinePieces (fun x => network relu Φ (fun _ : Fin 1 => x) j) S := by
  simpa using deepLinearRegionBound (List.replicate K D) Φ

end NeuralNetworks
