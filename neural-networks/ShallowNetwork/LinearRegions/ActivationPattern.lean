module

public import ShallowNetwork.MultiInput
public import Mathlib.Basic.Real.Basic
public import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.Ring

/-!
# Activation patterns

The activation pattern of a multi-input ReLU network at `x` is the set of hidden
units with positive preactivation `θ_d0 + ∑ i, θ_di * x_i`. Inputs sharing a
pattern form an intersection of half-spaces, and on that set every active unit
is affine and every inactive unit is zero. So the network is affine on each
pattern (`affine_on_pattern`).

To count patterns, it is enough to show that no `Dᵢ + 1` hidden units realize
all `2 ^ (Dᵢ + 1)` on/off combinations (`not_all_patterns`). Their weight
vectors live in `ℝ^Dᵢ`, so they satisfy a nontrivial relation
`∑ i, coef i * θ_i = 0` (`exists_weight_relation`). Then
`∑ i, coef i * preactivation i x` is a constant `μ`. After choosing the sign of
the relation, the pattern "active exactly where `coef > 0`" would make every
term nonnegative and the sum too large, so it never occurs (`not_pattern`).
-/

@[expose] public section

namespace NeuralNetworks.MultiInput

variable {Dᵢ D : ℕ}

/-- The preactivation `θ_d0 + ∑ i, θ_di * x_i` of hidden unit `d`. -/
def preactivation (Φ : Parameters ℝ Dᵢ D) (d : Fin D) (x : Fin Dᵢ → ℝ) : ℝ :=
  Φ.hiddenBiases d + ∑ i, Φ.hiddenWeights d i * x i

/-- The hidden units that are active (positive preactivation) at `x`. -/
noncomputable def activationPattern (Φ : Parameters ℝ Dᵢ D) (x : Fin Dᵢ → ℝ) : Finset (Fin D) :=
  Finset.univ.filter fun d => 0 < preactivation Φ d x

/-! ### The network is affine on each pattern -/

/-- Only the active units contribute to the output. -/
theorem network_eq_sum_active (Φ : Parameters ℝ Dᵢ D) (x : Fin Dᵢ → ℝ) :
    network relu x Φ = Φ.outputBias +
      ∑ d ∈ activationPattern Φ x, Φ.outputWeights d * preactivation Φ d x := by
  have unit : ∀ d, hiddenUnit relu (Φ.hiddenBiases d) (Φ.hiddenWeights d) x =
      if 0 < preactivation Φ d x then preactivation Φ d x else 0 := by
    intro d
    simp only [hiddenUnit, addBias, relu_eq_max]
    by_cases h : 0 < preactivation Φ d x
    · simp only [h, ↓reduceIte]; exact max_eq_left h.le
    · simp only [h, ↓reduceIte]; exact max_eq_right (not_lt.mp h)
  simp only [network_eq, addBias, unit, mul_ite, mul_zero, activationPattern, Finset.sum_filter]

/-- The network is affine on each activation pattern. -/
theorem affine_on_pattern (Φ : Parameters ℝ Dᵢ D) (P : Finset (Fin D)) :
    ∃ (m : Fin Dᵢ → ℝ) (c : ℝ), ∀ x, activationPattern Φ x = P →
      network relu x Φ = ∑ i, m i * x i + c := by
  refine ⟨fun i => ∑ d ∈ P, Φ.outputWeights d * Φ.hiddenWeights d i,
    Φ.outputBias + ∑ d ∈ P, Φ.outputWeights d * Φ.hiddenBiases d, fun x hx => ?_⟩
  rw [network_eq_sum_active, hx]
  simp only [preactivation, mul_add, Finset.sum_add_distrib, Finset.mul_sum, Finset.sum_mul]
  rw [Finset.sum_comm (s := Finset.univ)]
  simp only [mul_assoc]
  ring

/-! ### Few units cannot realize every pattern -/

/-- Any `Dᵢ + 1` hidden weight vectors in `ℝ^Dᵢ` are linearly dependent. -/
theorem exists_weight_relation (Φ : Parameters ℝ Dᵢ D) (T : Finset (Fin D))
    (hT : T.card = Dᵢ + 1) :
    ∃ coef : T → ℝ, coef ≠ 0 ∧ ∀ j, ∑ i, coef i * Φ.hiddenWeights i j = 0 := by
  -- The linear map `coef ↦ ∑ i, coef i • w_i` from `ℝ^(Dᵢ + 1)` to `ℝ^Dᵢ` has a kernel.
  let f : (T → ℝ) →ₗ[ℝ] (Fin Dᵢ → ℝ) :=
    { toFun := fun v j => ∑ i, v i * Φ.hiddenWeights i j
      map_add' := fun u v => by ext j; simp [add_mul, Finset.sum_add_distrib]
      map_smul' := fun c v => by ext j; simp [Finset.mul_sum, mul_assoc] }
  have hker : LinearMap.ker f ≠ ⊥ := by
    apply LinearMap.ker_ne_bot_of_finrank_lt
    simp [hT]
  obtain ⟨v, hv, hv0⟩ := (Submodule.ne_bot_iff _).mp hker
  exact ⟨v, hv0, fun j => congrFun (LinearMap.mem_ker.mp hv) j⟩

/-- A sign-normalized relation rules out one activation pattern on `T`. -/
theorem not_pattern (Φ : Parameters ℝ Dᵢ D) {T : Finset (Fin D)} (σ : T → ℝ) (μ : ℝ)
    (hrel : ∀ x, ∑ i : T, σ i * preactivation Φ i x = μ) (hμ : μ ≤ 0)
    (hstrict : μ < 0 ∨ ∃ i, 0 < σ i) (x : Fin Dᵢ → ℝ) :
    ¬ ∀ i : T, (0 < preactivation Φ i x ↔ 0 < σ i) := by
  intro h
  have nonneg : ∀ i : T, 0 ≤ σ i * preactivation Φ i x := by
    intro i
    by_cases hσ : 0 < σ i
    · exact (mul_pos hσ ((h i).mpr hσ)).le
    · exact mul_nonneg_of_nonpos_of_nonpos (not_lt.mp hσ)
        (not_lt.mp fun hp => hσ ((h i).mp hp))
  rcases hstrict with hμ' | ⟨i, hi⟩
  · have := Finset.sum_nonneg fun i (_ : i ∈ Finset.univ) => nonneg i
    linarith [hrel x]
  · have := Finset.sum_pos' (fun i (_ : i ∈ Finset.univ) => nonneg i)
      ⟨i, Finset.mem_univ _, mul_pos hi ((h i).mpr hi)⟩
    linarith [hrel x]

/-- No `Dᵢ + 1` hidden units realize all `2 ^ (Dᵢ + 1)` on/off combinations. -/
theorem not_all_patterns (Φ : Parameters ℝ Dᵢ D) (T : Finset (Fin D))
    (hT : T.card = Dᵢ + 1) :
    ¬ ∀ U ⊆ T, ∃ x, T ∩ activationPattern Φ x = U := by
  intro hall
  obtain ⟨coef, hcoef0, hW⟩ := exists_weight_relation Φ T hT
  -- The relation makes `∑ coef i * preactivation i x` constant in `x`.
  set μ := ∑ i, coef i * Φ.hiddenBiases i
  have hrel : ∀ x, ∑ i, coef i * preactivation Φ i x = μ := by
    intro x
    simp only [preactivation, mul_add, Finset.sum_add_distrib, Finset.mul_sum]
    rw [Finset.sum_comm]
    simp only [← mul_assoc, ← Finset.sum_mul, hW, zero_mul, Finset.sum_const_zero, add_zero, μ]
  -- Flip the sign of the relation if needed.
  obtain ⟨i₀, hi₀⟩ := Function.ne_iff.mp hcoef0
  obtain ⟨s, hsμ, hs⟩ : ∃ s : ℝ, s * μ ≤ 0 ∧ (s * μ < 0 ∨ 0 < s * coef i₀) := by
    rcases lt_trichotomy μ 0 with h | h | h
    · exact ⟨1, by linarith, Or.inl (by linarith)⟩
    · rcases lt_or_gt_of_ne hi₀ with hl | hl
      · exact ⟨-1, by simp [h], Or.inr (by simp at hl ⊢; linarith)⟩
      · exact ⟨1, by simp [h], Or.inr (by simpa using hl)⟩
    · exact ⟨-1, by linarith, Or.inl (by linarith)⟩
  set σ : T → ℝ := fun i => s * coef i
  have hrel' : ∀ x, ∑ i, σ i * preactivation Φ i x = s * μ := by
    intro x
    simp only [σ, mul_assoc, ← Finset.mul_sum, hrel x]
  -- The pattern "active exactly where `σ > 0`" never occurs.
  obtain ⟨x, hx⟩ := hall ((Finset.univ.filter fun i => 0 < σ i).map
    (Function.Embedding.subtype _)) (by intro d; simp; tauto)
  refine not_pattern Φ σ (s * μ) hrel' hsμ (hs.imp_right fun h => ⟨i₀, h⟩) x fun i => ?_
  have := Finset.ext_iff.mp hx i
  simpa [activationPattern] using this

end NeuralNetworks.MultiInput
