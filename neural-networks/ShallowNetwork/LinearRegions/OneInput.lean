module

public import ShallowNetwork.Model
public import Mathlib.Basic.Real.Basic
public import Mathlib.Data.Finset.Card
public import Mathlib.Order.Interval.Set.Defs
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.FieldSimp
public import Mathlib.Tactic.Ring

/-!
# Linear regions with one input

Source: Understanding Deep Learning, Sections 3.1–3.2.

Each hidden unit `ReLU(θ_d0 + θ_d1 * x)` bends at most once, at its joint
`x = -θ_d0 / θ_d1`, and is affine on either side. The network adds these units,
so it can change slope only at their joints. A network with `D` hidden units
therefore has at most `D` joints, which split the real line into at most
`D + 1` intervals, and it is affine on each one.

`LinearRegionBound` states this without counting regions directly: some set `S`
of at most `D` joints exists such that the network is affine on every closed
interval whose interior contains no joint. The pieces of the real line between
consecutive points of `S` are the linear regions, and there are `S.card + 1 ≤ D + 1`
of them. The bound is an upper bound: joints can coincide, and a unit with zero
output weight or zero input weight contributes no bend.

`linearRegionBound` proves it. The joints are `-θ_d0 / θ_d1` for the units with
nonzero input weight. On an interval with no joint inside, each unit's
preactivation keeps one sign, so ReLU acts as the identity or as zero and the
unit is affine (`hiddenUnit_affine`). A weighted sum of affine functions plus
the output bias is affine.
-/

@[expose] public section

namespace NeuralNetworks

/-- The linear region bound for one-input shallow ReLU networks. -/
def LinearRegionBound : Prop :=
  ∀ (D : ℕ) (Φ : ShallowNetworkParameters ℝ D),
    ∃ S : Finset ℝ, S.card ≤ D ∧
      ∀ a b : ℝ, (∀ s ∈ S, s ∉ Set.Ioo a b) →
        ∃ m c : ℝ, ∀ x ∈ Set.Icc a b, shallowNetwork relu x Φ = m * x + c

/-- A hidden unit is affine on `[a, b]` when its joint is not strictly inside.
A zero input weight gives a constant unit with no joint. -/
theorem hiddenUnit_affine (θ₀ θ₁ a b : ℝ)
    (h : θ₁ ≠ 0 → -θ₀ / θ₁ ∉ Set.Ioo a b) :
    ∃ m c : ℝ, ∀ x ∈ Set.Icc a b, hiddenUnit relu θ₀ θ₁ x = m * x + c := by
  simp only [hiddenUnit, addBias, relu_eq_max]
  by_cases hθ : θ₁ = 0
  · exact ⟨0, max θ₀ 0, fun x _ => by simp [hθ]⟩
  -- The preactivation is `θ₁ * (x - r)` with joint `r`.
  set r := -θ₀ / θ₁
  have hr : θ₀ + θ₁ * r = 0 := by simp only [r]; field_simp; ring
  -- On `[a, b]` it keeps one sign, so ReLU is the identity or zero there.
  have sign : (∀ x ∈ Set.Icc a b, 0 ≤ θ₀ + θ₁ * x) ∨ (∀ x ∈ Set.Icc a b, θ₀ + θ₁ * x ≤ 0) := by
    have hr' : r ≤ a ∨ b ≤ r := by
      by_contra hc
      exact h hθ ⟨lt_of_not_ge fun h' => hc (Or.inl h'),
        lt_of_not_ge fun h' => hc (Or.inr h')⟩
    rcases hr' with hra | hbr <;> rcases lt_or_gt_of_ne hθ with hneg | hpos
    · right; intro x hx; nlinarith [hx.1]
    · left; intro x hx; nlinarith [hx.1]
    · left; intro x hx; nlinarith [hx.2]
    · right; intro x hx; nlinarith [hx.2]
  rcases sign with hpos | hneg
  · exact ⟨θ₁, θ₀, fun x hx => by rw [max_eq_left (hpos x hx)]; ring⟩
  · exact ⟨0, 0, fun x hx => by rw [max_eq_right (hneg x hx)]; ring⟩

/-- **Linear region bound**: a one-input shallow ReLU network of width `D`
is affine between consecutive points of a set of at most `D` joints. -/
theorem linearRegionBound : LinearRegionBound := by
  intro D Φ
  -- The joints of the units with nonzero input weight.
  refine ⟨(Finset.univ.filter fun d => Φ.hiddenWeights d ≠ 0).image
      fun d => -Φ.hiddenBiases d / Φ.hiddenWeights d, ?_, ?_⟩
  · calc _ ≤ (Finset.univ.filter fun d => Φ.hiddenWeights d ≠ 0).card := Finset.card_image_le
      _ ≤ (Finset.univ : Finset (Fin D)).card := Finset.card_filter_le _ _
      _ = D := by simp
  intro a b hS
  -- Each unit is affine on `[a, b]`.
  have unit : ∀ d, ∃ m c : ℝ, ∀ x ∈ Set.Icc a b,
      hiddenUnit relu (Φ.hiddenBiases d) (Φ.hiddenWeights d) x = m * x + c :=
    fun d => hiddenUnit_affine _ _ a b fun hd =>
      hS _ (Finset.mem_image.mpr ⟨d, by simp [hd], rfl⟩)
  choose m c hmc using unit
  -- So is their weighted sum.
  refine ⟨∑ d, Φ.outputWeights d * m d,
    Φ.outputBias + ∑ d, Φ.outputWeights d * c d, fun x hx => ?_⟩
  simp only [shallowNetwork, addBias, hmc _ x hx, mul_add, Finset.sum_add_distrib,
    Finset.sum_mul, mul_assoc]
  ring

end NeuralNetworks
