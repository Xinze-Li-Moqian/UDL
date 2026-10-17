module

public import ShallowNetwork.UniversalApproximation.Interpolation
public import Mathlib.Topology.TietzeExtension
public import Mathlib.Topology.UniformSpace.HeineCantor
public import Mathlib.Algebra.Order.Floor.Semiring

/-!
# Universal approximation with one input

Source: Understanding Deep Learning, Section 3.2.

Every continuous real-valued function on a compact subset of the real line can
be approximated uniformly by a shallow ReLU network to any positive accuracy.
The width and parameters may depend on the target function, set, and accuracy;
one chosen network must satisfy the error bound at every point of the set.

`UniversalApproximation` states the proposition and `universalApproximation`
proves it. The proof follows the book's picture of ReLU networks as piecewise
linear functions:

1. Extend `g` from `K` to a continuous function `G` on all of `ℝ` (Tietze).
2. Enclose `K` in an interval `[a, b]`, on which `G` is uniformly continuous.
3. Split `[a, b]` into `N` cells of width `h`, short enough that `G` varies by
   less than `ε` across any cell.
4. Interpolate `G` linearly between grid points. Each cell contributes a
   `ramp` of two ReLU units, so the network has width `2 * N`.
5. On each cell the interpolant is a convex combination of `G` at the cell's
   endpoints, so its error is below `ε`.

The supporting lemmas are in `Representable` (functions computed by shallow ReLU
networks and their closure properties) and `Interpolation` (ramps and
piecewise-linear interpolation).
-/

@[expose] public section

namespace NeuralNetworks

/-- The universal approximation statement for the shallow ReLU model. -/
def UniversalApproximation : Prop :=
  ∀ (K : Set ℝ), IsCompact K →
    ∀ (g : ℝ → ℝ), ContinuousOn g K →
      ∀ (ε : ℝ), 0 < ε →
        ∃ (D : ℕ) (Φ : ShallowNetworkParameters ℝ D),
          ∀ x ∈ K, |shallowNetwork relu x Φ - g x| < ε

/-- **Universal approximation** for shallow ReLU networks on the real line.

The network is the piecewise-linear interpolant of a continuous extension of `g`
on a grid fine enough for uniform continuity; each grid cell uses two hidden units. -/
theorem universalApproximation : UniversalApproximation := by
  intro K hK g hg ε hε
  -- Extend `g` to a continuous function on the whole line.
  obtain ⟨G, hG⟩ := ContinuousMap.exists_restrict_eq hK.isClosed
    ⟨fun y : K => g y, continuousOn_iff_continuous_domRestrict.mp hg⟩
  have hGK : ∀ x ∈ K, G x = g x := fun x hx =>
    congrArg (fun f : C(K, ℝ) => f ⟨x, hx⟩) hG
  -- Enclose `K` in an interval `[a, b]` with `a < b`.
  obtain ⟨a, ha⟩ := hK.bddBelow
  obtain ⟨b₀, hb₀⟩ := hK.bddAbove
  set b := max b₀ (a + 1)
  have hab : a < b := by simp [b]
  have hKab : ∀ x ∈ K, x ∈ Set.Icc a b := fun x hx =>
    ⟨ha hx, (hb₀ hx).trans (le_max_left _ _)⟩
  -- `G` is uniformly continuous on `[a, b]`.
  obtain ⟨δ, hδ, hGδ⟩ := Metric.uniformContinuousOn_iff.mp
    (isCompact_Icc.uniformContinuousOn_of_continuous G.continuous.continuousOn) ε hε
  -- Choose a grid of `N` cells, each shorter than `δ`.
  obtain ⟨N, hN⟩ := exists_nat_gt ((b - a) / δ)
  have hNpos : (0 : ℝ) < N := lt_trans (div_pos (by linarith) hδ) hN
  set h := (b - a) / N
  have hh : 0 < h := div_pos (by linarith) hNpos
  have hhδ : h < δ := by
    rw [div_lt_iff₀ hNpos]; rw [div_lt_iff₀ hδ] at hN; linarith
  have hNh : a + N * h = b := by simp only [h]; field_simp; ring
  -- The piecewise-linear interpolant of `G` on the grid.
  set L : ℝ → ℝ := fun x => G a + ∑ k ∈ Finset.range N,
    (G (a + (k + 1) * h) - G (a + k * h)) * ramp (a + k * h) h x
  have hL : ReLURepresentable L :=
    (ReLURepresentable.const _).add (ReLURepresentable.sum _ fun k _ =>
      (ramp_representable _ _).mul_left _)
  obtain ⟨D, Φ, hΦ⟩ := hL
  refine ⟨D, Φ, fun x hx => ?_⟩
  obtain ⟨hax, hxb⟩ := hKab x hx
  rw [hΦ, ← hGK x hx]
  -- Locate the cell `j` containing `x`.
  have hN1 : 1 ≤ N := by exact_mod_cast hNpos
  set j := min ⌊(x - a) / h⌋₊ (N - 1)
  have hjN : j < N := by omega
  have hjN' : (j : ℝ) + 1 ≤ N := by exact_mod_cast hjN
  have hj₁ : a + j * h ≤ x := by
    have : (j : ℝ) ≤ (x - a) / h :=
      (Nat.cast_le.mpr (min_le_left _ _)).trans (Nat.floor_le (div_nonneg (by linarith) hh.le))
    rw [le_div_iff₀ hh] at this; linarith
  have hj₂ : x ≤ a + (j + 1) * h := by
    rcases min_choice ⌊(x - a) / h⌋₊ (N - 1) with hj | hj
    · have := Nat.lt_floor_add_one ((x - a) / h)
      rw [← hj, div_lt_iff₀ hh] at this
      change x - a < (j + 1) * h at this
      linarith
    · have : (j : ℝ) + 1 = N := by
        change ((min _ _ : ℕ) : ℝ) + 1 = N
        rw [hj]; exact_mod_cast Nat.sub_add_cancel hN1
      rw [this, hNh]; exact hxb
  have hjIcc : a + j * h ∈ Set.Icc a b :=
    ⟨by nlinarith [(Nat.cast_nonneg j : (0 : ℝ) ≤ j)], by nlinarith⟩
  have hj1Icc : a + (j + 1) * h ∈ Set.Icc a b :=
    ⟨by nlinarith [(Nat.cast_nonneg j : (0 : ℝ) ≤ j)], by nlinarith⟩
  have hA : |G (a + j * h) - G x| < ε := by
    have := hGδ _ hjIcc x ⟨hax, hxb⟩ (by rw [Real.dist_eq, abs_lt]; constructor <;> linarith)
    rwa [Real.dist_eq] at this
  have hB : |G (a + (j + 1) * h) - G x| < ε := by
    have := hGδ _ hj1Icc x ⟨hax, hxb⟩ (by rw [Real.dist_eq, abs_lt]; constructor <;> linarith)
    rwa [Real.dist_eq] at this
  set θ := (x - (a + j * h)) / h
  have hθ₀ : 0 ≤ θ := div_nonneg (by linarith) hh.le
  have hθ₁ : θ ≤ 1 := by rw [div_le_one hh]; linarith
  have key : L x - G x =
      (1 - θ) * (G (a + j * h) - G x) + θ * (G (a + (j + 1) * h) - G x) := by
    simp only [L]
    rw [interpolant_sum G hh hj₁ hj₂ N hjN]
    ring
  rw [key]
  exact abs_convex_lt hA hB hθ₀ hθ₁

end NeuralNetworks
