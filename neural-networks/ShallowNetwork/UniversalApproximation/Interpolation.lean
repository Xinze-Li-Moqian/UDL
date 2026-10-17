module

public import ShallowNetwork.UniversalApproximation.Representable
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.FieldSimp

/-!
# Piecewise-linear interpolation with ramps

`ramp t h` is `0` left of `t`, `1` right of `t + h`, and linear in between.
It is the difference of two shifted ReLU units scaled by `1 / h`, so it is
computed by a network of width two.

On the grid `a + k * h`, the sum
`G a + ∑ k < n, (G (a + (k + 1) * h) - G (a + k * h)) * ramp (a + k * h) h x`
interpolates `G` linearly between grid points. `interpolant_sum` evaluates it on
a single cell by telescoping. `abs_convex_lt` turns this into an error bound:
a convex combination of two errors below `ε` stays below `ε`.
-/

@[expose] public section

namespace NeuralNetworks

/-! ### Ramps -/

/-- A ramp rising linearly from `0` at `t` to `1` at `t + h`. -/
noncomputable def ramp (t h x : ℝ) : ℝ := (relu (x - t) - relu (x - (t + h))) / h

/-- A ramp uses two hidden units. -/
theorem ramp_representable (t h : ℝ) : ReLURepresentable (ramp t h) := by
  have := (ReLURepresentable.relu_sub t |>.mul_left (1 / h)).add
    (ReLURepresentable.relu_sub (t + h) |>.mul_left (-(1 / h)))
  convert this using 2 with x
  simp only [ramp]
  ring

/-- The ramp is `0` to the left of `t`. -/
theorem ramp_of_le {t h x : ℝ} (hh : 0 < h) (hx : x ≤ t) : ramp t h x = 0 := by
  simp only [ramp, relu_eq_max]
  rw [max_eq_right (by linarith), max_eq_right (by linarith)]
  simp

/-- The ramp is `1` to the right of `t + h`. -/
theorem ramp_of_ge {t h x : ℝ} (hh : 0 < h) (hx : t + h ≤ x) : ramp t h x = 1 := by
  simp only [ramp, relu_eq_max]
  rw [max_eq_left (by linarith), max_eq_left (by linarith)]
  field_simp
  ring

/-- The ramp is linear between `t` and `t + h`. -/
theorem ramp_mem_Icc {t h x : ℝ} (h₁ : t ≤ x) (h₂ : x ≤ t + h) :
    ramp t h x = (x - t) / h := by
  simp only [ramp, relu_eq_max]
  rw [max_eq_left (by linarith), max_eq_right (by linarith)]
  simp

/-! ### Interpolation -/

/-- Telescoping evaluation of the interpolant on the cell `[a + j h, a + (j + 1) h]`. -/
theorem interpolant_sum (G : ℝ → ℝ) {a h x : ℝ} (hh : 0 < h) {j : ℕ}
    (hj₁ : a + j * h ≤ x) (hj₂ : x ≤ a + (j + 1) * h) :
    ∀ n, j < n → ∑ k ∈ Finset.range n,
        (G (a + (k + 1) * h) - G (a + k * h)) * ramp (a + k * h) h x =
      G (a + j * h) - G a +
        (G (a + (j + 1) * h) - G (a + j * h)) * ((x - (a + j * h)) / h) := by
  have below : ∀ n, n ≤ j → ∑ k ∈ Finset.range n,
      (G (a + (k + 1) * h) - G (a + k * h)) * ramp (a + k * h) h x = G (a + n * h) - G a := by
    intro n
    induction n with
    | zero => simp
    | succ n ih =>
      intro hn
      have hn' : ((n : ℝ) + 1) ≤ j := by exact_mod_cast hn
      rw [Finset.sum_range_succ, ih (by omega),
        ramp_of_ge hh (by nlinarith)]
      push_cast
      ring
  intro n hn
  induction n with
  | zero => omega
  | succ n ih =>
    rw [Finset.sum_range_succ]
    rcases Nat.lt_succ_iff_lt_or_eq.mp hn with hlt | rfl
    · have hn' : (j : ℝ) + 1 ≤ n := by exact_mod_cast hlt
      rw [ih hlt, ramp_of_le hh (by nlinarith)]
      ring
    · rw [below j le_rfl, ramp_mem_Icc hj₁ (by linarith)]

/-- A convex combination of two values below `ε` in absolute value stays below `ε`. -/
theorem abs_convex_lt {A B θ ε : ℝ} (hA : |A| < ε) (hB : |B| < ε)
    (h₀ : 0 ≤ θ) (h₁ : θ ≤ 1) : |(1 - θ) * A + θ * B| < ε := by
  rw [abs_lt] at *
  have h₁' : 0 ≤ 1 - θ := by linarith
  constructor
  · have hm : 0 < min (A + ε) (B + ε) := lt_min (by linarith) (by linarith)
    nlinarith [mul_le_mul_of_nonneg_left (min_le_left (A + ε) (B + ε)) h₁',
      mul_le_mul_of_nonneg_left (min_le_right (A + ε) (B + ε)) h₀]
  · have hm : 0 < min (ε - A) (ε - B) := lt_min (by linarith) (by linarith)
    nlinarith [mul_le_mul_of_nonneg_left (min_le_left (ε - A) (ε - B)) h₁',
      mul_le_mul_of_nonneg_left (min_le_right (ε - A) (ε - B)) h₀]

end NeuralNetworks
