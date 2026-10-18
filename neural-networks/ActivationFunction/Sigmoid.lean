module

public import ActivationFunction
public import Mathlib.Analysis.SpecialFunctions.Sigmoid

/-!
# Logistic sigmoid

Source: Understanding Deep Learning, Section 5.4, equation (5.18).

    sig(z) = 1 / (1 + exp(−z)),    0 < sig(z) < 1.

This is Mathlib's real sigmoid, viewed as an `ActivationFunction ℝ`.
For binary classification, it converts the model's real score into a
Bernoulli probability. It is increasing and maps zero to one half.
-/

@[expose] public section

namespace NeuralNetworks

/-- Logistic sigmoid as a scalar activation function, reusing Mathlib's definition.

    sig : ℝ → (0, 1),    sig(z) = 1 / (1 + exp(−z)).
-/
noncomputable abbrev sigmoid : ActivationFunction ℝ := Real.sigmoid

/-- Equation (5.18), written with division by the positive denominator. -/
theorem sigmoid_eq (z : ℝ) : sigmoid z = 1 / (1 + Real.exp (-z)) := by
  rw [one_div]
  rfl

/-- Every finite score gives a strictly interior probability. -/
theorem sigmoid_mem_Ioo (z : ℝ) : sigmoid z ∈ Set.Ioo (0 : ℝ) 1 :=
  ⟨Real.sigmoid_pos z, Real.sigmoid_lt_one z⟩

@[simp] theorem sigmoid_zero : sigmoid 0 = (1 / 2 : ℝ) := by
  simpa only [one_div] using Real.sigmoid_zero

/-- Increasing scores give increasing probabilities. -/
theorem sigmoid_strictMono : StrictMono sigmoid := Real.sigmoid_strictMono

/-- A positive score gives probability above one half. -/
theorem half_lt_sigmoid_iff (z : ℝ) : (1 / 2 : ℝ) < sigmoid z ↔ 0 < z := by
  rw [← sigmoid_zero]
  exact Real.sigmoid_lt_iff

/-- A negative score gives probability below one half. -/
theorem sigmoid_lt_half_iff (z : ℝ) : sigmoid z < (1 / 2 : ℝ) ↔ z < 0 := by
  rw [← sigmoid_zero]
  exact Real.sigmoid_lt_iff

/-- Negating a score exchanges the two class probabilities. -/
theorem sigmoid_neg (z : ℝ) : sigmoid (-z) = 1 - sigmoid z := Real.sigmoid_neg z

end NeuralNetworks
