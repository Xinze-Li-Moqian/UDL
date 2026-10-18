module

public import ProbabilityModels.Categorical
public import Argmax

/-!
# Example: categorical probabilities for three labels

Start with the distribution in Figure 5.9, using three named categories:

    0 = cat,    1 = dog,    2 = bird,
    λ = (0.7, 0.2, 0.1).

These probabilities are specified for illustration, not learned from data.
The example verifies their probability interpretation and most likely label.
The softmax-based predictor is defined separately in `MulticlassClassification.Softmax`.
-/

@[expose] public section

namespace LossFunctions.Examples.MulticlassClassification

open ProbabilityModels
open scoped BigOperators ProbabilityModels Optimization

/-- A categorical parameter vector, with proofs that its entries form probabilities. -/
noncomputable def animalProbabilities : CategoricalParameters 3 :=
  CategoricalParameters.ofProbabilities ![(7 / 10 : ℝ), 2 / 10, 1 / 10]
    (by intro k; fin_cases k <;> norm_num)
    (by norm_num [Fin.sum_univ_succ])

-- Each label retrieves its own probability through the shared notation.
example : Pr[0 | animalProbabilities; categoricalMass] = (7 / 10 : ℝ) := by
  simp [animalProbabilities]

example : Pr[1 | animalProbabilities; categoricalMass] = (2 / 10 : ℝ) := by
  simp [animalProbabilities]

example : Pr[2 | animalProbabilities; categoricalMass] = (1 / 10 : ℝ) := by
  simp [animalProbabilities]

-- These values are the masses of an actual normalized probability measure.
example : (categoricalFamily animalProbabilities : MeasureTheory.Measure (Fin 3)).real {0} =
    (7 / 10 : ℝ) := by
  rw [categoricalFamily_real_singleton]
  simp [animalProbabilities]

example : ∑ k : Fin 3, Pr[k | animalProbabilities; categoricalMass] = 1 :=
  sum_categoricalMass animalProbabilities

/-- Cat is the most likely label, since 0.7 is at least every other mass. -/
theorem cat_isMode : (0 : Fin 3) ∈ argmax k, Pr[k | animalProbabilities; categoricalMass] := by
  intro k
  fin_cases k <;> norm_num [categoricalMass, animalProbabilities]

end LossFunctions.Examples.MulticlassClassification
