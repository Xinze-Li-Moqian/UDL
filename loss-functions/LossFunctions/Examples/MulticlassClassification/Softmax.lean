module

public import LossFunctions.Examples.MulticlassClassification.Categorical
public import LossFunctions.Inference
public import ActivationFunction.Softmax

/-!
# Example: softmax categorical predictions

Source: Understanding Deep Learning, Section 5.5, equations (5.22)–(5.23).

    z = f(x, φ),    λ = softmax(z),    Y | x, φ ∼ Categorical(λ).

The model supplies one real score per label. Softmax produces the distribution
parameters; the existing categorical family supplies the full distribution.
The label set must be nonempty. No training algorithm is supplied here.
-/

@[expose] public section

namespace LossFunctions.Examples.MulticlassClassification

open MeasureTheory ProbabilityModels
open scoped ProbabilityModels Optimization

variable {X Parameters : Type*} {K : ℕ} [NeZero K]

/-- Package softmax's probability vector using its positivity and normalization proofs.

    λₖ(x, φ) = softmaxₖ(f(x, φ)).
-/
noncomputable def predictedParameters (f : X → Parameters → Fin K → ℝ)
    (x : X) (φ : Parameters) : CategoricalParameters K :=
  CategoricalParameters.ofProbabilities (NeuralNetworks.softmax (f x φ))
    (fun k => (NeuralNetworks.softmax_pos _ k).le) (NeuralNetworks.sum_softmax _)

/-- Return the categorical distribution selected by the predicted probabilities.

    Pr[· | x; predictiveDistribution f, φ] = Categorical(softmax(f(x, φ))).
-/
noncomputable def predictiveDistribution (f : X → Parameters → Fin K → ℝ)
    (φ : Parameters) (x : X) : ProbabilityMeasure (Fin K) :=
  categoricalFamily.predictiveDistribution (predictedParameters f) x φ

/-- Equation (5.23): a label's predicted probability is its softmax component. -/
@[simp] theorem predictiveMass_eq (f : X → Parameters → Fin K → ℝ)
    (x : X) (φ : Parameters) (k : Fin K) :
    Pr[k | predictedParameters f x φ; categoricalMass] =
      NeuralNetworks.softmax (f x φ) k := by
  simp [predictedParameters]

/-- Every label has positive predicted mass, allowing real logarithms in later losses. -/
theorem predictiveMass_pos (f : X → Parameters → Fin K → ℝ)
    (x : X) (φ : Parameters) (k : Fin K) :
    0 < Pr[k | predictedParameters f x φ; categoricalMass] := by
  rw [predictiveMass_eq]
  exact NeuralNetworks.softmax_pos _ _

/-- A most probable label is exactly a label with a maximal raw score.

    ŷ ∈ argmaxₖ p(k | x, φ̂)  ↔  ŷ ∈ argmaxₖ fₖ(x, φ̂).
-/
theorem isModePrediction_iff_maximal_score (f : X → Parameters → Fin K → ℝ)
    (x : X) («φ̂» : Parameters) (k : Fin K) :
    IsModePrediction categoricalMass (predictedParameters f) x «φ̂» k ↔
      k ∈ argmax j, f x «φ̂» j := by
  change (∀ j, Pr[j | predictedParameters f x «φ̂»; categoricalMass] ≤
    Pr[k | predictedParameters f x «φ̂»; categoricalMass]) ↔
      (∀ j, f x «φ̂» j ≤ f x «φ̂» k)
  simp only [predictiveMass_eq, NeuralNetworks.softmax_le_softmax_iff]

/-- Illustrative scores whose exponentials are 7, 2, and 1. They are not trained parameters. -/
noncomputable def animalScores : Fin 3 → ℝ := ![Real.log 7, Real.log 2, 0]

/-- Softmax recovers the earlier cat, dog, bird probabilities (0.7, 0.2, 0.1). -/
theorem softmax_animalScores (k : Fin 3) :
    NeuralNetworks.softmax animalScores k = animalProbabilities.weights k := by
  fin_cases k <;>
    norm_num [NeuralNetworks.softmax, animalScores, animalProbabilities, Fin.sum_univ_succ,
      Real.exp_log (by norm_num : (0 : ℝ) < 7),
      Real.exp_log (by norm_num : (0 : ℝ) < 2)]

end LossFunctions.Examples.MulticlassClassification
