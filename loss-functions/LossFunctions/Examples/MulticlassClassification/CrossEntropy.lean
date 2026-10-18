module

public import LossFunctions.MulticlassCrossEntropy
public import LossFunctions.NegativeLogLikelihood
public import LossFunctions.Examples.MulticlassClassification.Softmax

/-!
# Example: multiclass cross-entropy training objective

Source: Understanding Deep Learning, Section 5.5, equation (5.24).

    L[φ] = −∑ᵢ log softmaxᵧᵢ(f(xᵢ, φ))
         = −∑ᵢ [fᵧᵢ(xᵢ, φ) − log(∑ₖ exp(fₖ(xᵢ, φ)))].

The empirical loss sums the reusable single-example categorical loss. Its
identity with negative log-likelihood lets the existing maximum-likelihood
equivalence apply. These are objective definitions and proofs, not a solver.
-/

@[expose] public section

namespace LossFunctions.Examples.MulticlassClassification

open ProbabilityModels
open scoped BigOperators LossFunctions ProbabilityModels Optimization

variable {X Parameters : Type*} {K I : ℕ} [NeZero K]

/-- Sum multiclass cross-entropy over the training pairs using softmax predictions. -/
noncomputable def crossEntropyLoss (f : X → Parameters → Fin K → ℝ)
    (x : Fin I → X) (y : Fin I → Fin K) : LossFunction ℝ Parameters :=
  empiricalLoss multiclassCrossEntropy (predictedParameters f) x y

/-- The empirical multiclass loss is the existing categorical negative log-likelihood. -/
theorem crossEntropyLoss_eq_negativeLogLikelihood (f : X → Parameters → Fin K → ℝ)
    (x : Fin I → X) (y : Fin I → Fin K) :
    crossEntropyLoss f x y =
      negativeLogLikelihood categoricalMass (predictedParameters f) x y := rfl

/-- Equation (5.24), first line: sum the negative log probabilities of true labels. -/
theorem crossEntropyLoss_eq (f : X → Parameters → Fin K → ℝ)
    (x : Fin I → X) (y : Fin I → Fin K) (φ : Parameters) :
    let L := crossEntropyLoss f x y
    L[φ] = -∑ i : Fin I, Real.log (NeuralNetworks.softmax (f (x i) φ) (y i)) := by
  change (∑ i : Fin I, -Real.log
    (categoricalMass (predictedParameters f (x i) φ) (y i))) = _
  simp only [predictiveMass_eq, Finset.sum_neg_distrib]

/-- Equation (5.24), second line: expand softmax and simplify the logarithm.

    L[φ] = −∑ᵢ [fᵧᵢ(xᵢ, φ) − log(∑ₖ exp(fₖ(xᵢ, φ)))].
-/
theorem crossEntropyLoss_eq_logSumExp (f : X → Parameters → Fin K → ℝ)
    (x : Fin I → X) (y : Fin I → Fin K) (φ : Parameters) :
    let L := crossEntropyLoss f x y
    L[φ] = -∑ i : Fin I,
      (f (x i) φ (y i) - Real.log (∑ k : Fin K, Real.exp (f (x i) φ k))) := by
  change (∑ i : Fin I, -Real.log
    (categoricalMass (predictedParameters f (x i) φ) (y i))) = _
  simp only [predictiveMass_eq, NeuralNetworks.neg_log_softmax]
  rw [← Finset.sum_neg_distrib]
  apply Finset.sum_congr rfl
  intro i _
  ring

/-- Positive softmax masses give equivalence with maximum likelihood. -/
theorem isMaximumLikelihoodEstimator_iff_mem_argmin (f : X → Parameters → Fin K → ℝ)
    (x : Fin I → X) (y : Fin I → Fin K) («φ̂» : Parameters) :
    let L := crossEntropyLoss f x y
    IsMaximumLikelihoodEstimator categoricalMass (predictedParameters f) x y «φ̂» ↔
      «φ̂» ∈ argmin φ, L[φ] :=
  isMaximumLikelihoodEstimator_iff_isNegativeLogLikelihoodMinimizer
    categoricalMass (predictedParameters f) x y
    (fun φ i => predictiveMass_pos f (x i) φ (y i)) «φ̂»

-- For the earlier animal probabilities, a true cat label uses only the cat probability.
example : multiclassCrossEntropy animalProbabilities 0 = -Real.log (7 / 10 : ℝ) := by
  simp [animalProbabilities]

end LossFunctions.Examples.MulticlassClassification
