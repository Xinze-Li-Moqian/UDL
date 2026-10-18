module

public import ProbabilityModels.Bernoulli
public import LossFunctions.BinaryCrossEntropy
public import LossFunctions.NegativeLogLikelihood
public import LossFunctions.Inference
public import ActivationFunction.Sigmoid

/-!
# Example: Bernoulli binary classification

Source: Understanding Deep Learning, Section 5.4, equations (5.16)–(5.20).

The target is a label y ∈ {0, 1}, represented by `Fin 2`. The model first
produces a real score, then a sigmoid converts it to the probability of label 1:

    z = f(x, φ),    λ(x, φ) = sig(z) = 1 / (1 + exp(−z)).

Prediction uses the existing Bernoulli family. Training uses the existing
negative log-likelihood, whose expansion is binary cross-entropy. These
definitions specify the objective without implementing a training algorithm.
-/

@[expose] public section

namespace LossFunctions.Examples.BinaryClassification

open MeasureTheory ProbabilityModels
open scoped LossFunctions ProbabilityModels Optimization

variable {X Parameters : Type*} {I : ℕ}

/-- Equation (5.18): map a real network score to the probability of label 1.

    λ(x, φ) = sig(f(x, φ)),    0 < λ(x, φ) < 1.

`ActivationFunction.Sigmoid` exposes Mathlib's sigmoid as a shared activation.
Mathlib's unit-interval version packages the same value with its range proof.
-/
noncomputable def predictedProbability (f : X → Parameters → ℝ)
    (x : X) (φ : Parameters) : unitInterval :=
  unitInterval.sigmoid (f x φ)

@[simp] theorem coe_predictedProbability (f : X → Parameters → ℝ)
    (x : X) (φ : Parameters) :
    (predictedProbability f x φ : ℝ) = NeuralNetworks.sigmoid (f x φ) := rfl

/-- Return the full distribution over the two labels.

    Pr[· | x; predictiveDistribution f, φ] = Bernoulli(λ(x, φ)).
-/
noncomputable def predictiveDistribution (f : X → Parameters → ℝ)
    (φ : Parameters) (x : X) : ProbabilityMeasure (Fin 2) :=
  bernoulliFamily.predictiveDistribution (predictedProbability f) x φ

/-- Equation (5.19): substitute the predicted probability into the Bernoulli mass.

    p(y | x, φ) = (1 − sig(f(x, φ)))^(1 − y) sig(f(x, φ))^y.
-/
theorem predictiveMass_eq (f : X → Parameters → ℝ)
    (x : X) (φ : Parameters) (y : Fin 2) :
    Pr[y | predictedProbability f x φ; bernoulliMass] =
      (1 - NeuralNetworks.sigmoid (f x φ)) ^ (1 - y.val) *
        NeuralNetworks.sigmoid (f x φ) ^ y.val :=
  bernoulliMass_eq (predictedProbability f x φ) y

/-- Both labels have strictly positive probability for every finite real score. -/
theorem predictiveMass_pos (f : X → Parameters → ℝ)
    (x : X) (φ : Parameters) (y : Fin 2) :
    0 < Pr[y | predictedProbability f x φ; bernoulliMass] :=
  bernoulliMass_pos (predictedProbability f x φ)
    (NeuralNetworks.sigmoid_mem_Ioo _).1 (NeuralNetworks.sigmoid_mem_Ioo _).2 y

/-- Sum the reusable single-example binary cross-entropy over the training data.

    L[φ] = −∑ᵢ log p(yᵢ | λ(xᵢ, φ)).
-/
noncomputable def crossEntropyLoss (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → Fin 2) : LossFunction ℝ Parameters :=
  empiricalLoss binaryCrossEntropy (predictedProbability f) x y

/-- The empirical binary cross-entropy is the existing Bernoulli negative log-likelihood. -/
theorem crossEntropyLoss_eq_negativeLogLikelihood (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → Fin 2) :
    crossEntropyLoss f x y =
      negativeLogLikelihood bernoulliMass (predictedProbability f) x y := rfl

/-- Equation (5.20): the Bernoulli negative log-likelihood is binary cross-entropy.

    L[φ] = ∑ᵢ [−(1 − yᵢ) log(1 − λᵢ) − yᵢ log λᵢ],
    λᵢ = sig(f(xᵢ, φ)).
-/
theorem crossEntropyLoss_eq (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → Fin 2) (φ : Parameters) :
    let L := crossEntropyLoss f x y
    L[φ] = ∑ i : Fin I,
      (-(1 - ((y i).val : ℝ)) * Real.log (1 - NeuralNetworks.sigmoid (f (x i) φ)) -
        ((y i).val : ℝ) * Real.log (NeuralNetworks.sigmoid (f (x i) φ))) := by
  change (∑ i : Fin I, -Real.log
    (bernoulliMass (predictedProbability f (x i) φ) (y i))) = _
  simp only [neg_log_bernoulliMass, coe_predictedProbability]

/-- The shared likelihood equivalence applies because sigmoid gives positive masses. -/
theorem isMaximumLikelihoodEstimator_iff_mem_argmin (f : X → Parameters → ℝ)
    (x : Fin I → X) (y : Fin I → Fin 2) («φ̂» : Parameters) :
    let L := crossEntropyLoss f x y
    IsMaximumLikelihoodEstimator bernoulliMass (predictedProbability f) x y «φ̂» ↔
      «φ̂» ∈ argmin φ, L[φ] :=
  isMaximumLikelihoodEstimator_iff_isNegativeLogLikelihoodMinimizer
    bernoulliMass (predictedProbability f) x y
    (fun φ i => predictiveMass_pos f (x i) φ (y i)) «φ̂»

/-- Select label 1 when its probability exceeds one half; ties select label 0.

    ŷ = 1 if λ(x, φ̂) > 1/2, and ŷ = 0 otherwise.
-/
noncomputable def predictLabel (f : X → Parameters → ℝ)
    (x : X) («φ̂» : Parameters) : Fin 2 := by
  classical
  exact if (1 / 2 : ℝ) < (predictedProbability f x «φ̂» : ℝ) then 1 else 0

/-- Thresholding the sigmoid probability at one half is equivalent to testing the score.

    ŷ = 1 if f(x, φ̂) > 0, and ŷ = 0 otherwise.
-/
theorem predictLabel_eq_score_rule (f : X → Parameters → ℝ)
    (x : X) («φ̂» : Parameters) :
    predictLabel f x «φ̂» = (if 0 < f x «φ̂» then (1 : Fin 2) else 0) := by
  classical
  simp only [predictLabel, coe_predictedProbability, NeuralNetworks.half_lt_sigmoid_iff]

/-- The threshold rule selects a mode of the predicted Bernoulli distribution. -/
theorem predictLabel_isModePrediction (f : X → Parameters → ℝ)
    (x : X) («φ̂» : Parameters) :
    IsModePrediction bernoulliMass (predictedProbability f) x «φ̂»
      (predictLabel f x «φ̂») := by
  classical
  intro y
  unfold predictLabel
  split_ifs with h
  · fin_cases y
    · change 1 - (predictedProbability f x «φ̂» : ℝ) ≤
        (predictedProbability f x «φ̂» : ℝ)
      linarith
    · exact le_rfl
  · fin_cases y
    · exact le_rfl
    · change (predictedProbability f x «φ̂» : ℝ) ≤
        1 - (predictedProbability f x «φ̂» : ℝ)
      linarith

end LossFunctions.Examples.BinaryClassification
