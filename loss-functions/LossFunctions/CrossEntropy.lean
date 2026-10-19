module

public import ProbabilityModels.EmpiricalDistribution
public import LossFunctions.NegativeLogLikelihood
public import LossFunctions.BinaryCrossEntropy
public import LossFunctions.MulticlassCrossEntropy

/-!
# Cross-entropy

Source: Understanding Deep Learning, Section 5.7, equations (5.28)–(5.31).

Cross-entropy averages a model's negative log density under a data distribution:

    H(q, pθ) = 𝔼[y ∼ q, −log p(y | θ)].

The density is certified relative to a fixed reference measure. For supervised
learning, average −log p(y | f(x, φ)) over a distribution of pairs (x, y).
An empirical distribution gives the average negative log-likelihood; multiplying
by the positive sample count preserves all minimizers.

These real-valued definitions use Mathlib's totalized integral and logarithm.
The usual finite interpretation requires integrable log density and positive
density almost everywhere under q. Empirical integrability is automatic;
positive observed densities are still needed to exclude impossible observations.

No KL identity between an atomic empirical measure and a continuous model is
claimed: the former is not absolutely continuous with respect to the latter.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open scoped ProbabilityModels LossFunctions Optimization BigOperators

variable {X Y Θ Parameters : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y}

/-- Expected negative log density under the data distribution.

    H(q, pθ) = ∫ −log p(y | θ) dq(y).
-/
noncomputable def crossEntropy (q : ProbabilityMeasure Y) (p : family.Density ν)
    (θ : Θ) : ℝ :=
  𝔼[(fun y => -Real.log Pr[y | θ; p]); (q : Measure Y)]

/-- Supervised cross-entropy uses the distribution of input/output pairs.

    L[φ] = 𝔼[(x, y) ∼ q, −log p(y | f(x, φ))].
-/
noncomputable def conditionalCrossEntropy [MeasurableSpace X]
    (q : ProbabilityMeasure (X × Y)) (p : family.Density ν)
    (f : X → Parameters → Θ) : LossFunction ℝ Parameters :=
  fun φ => 𝔼[(fun z => -Real.log Pr[z.2 | f z.1 φ; p]); (q : Measure (X × Y))]

/-- Equation (5.30): empirical cross-entropy is the average negative log density. -/
theorem crossEntropy_empiricalDistribution [MeasurableSingletonClass Y]
    {I : ℕ} [NeZero I] (y : Fin I → Y) (p : family.Density ν) (θ : Θ) :
    crossEntropy (empiricalDistribution y) p θ =
      (∑ i, -Real.log Pr[y i | θ; p]) / I :=
  expectation_empiricalDistribution y _

/-- A single observation gives the familiar negative log density. -/
@[simp] theorem crossEntropy_single [MeasurableSingletonClass Y]
    (y : Y) (p : family.Density ν) (θ : Θ) :
    crossEntropy (empiricalDistribution (fun _ : Fin 1 => y)) p θ =
      -Real.log Pr[y | θ; p] := by
  rw [crossEntropy_empiricalDistribution]
  simp

/-- Equation (5.31): averaging over training pairs recovers NLL / I. -/
theorem conditionalCrossEntropy_empiricalDistribution
    [MeasurableSpace X] [MeasurableSingletonClass X] [MeasurableSingletonClass Y]
    {I : ℕ} [NeZero I] (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) :
    conditionalCrossEntropy (empiricalDistribution (fun i => (x i, y i))) p f φ =
      negativeLogLikelihood p f x y φ / I := by
  exact expectation_empiricalDistribution (fun i => (x i, y i)) _

/-- Dropping the positive factor 1 / I preserves the set of fitted parameters. -/
theorem mem_argmin_conditionalCrossEntropy_iff
    [MeasurableSpace X] [MeasurableSingletonClass X] [MeasurableSingletonClass Y]
    {I : ℕ} [NeZero I] (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) :
    «φ̂» ∈ argmin φ,
        conditionalCrossEntropy (empiricalDistribution (fun i => (x i, y i))) p f φ ↔
      «φ̂» ∈ argmin φ, negativeLogLikelihood p f x y φ := by
  change (∀ φ, _ ≤ _) ↔ (∀ φ, _ ≤ _)
  simp_rw [conditionalCrossEntropy_empiricalDistribution]
  have hI : (0 : ℝ) < I := Nat.cast_pos.mpr (Nat.pos_of_ne_zero (NeZero.ne I))
  exact forall_congr' (fun _ => div_le_div_iff_of_pos_right hI)

/-- Binary cross-entropy is cross-entropy with the observed label's point mass. -/
theorem crossEntropy_single_bernoulli (predicted : unitInterval) (y : Fin 2) :
    crossEntropy (empiricalDistribution (fun _ : Fin 1 => y)) bernoulliMass predicted =
      binaryCrossEntropy predicted y :=
  crossEntropy_single y bernoulliMass predicted

/-- Multiclass cross-entropy is the same construction for a categorical label. -/
theorem crossEntropy_single_categorical {K : ℕ}
    (predicted : CategoricalParameters K) (y : Fin K) :
    crossEntropy (empiricalDistribution (fun _ : Fin 1 => y)) categoricalMass predicted =
      multiclassCrossEntropy predicted y :=
  crossEntropy_single y categoricalMass predicted

end LossFunctions
