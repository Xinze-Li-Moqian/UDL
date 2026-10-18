module

public import ProbabilityModels.Density
public import Mathlib.Geometry.Convex.ConvexSpace.Defs
public import Mathlib.Probability.ProbabilityMassFunction.Constructions

/-!
# Categorical distributions

Source: Understanding Deep Learning, Figure 5.9 and equation (5.21).

    P(Y = k | λ) = λₖ,    λₖ ≥ 0,    ∑ₖ λₖ = 1.

Mathlib's `Convexity.StdSimplex` encodes the probability-vector constraints. Its weights
give a finite `PMF`, which supplies a probability measure. The certified mass
function uses counting measure and the same family interface as Bernoulli.
Labels use `Fin K`, indexed from 0 rather than the book's 1 through K.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory
open scoped BigOperators

/-- A probability vector on K labels, using Mathlib's standard simplex. -/
abbrev CategoricalParameters (K : ℕ) := Convexity.StdSimplex ℝ (Fin K)

/-- Package specified probabilities with their nonnegativity and normalization proofs. -/
noncomputable def CategoricalParameters.ofProbabilities {K : ℕ}
    (p : Fin K → ℝ) (hnonneg : ∀ k, 0 ≤ p k) (hsum : ∑ k, p k = 1) :
    CategoricalParameters K where
  weights := Finsupp.equivFunOnFinite.symm p
  nonneg k := by simpa using hnonneg k
  total := by simpa [Finsupp.sum_fintype] using hsum

@[simp] theorem CategoricalParameters.ofProbabilities_weights {K : ℕ}
    (p : Fin K → ℝ) (hnonneg : ∀ k, 0 ≤ p k) (hsum : ∑ k, p k = 1) (k : Fin K) :
    (CategoricalParameters.ofProbabilities p hnonneg hsum).weights k = p k := by
  simp [CategoricalParameters.ofProbabilities]

/-- Adapt a probability vector to Mathlib's finite probability mass function. -/
noncomputable def categoricalPMF {K : ℕ} (p : CategoricalParameters K) : PMF (Fin K) :=
  PMF.ofFintype (fun k => ENNReal.ofReal (p.weights k)) (by
    rw [← ENNReal.ofReal_sum_of_nonneg (fun k _ => p.weights_nonneg k)]
    simp)

/-- The categorical probability family indexed by its vector of probabilities. -/
noncomputable def categoricalFamily {K : ℕ} : Family (CategoricalParameters K) (Fin K) :=
  fun p => (categoricalPMF p).toMeasure.toProbabilityMeasure

@[simp] theorem coe_categoricalFamily {K : ℕ} (p : CategoricalParameters K) :
    (categoricalFamily p : Measure (Fin K)) = (categoricalPMF p).toMeasure := rfl

/-- Equation (5.21): a label's mass is its specified probability.

    p(k | λ) = λₖ.
-/
noncomputable def categoricalMass {K : ℕ} :
    (categoricalFamily (K := K)).Density Measure.count where
  toFun p k := p.weights k
  nonneg p k := p.weights_nonneg k
  measurable p := measurable_of_finite _
  withDensity_eq p := by
    apply Measure.ext_of_singleton
    intro k
    rw [withDensity_apply _ (measurableSet_singleton k)]
    simp [categoricalPMF]

@[simp] theorem categoricalMass_apply {K : ℕ} (p : CategoricalParameters K) (k : Fin K) :
    categoricalMass p k = p.weights k := rfl

/-- The probability masses sum to one. -/
@[simp] theorem sum_categoricalMass {K : ℕ} (p : CategoricalParameters K) :
    ∑ k : Fin K, categoricalMass p k = 1 := p.total_of_fintype

/-- The chosen mass agrees with the probability of the corresponding singleton event. -/
theorem categoricalFamily_real_singleton {K : ℕ} (p : CategoricalParameters K) (k : Fin K) :
    (categoricalFamily p : Measure (Fin K)).real {k} = categoricalMass p k := by
  simp [Measure.real, categoricalPMF,
    ENNReal.toReal_ofReal (p.weights_nonneg k)]

end ProbabilityModels
