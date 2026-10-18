module

public import LossFunctions.MaximumLikelihood
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Log-likelihood

Source: Understanding Deep Learning, Section 5.1.3, equation (5.3).

The log-likelihood sums the logarithms of the predictive densities. For
positive densities at the observations, this equals the logarithm of the
likelihood. Strict monotonicity of `log` preserves the ordering of candidates
and hence the maximizers, while changing the objective values.

The definition uses Mathlib's totalized `Real.log`, for which `log 0 = 0`.
The likelihood equivalences below explicitly require strictly positive
observed densities. Zero-density cases with extended logarithms are not
covered by these real-valued results.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open Real (log)
open scoped ProbabilityModels Optimization

variable {X Y Θ Parameters : Type*} [MeasurableSpace Y]
variable {family : Family Θ Y} {ν : Measure Y} {I : ℕ}

/-- Equation (5.3): sum log densities at the observations, with the data fixed. -/
noncomputable def logLikelihood (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) : ℝ :=
  ∑ i : Fin I, log Pr[y i | f (x i) φ; p]

/-- The candidate `φ̂` maximizes the sum of log densities. -/
def IsMaximumLogLikelihoodEstimator (p : family.Density ν) (f : X → Parameters → Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) : Prop :=
  «φ̂» ∈ argmax φ, ∑ i : Fin I, log Pr[y i | f (x i) φ; p]

/-- For positive observed densities, the sum of logs is the log of the product. -/
theorem logLikelihood_eq_log_likelihood (p : family.Density ν)
    (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y) (φ : Parameters)
    (hpos : ∀ i, 0 < Pr[y i | f (x i) φ; p]) :
    logLikelihood p f x y φ = log (likelihood p f x y φ) := by
  exact (Real.log_prod (fun i _ => ne_of_gt (hpos i))).symm

/-- Taking logs preserves the ordering of two candidates with positive densities. -/
theorem logLikelihood_le_logLikelihood_iff (p : family.Density ν)
    (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y) (φ ψ : Parameters)
    (hφ : ∀ i, 0 < Pr[y i | f (x i) φ; p])
    (hψ : ∀ i, 0 < Pr[y i | f (x i) ψ; p]) :
    logLikelihood p f x y φ ≤ logLikelihood p f x y ψ ↔
      likelihood p f x y φ ≤ likelihood p f x y ψ := by
  rw [logLikelihood_eq_log_likelihood p f x y φ hφ,
    logLikelihood_eq_log_likelihood p f x y ψ hψ]
  exact Real.log_le_log_iff (likelihood_pos p f x y φ hφ) (likelihood_pos p f x y ψ hψ)

/-- With positive observed densities for every candidate, the maximizers coincide.
Neither existence nor uniqueness of a maximizer is assumed. -/
theorem isMaximumLikelihoodEstimator_iff_isMaximumLogLikelihoodEstimator
    (p : family.Density ν) (f : X → Parameters → Θ) (x : Fin I → X) (y : Fin I → Y)
    (hpos : ∀ φ i, 0 < Pr[y i | f (x i) φ; p]) («φ̂» : Parameters) :
    IsMaximumLikelihoodEstimator p f x y «φ̂» ↔
      IsMaximumLogLikelihoodEstimator p f x y «φ̂» := by
  change (∀ φ, likelihood p f x y φ ≤ likelihood p f x y «φ̂») ↔
    (∀ φ, logLikelihood p f x y φ ≤ logLikelihood p f x y «φ̂»)
  exact forall_congr' fun φ =>
    (logLikelihood_le_logLikelihood_iff p f x y φ «φ̂» (hpos φ) (hpos «φ̂»)).symm

end LossFunctions
