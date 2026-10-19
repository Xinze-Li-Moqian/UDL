module

public import ProbabilityModels.Product
public import LossFunctions.NegativeLogLikelihood

/-!
# Independent multiple-output predictions

Source: Understanding Deep Learning, Section 5.6, equations (5.25)–(5.26).

For each output d, a predictor supplies parameters θᵈ(x, φ) for an existing
family. The input and model parameters are shared, while the output types
and families may differ. With `ι = Fin D`, the joint target has D components.

    p(y | x, φ) = ∏ᵈ pᵈ(yᵈ | θᵈ(x, φ)).

The product family explicitly chooses an independent law at each fixed input
and parameter value. Single-output models are recovered with `ι = Fin 1`.
-/

@[expose] public section

namespace LossFunctions.MultipleOutputs

open MeasureTheory ProbabilityModels
open scoped BigOperators LossFunctions ProbabilityModels

variable {ι X Parameters : Type*} [Fintype ι] {Θ Y : ι → Type*}
variable [∀ d, MeasurableSpace (Y d)]

/-- Construct the joint prediction from per-target families and parameter predictors.

    Y | x, φ ∼ ⨂ᵈ familyᵈ(θᵈ(x, φ)).
-/
noncomputable def predictiveDistribution (families : ∀ d, Family (Θ d) (Y d))
    (f : ∀ d, X → Parameters → Θ d) (φ : Parameters) (x : X) :
    ProbabilityMeasure (∀ d, Y d) :=
  (Family.pi families).predictiveDistribution (fun x φ d => f d x φ) x φ

/-- Extracting any output gives exactly its original predictive distribution. -/
theorem predictiveDistribution_map_eval (families : ∀ d, Family (Θ d) (Y d))
    (f : ∀ d, X → Parameters → Θ d) (φ : Parameters) (x : X) (d : ι) :
    (predictiveDistribution families f φ x : Measure (∀ d, Y d)).map (Function.eval d) =
      (families d (f d x φ) : Measure (Y d)) :=
  Family.pi_map_eval families (fun d => f d x φ) d

variable {families : ∀ d, Family (Θ d) (Y d)} {ν : ∀ d, Measure (Y d)}
variable [∀ d, SigmaFinite (ν d)]

/-- Equation (5.25): the chosen joint density factors over output coordinates. -/
theorem predictiveDensity_eq (p : ∀ d, (families d).Density (ν d))
    (f : ∀ d, X → Parameters → Θ d) (x : X) (φ : Parameters) (y : ∀ d, Y d) :
    Pr[y | (fun d => f d x φ); Family.Density.pi p] =
      ∏ d, Pr[y d | f d x φ; p d] := rfl

/-- The existing negative log-likelihood applied to the joint family. -/
noncomputable def loss {I : ℕ} (p : ∀ d, (families d).Density (ν d))
    (f : ∀ d, X → Parameters → Θ d) (x : Fin I → X) (y : Fin I → ∀ d, Y d) :
    LossFunction ℝ Parameters :=
  negativeLogLikelihood (Family.Density.pi p) (fun x φ d => f d x φ) x y

/-- Equation (5.26): positive observed densities turn output products into loss sums.

    L[φ] = −∑ᵢ ∑ᵈ log pᵈ(yᵢᵈ | θᵈ(xᵢ, φ)).
-/
theorem loss_eq_sum {I : ℕ} (p : ∀ d, (families d).Density (ν d))
    (f : ∀ d, X → Parameters → Θ d) (x : Fin I → X) (y : Fin I → ∀ d, Y d)
    (φ : Parameters) (hpos : ∀ i d, 0 < Pr[y i d | f d (x i) φ; p d]) :
    let L := loss p f x y
    L[φ] = -∑ i : Fin I, ∑ d, Real.log Pr[y i d | f d (x i) φ; p d] := by
  change (∑ i : Fin I, -Real.log (∏ d, p d (f d (x i) φ) (y i d))) = _
  simp_rw [Real.log_prod (fun d _ => ne_of_gt (hpos _ d))]
  simp only [Finset.sum_neg_distrib]

section Single

variable {Parameter Output : Type*} [MeasurableSpace Output]
variable {family : Family Parameter Output} {reference : Measure Output}
variable [SigmaFinite reference]

/-- A single target recovers any existing single-output negative log-likelihood. -/
theorem loss_single {I : ℕ} (p : family.Density reference)
    (f : X → Parameters → Parameter) (x : Fin I → X) (y : Fin I → Output) (φ : Parameters) :
    loss (fun _ : Fin 1 => p) (fun _ => f) x (fun i _ => y i) φ =
      negativeLogLikelihood p f x y φ := by
  change (∑ i : Fin I, -Real.log
    (Family.Density.pi (fun _ : Fin 1 => p) (fun _ => f (x i) φ) (fun _ => y i))) = _
  simp only [Family.Density.pi_single]
  rfl

end Single

end LossFunctions.MultipleOutputs
