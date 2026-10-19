module

public import ProbabilityModels.Density
public import Mathlib.MeasureTheory.Measure.FiniteMeasurePi
public import Mathlib.MeasureTheory.Integral.Pi
public import Mathlib.Probability.Independence.Basic

/-!
# Finite products of probability families

Each output coordinate may have its own output type, parameter type, and
distribution family. The product construction chooses an independent joint law:

    Pθ = ⨂ᵈ Pᵈθᵈ,    p(y | θ) = ∏ᵈ pᵈ(yᵈ | θᵈ).

The density theorem uses sigma-finite reference measures, covering Lebesgue
and finite counting measures. Independence is proved for the constructed law;
it is a modeling choice about the outputs at fixed parameters.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory
open scoped BigOperators

variable {ι : Type*} [Fintype ι] {Θ Y : ι → Type*}
variable [∀ d, MeasurableSpace (Y d)]

/-- Combine finitely many families into their independent product family. -/
noncomputable def Family.pi (families : ∀ d, Family (Θ d) (Y d)) :
    Family (∀ d, Θ d) (∀ d, Y d) :=
  fun θ => ProbabilityMeasure.pi (fun d => families d (θ d))

@[simp] theorem Family.coe_pi (families : ∀ d, Family (Θ d) (Y d)) (θ : ∀ d, Θ d) :
    (Family.pi families θ : Measure (∀ d, Y d)) =
      Measure.pi (fun d => (families d (θ d) : Measure (Y d))) := rfl

/-- Each coordinate of the product law has exactly its original distribution. -/
theorem Family.pi_map_eval (families : ∀ d, Family (Θ d) (Y d))
    (θ : ∀ d, Θ d) (d : ι) :
    (Family.pi families θ : Measure (∀ d, Y d)).map (Function.eval d) =
      (families d (θ d) : Measure (Y d)) :=
  (measurePreserving_eval (fun d => (families d (θ d) : Measure (Y d))) d).map_eq

/-- The output coordinates are mutually independent under the constructed joint law. -/
theorem Family.pi_independent (families : ∀ d, Family (Θ d) (Y d)) (θ : ∀ d, Θ d) :
    iIndepFun (fun d (y : ∀ d, Y d) => y d)
      (Family.pi families θ : Measure (∀ d, Y d)) :=
  iIndepFun_pi (X := fun _ => id) (fun _ => aemeasurable_id)

variable {families : ∀ d, Family (Θ d) (Y d)} {ν : ∀ d, Measure (Y d)}
variable [∀ d, SigmaFinite (ν d)]

/-- The product of certified component densities certifies the product probability law.

    p(y | θ) = ∏ᵈ pᵈ(yᵈ | θᵈ),    reference measure = ⨂ᵈ νᵈ.
-/
noncomputable def Family.Density.pi (p : ∀ d, (families d).Density (ν d)) :
    (Family.pi families).Density (Measure.pi ν) where
  toFun θ y := ∏ d, p d (θ d) (y d)
  nonneg θ y := Finset.prod_nonneg (fun d _ => (p d).nonneg _ _)
  measurable θ := Finset.measurable_prod _ (fun d _ =>
    ((p d).measurable (θ d)).comp (measurable_pi_apply d))
  withDensity_eq θ := by
    classical
    apply (Measure.pi_eq _).symm
    intro s hs
    rw [withDensity_apply _ (MeasurableSet.univ_pi hs), Measure.restrict_pi_pi]
    rw [← ofReal_integral_eq_lintegral_ofReal
      (Integrable.fintype_prod_dep (fun d => ((p d).integrable (θ d)).restrict))
      (Filter.Eventually.of_forall (fun y => Finset.prod_nonneg
        (fun d _ => (p d).nonneg _ _)))]
    rw [integral_fintype_prod_eq_prod]
    rw [ENNReal.ofReal_prod_of_nonneg (fun d _ =>
      integral_nonneg (fun y => (p d).nonneg _ y))]
    apply Finset.prod_congr rfl
    intro d _
    rw [ofReal_integral_eq_lintegral_ofReal ((p d).integrable (θ d)).restrict
      (Filter.Eventually.of_forall (fun y => (p d).nonneg _ y))]
    exact ((p d).probability_eq_lintegral (θ d) (hs d)).symm

/-- Equation (5.25), at the level of chosen probability masses or densities. -/
@[simp] theorem Family.Density.pi_apply (p : ∀ d, (families d).Density (ν d))
    (θ : ∀ d, Θ d) (y : ∀ d, Y d) :
    Family.Density.pi p θ y = ∏ d, p d (θ d) (y d) := rfl

section Single

variable {Parameter Output : Type*} [MeasurableSpace Output]
variable {family : Family Parameter Output} {reference : Measure Output}
variable [SigmaFinite reference]

/-- With one target, the joint density is the original single-target density. -/
@[simp] theorem Family.Density.pi_single (p : family.Density reference)
    (θ : Parameter) (y : Output) :
    Family.Density.pi (fun _ : Fin 1 => p) (fun _ => θ) (fun _ => y) = p θ y := by
  simp only [Family.Density.pi_apply, Fin.prod_univ_one]

/-- Projecting the sole coordinate recovers any original single-output family. -/
theorem Family.pi_single_map_zero (family : Family Parameter Output) (θ : Parameter) :
    (Family.pi (fun _ : Fin 1 => family) (fun _ => θ) : Measure (Fin 1 → Output)).map
        (Function.eval 0) = (family θ : Measure Output) :=
  Family.pi_map_eval (fun _ : Fin 1 => family) (fun _ => θ) 0

end Single

end ProbabilityModels
