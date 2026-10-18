module

public import ProbabilityModels.Variance
public import Mathlib.Probability.Independence.Integration
public import Mathlib.Probability.Independence.Conditional

/-!
# Independence

Use `ProbabilityTheory.IndepFun` for two independent random variables and
`ProbabilityTheory.iIndepFun` for an independent family. Mathlib also supplies
conditional independence in `Probability.Independence.Conditional`.

These are the probabilistic assumptions used to justify product laws for
observations. The construction of a sample likelihood from conditional
densities belongs to `loss-functions/`.

The results below connect independence to event probabilities, joint laws,
integrability and expectations of products, and variances of sums. Products
over a finite family require mutual independence (`iIndepFun`). Pairwise
independence suffices for the variance-of-a-sum theorem. No identical-law
assumption is needed for these results.
-/

@[expose] public section

namespace ProbabilityModels

open MeasureTheory ProbabilityTheory

variable {Ω : Type*} {mΩ : MeasurableSpace Ω} {P : Measure Ω}

section Laws

variable {A B : Type*} [MeasurableSpace A] [MeasurableSpace B]
variable {X : Ω → A} {Y : Ω → B}

/-- Independent variables multiply probabilities of measurable output events. -/
theorem independent_probability_inter (h : IndepFun X Y P)
    {s : Set A} {t : Set B} (hs : MeasurableSet s) (ht : MeasurableSet t) :
    P (X ⁻¹' s ∩ Y ⁻¹' t) = P (X ⁻¹' s) * P (Y ⁻¹' t) :=
  h.measure_inter_preimage_eq_mul s t hs ht

/-- Independence is equivalent to the joint law being the product of marginal laws. -/
theorem independent_iff_jointLaw_eq_prod [IsProbabilityMeasure P]
    (hX : AEMeasurable X P) (hY : AEMeasurable Y P) :
    IndepFun X Y P ↔ P.map (fun ω => (X ω, Y ω)) = (P.map X).prod (P.map Y) :=
  indepFun_iff_map_prod_eq_prod_map_map hX hY

variable {ι : Type*} {Output : ι → Type*} [∀ i, MeasurableSpace (Output i)]
variable {Z : (i : ι) → Ω → Output i}

/-- Mutual independence factors simultaneous events over any finite index set. -/
theorem independent_probability_iInter (h : iIndepFun Z P) (s : Finset ι)
    (events : ∀ i, Set (Output i)) (hEvents : ∀ i ∈ s, MeasurableSet (events i)) :
    P (⋂ i ∈ s, Z i ⁻¹' events i) = ∏ i ∈ s, P (Z i ⁻¹' events i) :=
  h.measure_inter_preimage_eq_mul s hEvents

/-- For a finite independent family, the probability that every variable lies
in its specified measurable set is the product of the individual probabilities.
Use `ι = Fin I` for a collection of `I` observations. -/
theorem independent_probability_forall [Fintype ι] (h : iIndepFun Z P)
    (events : ∀ i, Set (Output i)) (hEvents : ∀ i, MeasurableSet (events i)) :
    P {ω | ∀ i, Z i ω ∈ events i} = ∏ i, P {ω | Z i ω ∈ events i} := by
  simpa [Set.preimage, Set.ofPred_forall] using
    independent_probability_iInter h Finset.univ events (fun i _ => hEvents i)

/-- A finite family is mutually independent exactly when its joint law factors. -/
theorem independent_iff_jointLaw_eq_pi [Fintype ι] [IsProbabilityMeasure P]
    (hZ : ∀ i, AEMeasurable (Z i) P) :
    iIndepFun Z P ↔ P.map (fun ω i => Z i ω) = Measure.pi (fun i => P.map (Z i)) :=
  iIndepFun_iff_map_fun_eq_pi_map hZ

end Laws

section Moments

variable {X Y : Ω → ℝ}

/-- Independence makes the product of integrable variables integrable. -/
theorem integrable_mul_of_independent (h : IndepFun X Y P)
    (hX : Integrable X P) (hY : Integrable Y P) :
    Integrable (fun ω => X ω * Y ω) P :=
  h.integrable_mul hX hY

/-- The expectation of a product factors for independent, integrable variables. -/
theorem expectation_mul_of_independent (h : IndepFun X Y P)
    (hX : Integrable X P) (hY : Integrable Y P) :
    𝔼[(fun ω => X ω * Y ω); P] = 𝔼[X; P] * 𝔼[Y; P] :=
  h.integral_fun_mul_eq_mul_integral hX.aestronglyMeasurable hY.aestronglyMeasurable

/-- Independent variables with finite second moments have additive variances. -/
theorem variance_add_of_independent (h : IndepFun X Y P)
    (hX : MemLp X 2 P) (hY : MemLp Y 2 P) :
    Var[(fun ω => X ω + Y ω); P] = Var[X; P] + Var[Y; P] :=
  h.variance_fun_add hX hY

variable {ι : Type*} {Z : ι → Ω → ℝ}

/-- Pairwise independence already suffices for variance of a finite sum. -/
theorem variance_sum_of_pairwise_independent (s : Finset ι)
    (h : Set.Pairwise (↑s : Set ι) (fun i j => IndepFun (Z i) (Z j) P))
    (hZ : ∀ i ∈ s, MemLp (Z i) 2 P) :
    Var[(fun ω => ∑ i ∈ s, Z i ω); P] = ∑ i ∈ s, Var[Z i; P] := by
  simpa only [Finset.sum_fn] using! IndepFun.variance_sum hZ h

/-- A finite product of mutually independent, integrable variables is integrable. -/
theorem integrable_prod_of_independent [Fintype ι] (h : iIndepFun Z P)
    (hZ : ∀ i, Integrable (Z i) P) : Integrable (fun ω => ∏ i, Z i ω) P := by
  have : IsProbabilityMeasure P := h.isProbabilityMeasure
  classical
  have hFinset (s : Finset ι) : Integrable (fun ω => ∏ i ∈ s, Z i ω) P := by
    induction s using Finset.induction_on with
    | empty => simp
    | @insert i s hi ih =>
      have hIndep := (h.indepFun_finsetProd_of_notMem₀
        (fun j => (hZ j).aemeasurable) hi).symm
      have ih' : Integrable (∏ j ∈ s, Z j) P := by
        simpa only [Finset.prod_fn] using! ih
      simpa only [Finset.prod_insert hi, Finset.prod_apply] using!
        integrable_mul_of_independent hIndep (hZ i) ih'
  exact hFinset Finset.univ

/-- Expectations of finite products factor under mutual independence. -/
theorem expectation_prod_of_independent [Fintype ι] (h : iIndepFun Z P)
    (hZ : ∀ i, Integrable (Z i) P) :
    𝔼[(fun ω => ∏ i, Z i ω); P] = ∏ i, 𝔼[Z i; P] :=
  h.integral_fun_prod_eq_prod_integral (fun i => (hZ i).aestronglyMeasurable)

end Moments

end ProbabilityModels
