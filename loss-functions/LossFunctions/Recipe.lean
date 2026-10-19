module

public import LossFunctions.CrossEntropy
public import LossFunctions.Inference

/-!
# A recipe for probabilistic prediction and training losses

Source: Understanding Deep Learning, Section 5.2.

Fill in a distribution family, its reference measure and certified density,
and a predictor of valid distribution parameters. The shared operations give:

    θ = f(x, φ),    prediction = p(· | θ),
    L[φ] = −∑ᵢ log p(yᵢ | f(xᵢ, φ)).

Fitting and point prediction are optimality conditions, not optimization
algorithms. A recipe does not assert that an optimum exists.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory ProbabilityModels
open scoped ProbabilityModels LossFunctions Optimization

/-- Fill in these four fields to obtain a probabilistic model and its training loss.
`Y` is the observed output type; `Θ` is the distribution-parameter type. -/
structure Recipe (X Y Parameters Θ : Type*) [MeasurableSpace Y] where
  /-- Step 1: choose the output distribution family p(· | θ). -/
  family : Family Θ Y
  /-- The reference measure: e.g. volume for Gaussian, count for categorical. -/
  referenceMeasure : Measure Y
  /-- A normalized density or mass function for the chosen family. -/
  density : family.Density referenceMeasure
  /-- Step 2: compute valid distribution parameters θ = f(x, φ). -/
  predictParameters : X → Parameters → Θ

variable {X Y Parameters Θ : Type*} [MeasurableSpace Y] {I : ℕ}

/-- Step 4: return the complete distribution for an input and fixed model parameters. -/
def Recipe.predictiveDistribution (recipe : Recipe X Y Parameters Θ)
    (φ : Parameters) (x : X) : ProbabilityMeasure Y :=
  recipe.family.predictiveDistribution recipe.predictParameters x φ

/-- Step 3: construct the common training objective from paired observations.

    L[φ] = −∑ᵢ log p(yᵢ | θ(xᵢ, φ)).
-/
noncomputable def Recipe.loss (recipe : Recipe X Y Parameters Θ)
    (x : Fin I → X) (y : Fin I → Y) : LossFunction ℝ Parameters :=
  negativeLogLikelihood recipe.density recipe.predictParameters x y

/-- The generated loss is exactly the shared negative log-likelihood formula. -/
theorem Recipe.loss_apply (recipe : Recipe X Y Parameters Θ)
    (x : Fin I → X) (y : Fin I → Y) (φ : Parameters) :
    let L := recipe.loss x y
    L[φ] = -∑ i : Fin I,
      Real.log Pr[y i | recipe.predictParameters (x i) φ; recipe.density] :=
  negativeLogLikelihood_apply recipe.density recipe.predictParameters x y φ

/-- Step 3's fitting criterion, with no solver or existence assumption.

    φ̂ ∈ argmin φ, L[φ].
-/
def Recipe.IsFit (recipe : Recipe X Y Parameters Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) : Prop :=
  «φ̂» ∈ argmin φ, (recipe.loss x y)[φ]

/-- Step 4's point prediction criterion: maximize the predicted density over y. -/
def Recipe.IsPointPrediction (recipe : Recipe X Y Parameters Θ)
    (x : X) («φ̂» : Parameters) (ŷ : Y) : Prop :=
  IsModePrediction recipe.density recipe.predictParameters x «φ̂» ŷ

/-- The same fitted parameters minimize empirical conditional cross-entropy. -/
theorem Recipe.isFit_iff_minimizes_crossEntropy
    [MeasurableSpace X] [MeasurableSingletonClass X] [MeasurableSingletonClass Y]
    [NeZero I] (recipe : Recipe X Y Parameters Θ)
    (x : Fin I → X) (y : Fin I → Y) («φ̂» : Parameters) :
    recipe.IsFit x y «φ̂» ↔
      «φ̂» ∈ argmin φ,
        conditionalCrossEntropy (empiricalDistribution (fun i => (x i, y i)))
          recipe.density recipe.predictParameters φ :=
  (mem_argmin_conditionalCrossEntropy_iff recipe.density recipe.predictParameters
    x y «φ̂»).symm

end LossFunctions
