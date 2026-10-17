module

public import Network.Model
public import ProbabilityModels.Family

/-!
# A network predicts a distribution

Source: Understanding Deep Learning, Section 5.1.1.

The network computes a vector, `toParameters` converts it to distribution
parameters `θ`, and `family θ` supplies a probability distribution on outputs `Y`.
-/

@[expose] public section

namespace LossFunctions

open MeasureTheory NeuralNetworks ProbabilityModels
open scoped ProbabilityModels

/-- Use an existing network to select a distribution from a chosen family.

- `Y` is the observation space; `Θ` is the distribution parameter space.
- `family : Θ → ProbabilityMeasure Y` specifies the distribution family.
- `A` and `a` specify the network architecture and activation function.
- `φ` contains the network weights and biases; `x` is the input vector.
- `toParameters` converts the network output into valid parameters `θ`.
  For example, it can pair a predicted mean with a fixed positive variance.

The result `Pr[· | θ; family]` is the entire distribution `family θ`.
Varying `x` or `φ` can select another member of this family; choosing a different
family requires passing a different `family` and a compatible `toParameters`.

After fixing these choices as a function `model φ x`, its prediction can be
written `Pr[· | x; model, φ]` using the notation from `ProbabilityModels.Family`.
-/
def networkPredictiveDistribution {Y Θ : Type*} [MeasurableSpace Y]
    {A : Architecture} (family : Family Θ Y) (a : ActivationFunction ℝ)
    (toParameters : (Fin A.outputs → ℝ) → Θ)
    (φ : NetworkParameters ℝ A) (x : Fin A.inputs → ℝ) : ProbabilityMeasure Y :=
  let θ := toParameters (network a φ x)
  -- The notation from `ProbabilityModels.Family` expands to `family θ`.
  Pr[· | θ; family]

/-- The prediction at input `x` is exactly the family member with parameters
`θ = toParameters (network a φ x)`. Both sides denote the entire distribution
on outputs `Y`, not a distribution on the input coordinates. -/
@[simp] theorem networkPredictiveDistribution_eq {Y Θ : Type*} [MeasurableSpace Y]
    {A : Architecture} (family : Family Θ Y) (a : ActivationFunction ℝ)
    (toParameters : (Fin A.outputs → ℝ) → Θ)
    (φ : NetworkParameters ℝ A) (x : Fin A.inputs → ℝ) :
    Pr[· | x; networkPredictiveDistribution (A := A) family a toParameters, φ] =
      Pr[· | toParameters (network a φ x); family] := rfl

end LossFunctions
