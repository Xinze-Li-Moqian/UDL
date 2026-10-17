module

public import Verification.LeastSquaresSolution
public import Generative.LinearRegression
public import Mathlib.Tactic.NormNum

/-!
# Comparing discriminative and generative linear regression

Source: Understanding Deep Learning, Chapter 2, Problem 2.3*.

The generative model `x = g[y, φ] = φ₀ + φ₁*y` fits the same training pairs
with loss `L[φ] = Σᵢ (φ₀ + φ₁*yᵢ - xᵢ)²`.
Solving `x = φ₀ + φ₁*y` gives `y = (x - φ₀)/φ₁` when `φ₁ ≠ 0`.
These are `leastSquaresLoss` and `inverseModel` in `Generative/LinearRegression.lean`.

For the three pairs `(0, 0), (1, 1), (2, 1)`, global least-squares fits are:
- Discriminative: `y = 1/6 + (1/2)*x`.
- Generative: `x = (3/2)*y`, hence `y = (2/3)*x` during inference.

At the same input `x = 3/2`, the predictions are `11/12` and `1`.
Thus the two fitting procedures do not always give the same predictions.
They minimize errors in different directions: `y` for the discriminative loss
and `x` for the generative loss.

The theorem below proves this exact counterexample, including global optimality
of both parameter pairs. `Examples.lean` runs gradient descent on the same data;
that finite training run gives approximate fits, while this proof uses exact ones.
-/

@[expose] public section

namespace SupervisedLearning.ModelComparison

/-- The same three pairs, usable over the reals for proofs and rationals for evaluation. -/
def threePoints (Scalar : Type) [Zero Scalar] [One Scalar] [OfNat Scalar 2] :
    TrainingDataset Scalar 1 1 3 :=
  let xs : Vector Scalar 3 := #v[0, 1, 2]
  let ys : Vector Scalar 3 := #v[0, 1, 1]
  { x := fun i => #v[xs[i]], y := fun i => #v[ys[i]] }

/-- Even globally optimal fits can predict different outputs from the same input. -/
theorem fits_can_predict_differently :
    ∃ φD : Discriminative.LinearRegression.Parameters ℝ,
      ∃ φG : Generative.LinearRegression.Parameters ℝ,
        ∃ hSlope : φG.φ₁ ≠ 0,
          Discriminative.LinearRegression.IsLeastSquaresMinimizer (threePoints ℝ) φD ∧
          Generative.LinearRegression.IsLeastSquaresMinimizer (threePoints ℝ) φG ∧
          (Discriminative.LinearRegression.linearModel #v[3 / 2] φD)[0] ≠
            (Generative.LinearRegression.inverseModel #v[3 / 2] φG hSlope)[0] := by
  refine ⟨⟨1 / 6, 1 / 2⟩, ⟨0, 3 / 2⟩, by norm_num, ?_, ?_, ?_⟩
  -- The discriminative fit satisfies the normal equations.
  · apply Discriminative.LinearRegression.isLeastSquaresMinimizer_of_stationary
    norm_num [Discriminative.LinearRegression.IsLeastSquaresStationary,
      threePoints, Fin.sum_univ_succ]
  -- Swap the two data columns to reuse the same least-squares argument for g.
  · let reversed : TrainingDataset ℝ 1 1 3 :=
      { x := (threePoints ℝ).y, y := (threePoints ℝ).x }
    have hG : Discriminative.LinearRegression.IsLeastSquaresMinimizer
        reversed ⟨0, 3 / 2⟩ := by
      apply Discriminative.LinearRegression.isLeastSquaresMinimizer_of_stationary
      norm_num [Discriminative.LinearRegression.IsLeastSquaresStationary,
        reversed, threePoints, Fin.sum_univ_succ]
    intro ψ
    exact hG ⟨ψ.φ₀, ψ.φ₁⟩
  -- At x = 3/2, the predictions are 11/12 and 1.
  · change (1 / 6 : ℝ) + (1 / 2) * (3 / 2) ≠ ((3 / 2) - 0) / (3 / 2)
    norm_num

end SupervisedLearning.ModelComparison
