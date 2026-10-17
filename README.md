# UDL — Deep Learning

A repository for learning deep learning, with
[*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) by
Simon J. D. Prince as the main reference.

The book is freely available as a PDF from its official website under a
[Creative Commons Attribution–NonCommercial–NoDerivatives (CC BY-NC-ND) license](https://github.com/udlbook/udlbook/blob/main/LICENSE).
The [companion repository](https://github.com/udlbook/udlbook) contains notebooks,
slides, and other learning materials.

The repository brings together theory notes, implementations, and experiments.
Lean 4 is used to express mathematical definitions and verify selected results.

Current topics include supervised learning, linear regression, loss functions,
gradient descent, discriminative versus generative models, and activation functions.

Ideas and exercises from the book are integrated into modules by topic, with
source references in the code.

## Working Approach

For each topic or experiment, aim to record:

1. The question being explored and the relevant background.
2. The assumptions, mathematical ideas, and method.
3. An implementation, with a simple baseline where useful.
4. The setup and instructions needed to reproduce the results.
5. What worked, what failed, and what remains unclear.

Keep the structure simple and let shared tooling grow out of recurring needs. Progress means being able to explain the ideas, reproduce the results, and revisit earlier work with a clearer understanding.

## Lean Verification

The project pins Lean and Mathlib to version 4.35.0. From the repository root, run
`lake exe cache get` to download Mathlib's compiled dependencies, then `lake build`
to check the definitions and proofs.

Proofs are grouped by topic in `supervised-learning/Verification/`:

- `LossFormula.lean`: expansion of the least-squares loss.
- `Optimality.lean`: equivalence of the minimizer conditions.
- `TrainingIteration.lean`: zero steps and training in stages.
- `TrainingLoss.lean`: the loss bound assuming every update does not increase loss.
- `GradientFormula.lean`: the two partial derivatives over the reals for each model direction,
  including explicit formulas for the discriminative loss.
- `StepSize.lean`: a sufficient step-size condition for nonincreasing discriminative loss.
- `LeastSquaresSolution.lean`: normal equations, the closed-form parameters and proof of
  existence when the determinant is nonzero, global minimality, and uniqueness.
- `ModelComparison.lean`: an exact three-point counterexample showing that globally
  optimal discriminative and generative fits can predict differently.

`supervised-learning/Verification.lean` imports all verification topics. The training proofs
do not establish convergence or that a finite run finds an exact minimizer.

Material from Understanding Deep Learning is integrated by mathematical topic.
Chapter 2, Problems 2.1–2.3 motivate the gradient formulas, closed-form solution,
and model comparison respectively; their numbers remain as source references in
those modules. Definitions live with the models, proofs in `Verification/`, and
executable experiments in `Examples.lean`. The model-comparison proof and the
training experiment share the same three-point dataset.

`supervised-learning/SupervisedLearning.lean` contains the shared input, output,
training-dataset, and loss-function types. Models are grouped by direction:

- `Discriminative/`: `Model.lean` defines `y = f[x, φ]` and direct inference;
  `LinearRegression.lean` defines its linear model and loss;
  `LinearRegressionTraining.lean` defines gradient descent training.
- `Generative/`: `Model.lean` defines `x = g[y, φ]`;
  `LinearRegression.lean` defines its linear model, loss, minimizer condition,
  and inverse inference requiring a nonzero slope;
  `LinearRegressionTraining.lean` supplies its gradient for training.

Both training files call the same `Optimization.gradientDescent` function in
`optimization/GradientDescent.lean`. This shared optimizer updates parameters;
each model supplies the gradient of its own loss. The discriminative slope
gradient uses `xᵢ`, while the generative slope gradient uses `yᵢ`.

`supervised-learning/Examples.lean` uses the same affine formula in three short
scenarios: discriminative prediction, generative evaluation, and inference by
inverting the generative model. It also trains both directions on the same three
pairs for 500 updates, printing parameters, loss before and after training, and
predictions. Arithmetic is rational; training results are displayed as rounded
decimals. Each model's loss measures a different output quantity, so compare it
to its own initial loss. Run `lake env lean supervised-learning/Examples.lean`
to see the three exact values `7`, `7`, and `1`, followed by the training results.

## Chapter 3: Shallow Neural Networks

This topic is being developed one definition at a time, starting with Section 3.1.
`neural-networks/ActivationFunction.lean` defines the type of a scalar activation
function, `Scalar → Scalar`, and ReLU: preserve nonnegative inputs and return zero
for negative inputs. It also defines a hidden unit, `activation (bias + weight * x)`.
Run `lake build` to check these modules along with the existing supervised-learning
modules.

For a visual example, open `neural-networks/ActivationPlot.lean` in VS Code with
the Lean extension and place the cursor on `#html activationPlots`. The Lean
Infoview shows `ReLU(x)` and `ReLU(x - 1)` using ProofWidgets. Both plots evaluate
the existing definitions with `Float` samples and connect the samples with
straight segments. Change the bias and weight in the `hiddenUnit` call to explore
how the graph changes.
