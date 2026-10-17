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
gradient descent, discriminative versus generative models, and shallow neural networks.

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

This topic is being developed one definition at a time through Sections 3.1–3.2.
The files separate definitions, proofs, and demonstrations:

```text
neural-networks/
├── Bias.lean
├── ActivationFunction.lean
├── ShallowNetwork.lean          # Library entry point
├── ShallowNetwork/
│   ├── Model.lean               # Shallow network definitions
│   ├── WidthThree.lean          # The book's three-unit case
│   ├── MultiInput.lean          # Networks with several inputs
│   ├── Verification.lean        # Entry point for definition checks
│   ├── Verification/
│   │   ├── Model.lean           # One-input model at zero width
│   │   ├── WidthThree.lean      # The book's width-three formula
│   │   └── MultiInput.lean      # Multi-input model checks
│   ├── UniversalApproximation.lean  # Entry point for the theorem
│   └── UniversalApproximation/
│       ├── Representable.lean   # Functions computed by ReLU networks
│       ├── Interpolation.lean   # Ramps and piecewise-linear interpolation
│       └── OneInput.lean        # The theorem for one input
└── Example/
    ├── Plot.lean                # Shared sampling and plotting helpers
    ├── WithBias.lean            # Numerical example and plots
    └── WithoutBias.lean         # Numerical example and plots
```

`neural-networks/ActivationFunction.lean` defines the type of a scalar activation
function, `Scalar → Scalar`, and ReLU: preserve nonnegative inputs and return zero
for negative inputs. Call `relu z` for zero bias, or `relu z bias` to add a bias
before rectifying; `relu z (bias := bias)` is the equivalent named form.

`neural-networks/Bias.lean` defines `addBias bias value = bias + value`, the
constant offset represented by the orange edges in the network diagram.

`neural-networks/ShallowNetwork/Model.lean` imports `ActivationFunction` and
defines the default shallow network for any hidden width `D`, following
equations (3.5)–(3.6).
It defines `hiddenUnit` as `activation (addBias bias (weight * x))` and
`ShallowNetworkParameters Scalar D` with an output bias and three
collections indexed by `Fin D`: output weights, hidden biases, and hidden weights.
`shallowNetwork activation x Φ` sums the weighted hidden activations and adds the
output bias. Lean numbers the units from `0` to `D - 1`.

`networkCapacity Φ` returns the number of hidden units `D`, the width-based
capacity measure used in Section 3.2. It depends on the network architecture,
not the parameter values, and is not a norm. Both examples report capacity `3`,
even though the chosen weights in the example without biases yield zero output.

`neural-networks/ShallowNetwork/WidthThree.lean` imports `ShallowNetwork.Model`
and places the book's three-unit case in the `NeuralNetworks.WidthThree`
namespace. `WidthThree.Parameters` retains its ten named parameters;
`WidthThree.Parameters.toShallowNetworkParameters` converts them to
`ShallowNetworkParameters Scalar 3`. Both `WidthThree.networkWithParameters` and
`WidthThree.network` reuse `shallowNetwork`, giving equation (3.4),
`φ₀ + φ₁ * h₁ + φ₂ * h₂ + φ₃ * h₃`.

`neural-networks/ShallowNetwork/MultiInput.lean` generalizes the model to `Dᵢ`
inputs (Sections 3.3–3.4), still with one output. The input is a vector
`x : Fin Dᵢ → Scalar`, and each hidden unit has a bias and a weight vector:
`h_d = a[θ_d0 + ∑ i, θ_di * x_i]`. The namespace `NeuralNetworks.MultiInput`
contains `hiddenUnit`, `Parameters Scalar Dᵢ D`, and `network`.
`ShallowNetworkParameters.toMultiInput` views one-input parameters as the case
`Dᵢ = 1`.

`neural-networks/ShallowNetwork/Verification/` checks that the definitions mean
what they claim. These are consistency checks, not new mathematical results:

- `Model.lean`: zero width leaves only the output bias (`shallowNetwork_zero_width`).
- `WidthThree.lean`: `WidthThree.networkWithParameters_eq` and
  `WidthThree.network_eq` verify that the wrappers use the same model and expand
  to the book's formula.
- `MultiInput.lean`: `MultiInput.network_zero_width` is the multi-input
  zero-width result, and `shallowNetwork_eq_multiInput` proves that the
  one-input model equals the multi-input model with `Dᵢ = 1`.

`neural-networks/ShallowNetwork/Verification.lean` imports all three.
Definitions depend only on earlier layers:
`WidthThree` and `MultiInput` import `Model`, which imports `ActivationFunction`, which imports
`Bias`. Proofs build on those definitions; examples and plots build on the library.
`neural-networks/ShallowNetwork.lean` is the entry point importing `Model`,
`WidthThree`, `MultiInput`, `Verification`, and `UniversalApproximation`. Demonstrations remain
separate imports.

`neural-networks/ShallowNetwork/UniversalApproximation/` holds the universal
approximation theorem and its supporting lemmas. `OneInput.lean` states and
proves it for one input. The proposition `UniversalApproximation` says: for every
continuous function on a compact subset of the real line and every positive
error tolerance, some width and parameters give a ReLU network whose error is
below that tolerance throughout the set. The width and parameters are chosen
before the input point. `universalApproximation` proves it: extend the target
continuously to the whole line (Tietze), enclose the set in an interval where
the extension is uniformly continuous, and interpolate it linearly on a fine
grid. The supporting lemmas build that argument in layers:

- `Representable.lean`: `ReLURepresentable f` means some shallow ReLU network
  computes `f` exactly; such functions are closed under constants, sums, and
  scalar multiples.
- `Interpolation.lean`: a ramp from `0` to `1` made of two ReLU units, so `N`
  grid cells give a network of width `2 * N`; the interpolant's value on one grid
  cell; and the error bound for a convex combination.

`neural-networks/ShallowNetwork/UniversalApproximation.lean` imports the theorem.
The multi-input version, approximating surfaces and higher-dimensional graphs,
is not yet formalized.

Run `lake build` to check these modules along with the existing supervised-learning
modules.

`neural-networks/Example/WithBias.lean` and `WithoutBias.lean` each contain a
complete numerical and visual example. Their namespaces are
`NeuralNetworks.Examples.WithBias` and `NeuralNetworks.Examples.WithoutBias`.
Each defines `parameters : ShallowNetworkParameters ℚ 3` and uses `#eval` to print
hidden activations and the output of `shallowNetwork` with exact rational
arithmetic. Both examples use hidden weights `(1, 1, 1)`, output weights
`(1, -2, 1)`, and zero output bias. **With bias** uses hidden biases `(0, -1, -2)`;
**without bias** sets all hidden biases to zero.

The example with bias gives `ReLU(x) - 2 * ReLU(x - 1) + ReLU(x - 2)`.
At `x = 3/2`, the hidden values are `(3/2, 1/2, 0)` and the output is `1/2`.
Without bias, the hidden units coincide, and these particular output weights
cancel to give zero. The comparison illustrates the effect of the hidden biases.
Run either example:

```sh
lake env lean neural-networks/Example/WithBias.lean
lake env lean neural-networks/Example/WithoutBias.lean
```

Open either example file in VS Code with the Lean extension and place the cursor
on `#html plot`. The Lean Infoview shows ten charts: three affine inputs, three
ReLU activations, three weighted contributions, and the final output. Each file
also exposes its sampled chart data as JSON through `plotData`.

Both examples import `neural-networks/Example/Plot.lean` for shared sampling
and ProofWidgets plotting functions. This helper imports `ShallowNetwork.Model`
and is independent of the example parameters. Each example supplies its own
parameters to the numerical evaluation and the charts.
