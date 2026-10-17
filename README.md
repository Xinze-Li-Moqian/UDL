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
│   ├── UniversalApproximation/
│   │   ├── Representable.lean   # Functions computed by ReLU networks
│   │   ├── Interpolation.lean   # Ramps and piecewise-linear interpolation
│   │   ├── Ridge.lean           # Ridge functions and the exponential ridge algebra
│   │   ├── OneInput.lean        # The theorem for one input
│   │   └── MultiInput.lean      # The theorem for several inputs
│   ├── LinearRegions.lean       # Entry point for the region bound
│   └── LinearRegions/
│       ├── ActivationPattern.lean # Activation patterns and their count
│       ├── OneInput.lean        # The bound for one input
│       └── MultiInput.lean      # The bound for several inputs
└── Example/
    ├── Plot.lean                # Shared sampling and plotting helpers
    ├── WithBias.lean            # Numerical example and plots
    └── WithoutBias.lean         # Numerical example and plots
```

`neural-networks/ActivationFunction.lean` defines the type of a scalar activation
function, `Scalar → Scalar`, and ReLU: preserve nonnegative inputs and return zero
for negative inputs. Call `relu z` for zero bias, or `relu z bias` to add a bias
before rectifying; `relu z (bias := bias)` is the equivalent named form.
`relu_eq_max` proves that `relu z = max z 0` over any linearly ordered scalar
type; the proofs about ReLU networks use this form.

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
  scalar multiples. `MultiInput.ReLURepresentable` is the multi-input version,
  and `comp_linear` shows that a one-input network applied to `∑ i, w i * x i`
  is a multi-input network.
- `Interpolation.lean`: a ramp from `0` to `1` made of two ReLU units, so `N`
  grid cells give a network of width `2 * N`; the interpolant's value on one grid
  cell; and the error bound for a convex combination.
- `Ridge.lean`: a ridge function `φ (∑ i, w i * x i)` with continuous `φ` is
  approximable, by the one-input theorem applied to `φ`. The exponential ridges
  `exp (∑ i, w i * x i)` are closed under products, so the algebra they
  generate is their linear span, and it separates points.

`MultiInput.lean` states and proves the theorem for `Dᵢ` inputs
(`MultiInput.UniversalApproximation`, `MultiInput.universalApproximation`). With
two inputs the target's graph is a surface. The one-input grid argument does not
carry over, because every hidden unit is constant along hyperplanes and cannot
form a localized bump. Instead, extend the target (Tietze), approximate it by a
linear combination of exponential ridges (Stone–Weierstrass), and approximate
each ridge by the one-input theorem.

`neural-networks/ShallowNetwork/UniversalApproximation.lean` imports both theorems.

`neural-networks/ShallowNetwork/LinearRegions/` bounds the number of affine pieces.
`OneInput.lean` defines the proposition `LinearRegionBound`: for every width `D`
and parameters, some set `S` of at most `D` joints exists such that the network
is affine on every closed interval whose interior contains no joint. The points
of `S` split the real line into at most `D + 1` linear regions.
`linearRegionBound` proves it: the joints are `-θ_d0 / θ_d1` for units with
nonzero input weight, each unit is affine on an interval with no joint inside
(its preactivation keeps one sign there), and a weighted sum of affine functions
is affine.

`MultiInput.lean` proves the bound for `Dᵢ` inputs
(`MultiInput.LinearRegionBound`, `MultiInput.linearRegionBound`): the inputs fall
into at most `∑ j ≤ Dᵢ, (D choose j)` activation patterns, the sets of hidden
units with positive preactivation, and the network is affine on each. For
`Dᵢ = 1` this is `D + 1`; for `Dᵢ = 2` it is `1 + D + D * (D - 1) / 2`. Rather
than counting cells of the hyperplane arrangement, the proof shows that no
`Dᵢ + 1` units realize all their on/off combinations, because their weight
vectors in `ℝ^Dᵢ` are linearly dependent, and then applies the Sauer–Shelah
lemma. `ActivationPattern.lean` holds the definitions and both supporting facts.

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

## Chapter 4: Deep Neural Networks

```text
neural-networks/
├── ComposedNetwork.lean         # Entry point for Section 4.1
├── ComposedNetwork/
│   ├── Model.lean               # Shallow networks composed in order
│   ├── WidthThree.lean          # The book's two width-three networks
│   └── Verification.lean        # Definition checks
├── DeepNetwork.lean             # Entry point for Sections 4.3–4.4
└── DeepNetwork/
    ├── Model.lean               # Deep networks in matrix notation
    ├── WidthThree.lean          # Two layers of three units, named parameters
    └── Verification.lean        # Definition checks
```

Composed networks (Section 4.1) and deep networks (Sections 4.3–4.4) are separate
concepts with separate folders. Section 4.2, which shows that composed networks
are a special case of deep networks, will connect them.

`neural-networks/ComposedNetwork/Model.lean` follows Section 4.1: the output of
one shallow network becomes the input of the next. `composedNetwork activation Ns x`
takes a list `Ns` of shallow networks, each paired with its hidden width, and
applies them in order; the empty list returns `x`. The networks may have
different widths, and the list length is the number of composed networks.
`ComposedNetwork/WidthThree.lean` gives the book's example, equations
(4.1)–(4.4): `WidthThree.composedNetwork activation x Φ Φ'` feeds the output `y`
of a width-three network with parameters `Φ` into a second width-three network
with parameters `Φ'`, returning `y'`. Both reuse `WidthThree.Parameters` from
Chapter 3. `ComposedNetwork/Verification.lean` checks that this example is the
general composition applied to the list of the two networks
(`WidthThree.composedNetwork_eq`).

`neural-networks/DeepNetwork/Model.lean` defines deep networks in the general
form of Section 4.4: `h₁ = a[β₀ + Ω₀ x]`, `h_k = a[β_{k-1} + Ω_{k-1} h_{k-1}]`,
and `y = β_K + Ω_K h_K`, with `Dᵢ` inputs and `Dₒ` outputs. `LayerParameters`
holds one layer's bias vector `β` and weight matrix `Ω`.
`DeepNetworkParameters Scalar Dᵢ widths Dₒ` holds one layer per hidden width plus
the output layer; the hidden widths are part of the type, so consecutive layers
always have matching sizes. `deepNetwork activation Φ x` evaluates equation
(4.16), applying the layers `h ↦ a[β + Ω h]` in turn and then the output layer.
For example, `DeepNetworkParameters ℝ 1 [3, 3] 1` is the two-layer network of
Section 4.3. The hyperparameters of Section 4.3.1 are read off the type:
`Φ.depth` is the number of hidden layers `K`, `Φ.width k` is the number of
hidden units `D_k` in layer `k` (numbered from `0`), and `Φ.capacity` is the total
number of hidden units.

`neural-networks/DeepNetwork/WidthThree.lean` gives that network with the book's
named parameters, equations (4.7)–(4.9): six `θ` for the first layer, twelve `ψ`
for the second, and four `φ'` for the output. `WidthThree.DeepParameters` stores
them, `toDeepNetworkParameters` arranges them into the bias vectors and weight
matrices of equations (4.11)–(4.13), and `WidthThree.deepNetwork` evaluates the
network at a scalar input.

`DeepNetwork/Verification.lean` checks that a deep network with one hidden layer
is a shallow network: `MultiInput.Parameters.toDeep` converts a shallow network
with `Dᵢ` inputs and width `D` into a deep network with hidden widths `[D]`, and
`deepNetwork_toDeep` shows that both compute the same output; the converted
network has depth one and capacity `D` (`toDeep_capacity`). The capacity is the
sum of the layer widths (`capacity_eq_sum_width`). It also checks that
the width-three network expands to the single expression of equation (4.10)
(`WidthThree.deepNetwork_eq`).
