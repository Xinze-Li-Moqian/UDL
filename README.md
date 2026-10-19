# UDL — Learning Deep Learning in Lean

I am learning deep learning by building models directly in Lean, following
[*Understanding Deep Learning*](https://udlbook.github.io/udlbook/) by
Simon J. D. Prince.

This repository grows as I work through the book: turn a formula into code,
try it on data, inspect the results, and use Lean to prove properties of the
same models and algorithms. The emphasis is on learning through implementation
and experiment.

## Acknowledgements

Thank you to **Sabatino Costanzo** and **Loren Trigo**, instructors of
[APS1053H: Case Studies in AI in Finance](https://gradstudies.engineering.utoronto.ca/master-of-engineering-meng/elite-emphasis/elite-finance-management-courses/)
at the University of Toronto, for their teaching and emphasis on learning
through practical case studies.

## What is here

The current work covers supervised learning, shallow and deep networks,
probabilistic predictions, loss functions, and the beginning of model fitting
in Chapter 6.

| Directory | Topic |
| --- | --- |
| [supervised-learning/](supervised-learning/) | Linear regression, training, and worked examples |
| [neural-networks/](neural-networks/) | Network architectures, parameters, and activations |
| [probability/](probability/README.md) | Distributions, expectations, densities, and conditioning |
| [information-theory/](information-theory/README.md) | KL divergence and related foundations |
| [loss-functions/](loss-functions/README.md) | Constructing training losses from predictive distributions |
| [optimization/](optimization/) | Shared optimization definitions and update rules |
| [fitting-models/](fitting-models/README.md) | Gradient descent, executable training, and result verification |

Definitions and proofs use Mathlib. Executable examples use exact rational
arithmetic or FloatLib floating-point arithmetic. Detailed explanations and
book references live with the relevant code and in the directory guides.

## Run the examples

The project uses Lean and Mathlib 4.35.0, with a pinned FloatLib dependency.
From the repository root:

```sh
lake exe cache get
lake build
```

To follow the current linear regression example:

```sh
# Train on ten toy observations using FloatLib binary32 arithmetic.
lake build FittingModels.Examples.FloatLinearRegression

# Check parameter and loss error bounds at selected training steps.
lake build FittingModels.Examples.FloatLinearRegressionVerification
```

The example shows the path from data to a loss function, parameter updates,
and predictions. Its verification checks the actual floating-point outputs
against the exact least-squares solution and an exact-arithmetic training run.
See [the model-fitting guide](fitting-models/README.md) for the assumptions,
proved bounds, and remaining work.

## Reference

The book and its [companion materials](https://github.com/udlbook/udlbook) are
the main learning resources. Book material is available under its
[CC BY-NC-ND license](https://github.com/udlbook/udlbook/blob/main/LICENSE).
