# Fitting models

Source: Understanding Deep Learning, Chapter 6.

Chapter 5 constructs a loss L[φ]. This chapter studies how to update model
parameters φ to reduce that loss, starting with Section 6.1, gradient descent.

Import `FittingModels` for the chapter interface. Shared optimization operations
live in `optimization/`; training objectives come from `loss-functions/`.

| File | Contents |
| --- | --- |
| `FittingModels/Gradient.lean` | Mathlib gradients, partial derivatives, the downhill direction, and stationarity at a local minimum |
| `FittingModels/GradientDescent.lean` | Loss-based updates and iteration, compatibility with a verified gradient formula, and fixed points |

## Section 6.1: Gradient descent

For a real-valued loss on a real Hilbert space, Mathlib supplies `gradient` and
`HasGradientAt`. With `open scoped Gradient`, write `∇ L φ`.
`hasDerivAt_coordinate` expresses equation (6.2) on Euclidean parameter vectors:
varying only coordinate j gives the j-th gradient component.

The update and iteration reuse `Optimization.gradientDescentStep` and
`Optimization.gradientDescent`:

```text
φₖ₊₁ = φₖ − α ∇L(φₖ),    α > 0.
```

`FittingModels.gradientDescentStep L α φ` performs this mathematical update;
`FittingModels.gradientDescent L α initial k` gives its k-th iterate.
These definitions use the mathematical gradient and are noncomputable.
`gradientDescent_eq_of_hasGradientAt` proves that supplying a verified gradient
formula to the existing loop gives the same sequence of real iterates. It
does not yet establish a floating-point implementation or rounding-error bound.

The first-order downhill identity is

```text
d/dt L(φ − t g) at t = 0 equals −‖g‖²,    g = ∇L(φ).
```

For nonzero g the derivative is negative, but no loss-decrease guarantee for
an arbitrary positive step size is claimed. At a differentiable unconstrained
local minimum, the gradient is zero and the update leaves the parameters fixed.
Zero gradient alone does not imply a minimum. Nondifferentiable points are not
covered by these derivative interpretations; Mathlib totalizes their gradient
to zero. Line search, stopping criteria, and convergence results remain later
steps.

Check this layer with `lake build FittingModels`.

## A first executable example

[`Examples/Quadratic.lean`](FittingModels/Examples/Quadratic.lean) minimizes
`L[φ] = (φ − 3)²`, starting at zero with learning rate `1/4`. It supplies the
formula `g(φ) = 2(φ − 3)` and proves that this is the real gradient, so the
existing executable update loop agrees with mathematical gradient descent
over the reals. A separate proof shows that each update halves the parameter
error and divides the loss by four.

The executable trace uses exact rational arithmetic:

| Step | Parameter | Gradient | Loss |
| --- | --- | --- | --- |
| 0 | 0 | −6 | 9 |
| 1 | 1.5 | −3 | 2.25 |
| 2 | 2.25 | −1.5 | 0.5625 |
| 3 | 2.625 | −0.75 | 0.140625 |
| 4 | 2.8125 | −0.375 | 0.03515625 |

Run `lake build FittingModels` to compile the example and display its rational
trace. This is an introductory one-parameter quadratic, preceding the book's
two-parameter linear regression example in Section 6.1.1.

## Section 6.1.1: linear regression on actual samples

[`Examples/LinearRegression.lean`](FittingModels/Examples/LinearRegression.lean)
shows the complete data-to-prediction workflow. It reuses the ten explicitly
listed synthetic observations in
[`ToyRegression.lean`](../loss-functions/LossFunctions/Examples/Data/ToyRegression.lean),
the Chapter 2 linear model, its squared-error loss, its verified gradient
formulas, and its training loop.

```text
data (xᵢ, yᵢ)
    → f(x, φ) = φ₀ + φ₁x
    → L[φ] = ∑ᵢ (f(xᵢ, φ) − yᵢ)²
    → gradient updates
    → fitted parameters
    → fittedModel(newInput)
```

The run starts from `(0, 0)`, uses learning rate `0.025`, and performs 100
updates. `trainingLoss`, `parametersAfter`, `fittedParameters`, and `fittedModel`
name the stages explicitly. Running `lake build FittingModels` displays the
data, parameter/gradient/loss checkpoints, and a new-input prediction.

| Updates | Intercept | Slope | Loss |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 28.07 |
| 1 | 0.5 | 0.445 | 13.520310 |
| 10 | 0.999023 | 1.854117 | 0.192826 |
| 100 | 1.000000 | 2.022727 | 0.067727 |

The fitted model predicts approximately `2.517045` at `x = 0.75`. The printed
values are rounded for display; updates use exact rational arithmetic. This is
a finite training demonstration, not a proof that 100 steps attain the exact
minimum. The gradient formulas' real-valued derivative proofs remain in
`supervised-learning/Verification/GradientFormula.lean`.

[`Examples/LinearRegressionVerification.lean`](FittingModels/Examples/LinearRegressionVerification.lean)
verifies the result against the same observations:

- The real parameters `(1, 89/44)` satisfy the normal equations and globally
  minimize the training loss.
- Their exact loss is `149/2200`.
- The actual 100-step rational output has intercept and slope errors below
  `10⁻⁹`, and excess loss between zero and `10⁻¹⁸`.

The finite checks use `decide +kernel` on the unrounded computed values.
They are Lean proofs checked by the kernel, rather than comparisons of the
printed decimals. The global optimality proof reuses the existing general
least-squares theorem; the finite iterate is certified as an approximation.
These statements concern this training dataset, not accuracy on unseen data.

## FloatLib integration

The Lake configuration pins FloatLib to commit
`1f5771b3e36a98f17fcac2c6d76aaa024b36aa37`. It uses the same Lean 4.35.0
and Mathlib 4.35.0 versions as this project. The lock file retains the existing
Mathlib revision.

[`Examples/FloatArithmetic.lean`](FittingModels/Examples/FloatArithmetic.lean)
is an optional first integration example. It selects binary32, computes two
sums, decodes their stored values to exact rationals, and proves that addition
agrees with FloatLib's specification. It imports just the binary instances
and arithmetic proof interface rather than all format families.

```sh
lake build FittingModels.Examples.FloatArithmetic
```

The decoded sums are `some (15/4)` for `1.5 + 2.25` and
`some (5033165/16777216)` for binary32 `0.1 + 0.2`. The latter is the stored
floating-point value, rather than the exact rational `3/10`.

The arithmetic example can be built separately. The chapter's default build
also includes the floating-point linear regression example below; its first
build compiles the required FloatLib dependencies.

## Linear regression executed in binary32

[`Examples/FloatLinearRegression.lean`](FittingModels/Examples/FloatLinearRegression.lean)
uses the same ten original observations and existing model, gradient formula,
and update loop, instantiated with `FittingModels.Binary32`. The conversion
helpers live in [`FloatingPoint.lean`](FittingModels/FloatingPoint.lean).

Each rational observation and the learning rate `1/40` are rounded once to
binary32 using nearest-even rounding. The gradient, parameter updates, loss
evaluation, and new-input prediction then use FloatLib's binary32 arithmetic.
Only the readable decimal display passes through Lean's native `Float`.

Since floating-point addition is not associative, loss evaluation uses
`orderedEmpiricalLoss`, with the explicit right-associated order
`ℓ₀ + (ℓ₁ + … + (ℓ_last + 0))`. The existing gradient formula uses that same
list-sum convention. `orderedEmpiricalLoss_eq_empiricalLoss` proves agreement
with the original mathematical sum when the scalar has an `AddCommMonoid`;
no such algebraic laws are asserted for binary32.

```sh
lake build FittingModels.Examples.FloatLinearRegression
```

After 100 updates the stored parameters decode exactly to
`(16777215/16777216, 8483931/4194304)`. Displayed to six decimal places they
are `(1.000000, 2.022727)`, with floating-point training loss approximately
`0.067727` and prediction approximately `2.517045` at `x = 0.75`.
The displayed intercept rounds to one but its stored value is slightly smaller.

The previous rational-run tolerances do not certify this different execution.
[`Examples/FloatLinearRegressionVerification.lean`](FittingModels/Examples/FloatLinearRegressionVerification.lean)
supplies a separate certificate for selected binary32 checkpoints. It decodes
the actual output of `parametersAfter k`, checks both parameters are finite,
and bounds:

- Total parameter error relative to `(1, 89/44)`.
- Parameter error relative to the exact rational run after the same k steps.
  This comparison includes input and learning-rate conversion as well as
  rounding during arithmetic.
- Excess loss relative to `149/2200`, evaluated exactly on the original
  observations using the decoded parameters.

| Updates | Error bound for each parameter relative to the optimum | Excess-loss bound |
| --- | --- | --- |
| 0 | 3 | 29 |
| 1 | 1.6 | 14 |
| 2 | 1.25 | 8 |
| 5 | 0.6 | 1.6 |
| 10 | 0.2 | 0.13 |
| 25 | 0.005 | 0.0001 |
| 100 | 10⁻⁶ | 10⁻¹¹ |

At every listed checkpoint, each parameter differs from its exact-run
counterpart by at most `10⁻⁶`. All inequalities are non-strict bounds.
The certificate uses `decide +kernel` on the program itself, without replacing
its output by hand-entered decimal or bit-pattern constants. It does not use
`native_decide`.

Build it with `lake build FittingModels.Examples.FloatLinearRegressionVerification`,
or as part of `lake build FittingModels`. These are proofs for the listed
finite checkpoints, not a general error formula or convergence theorem for
arbitrary k. They do not claim that more floating-point updates always improve
the result.
