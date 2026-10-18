# Heteroscedastic univariate regression

"Univariate" specifies the target dimension: `y : ℝ`. "Heteroscedastic"
describes variation of the conditional variance with the input. A univariate
model can have constant or input-dependent variance; the same distinction
also applies to multivariate targets, which are outside this directory's scope.

The shared interface remains in `../Model.lean`. Each distribution can supply
its own implementation here, reusing `Model.predictiveDistribution` and
`Model.loss`. `../Heteroscedastic.lean` is the import entry point.

## Gaussian implementation

`Gaussian.lean` provides `Heteroscedastic.gaussianModel μ v`:

    Y | x, φ ∼ 𝒩(μ(x, φ), v(x, φ)),    v(x, φ) > 0.

Both predictors use the same model parameters. The variance predictor may be
constant. Indeed, the existing fixed-variance and learned-shared-variance
constructors in `../Gaussian.lean` are now specializations of this constructor.

The family and density come from `ProbabilityModels.Gaussian`.
`gaussianLoss` reuses the existing negative log-likelihood, and
`gaussianLoss_eq` proves

    L[φ] = ∑ᵢ [log(2πvᵢ) / 2 + (yᵢ − μᵢ)² / (2vᵢ)],
    μᵢ = μ(xᵢ, φ),    vᵢ = v(xᵢ, φ).

`mean_isModePrediction` reuses the Gaussian mode theorem for inference.

For the book's two-output construction (5.14), `squaredOutputModel f hne`
sets the mean to `f x φ 0` and the variance to `(f x φ 1)²`.
The hypothesis `hne` requires the second output to be nonzero for every
admissible input and parameter. A zero output would give zero variance,
which is outside the positive-variance density model.
`squaredOutputModel_loss_eq` proves the resulting loss expansion for (5.15).

These definitions and proofs specify the model and objective. They do not
implement training or assume a minimizer exists. Other distribution modules
can be added as needed; only the Gaussian case is implemented here.
