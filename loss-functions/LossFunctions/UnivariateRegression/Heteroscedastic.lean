module

public import LossFunctions.UnivariateRegression.Heteroscedastic.Gaussian

/-!
# Heteroscedastic univariate regression

The target remains a single real number. Distribution parameters describing
its spread may depend on the input. Implementations reuse the shared
`UnivariateRegression.Model` interface and its prediction and loss operations.
-/
