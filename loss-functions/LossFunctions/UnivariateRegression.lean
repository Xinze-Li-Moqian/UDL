module

public import LossFunctions.UnivariateRegression.Model
public import LossFunctions.UnivariateRegression.Heteroscedastic
public import LossFunctions.UnivariateRegression.Gaussian

/-!
# Univariate regression

`Model` provides the shared interface for a real-valued target. Distribution
implementations and their loss derivations live in `UnivariateRegression/`.
-/
