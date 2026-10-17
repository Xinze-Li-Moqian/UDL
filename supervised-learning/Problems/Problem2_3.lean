module

public import LinearRegression

/-!
# Problem 2.3* — Linear regression as a generative model

Source: Understanding Deep Learning, Chapter 2, Problem 2.3*.

Reformulate linear regression as a generative model:

`x = g[y, φ] = φ₀ + φ₁ * y`.

1. Write the new loss function for fitting this model.
2. Find the inverse `y = g⁻¹[x, φ]` used to perform inference.
3. Determine whether fitting this model gives the same predictions as the
   discriminative model, using the same training pairs `{(xᵢ, yᵢ)}`.

The book suggests exploring the comparison by fitting both models to three
data points and comparing their predictions of `y` from `x`.

Learning goal: understand how the direction of the model affects fitting
and inference. Check when the inverse model is defined.

Solution: to be developed.
-/
