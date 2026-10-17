module

public import Mathlib.Probability.Density

/-!
# Probability densities

Use `MeasureTheory.HasPDF X P μ` for a random variable whose law under `P`
has a density with respect to a reference measure `μ`.
`MeasureTheory.pdf X P μ` is the corresponding Radon-Nikodym derivative.
`MeasureTheory.Measure.withDensity` constructs a measure from a density.

For likelihoods, choose a common reference measure across the parameter
family and specify the density used at observed values. Counting measure
gives probability masses; Lebesgue measure gives continuous densities.
Connecting such a density to `ProbabilityModels.Family` is a further step.
-/
