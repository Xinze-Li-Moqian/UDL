module

public import LossFunctions.Examples.MulticlassClassification.Categorical
public import LossFunctions.Examples.MulticlassClassification.Softmax
public import LossFunctions.Examples.MulticlassClassification.CrossEntropy

/-!
# Example: multiclass classification

Begin with specified categorical probabilities, as in Figure 5.9.
Then use softmax to construct categorical parameters from a real score vector.
The cross-entropy module reuses the empirical loss and proves both lines of (5.24).
-/
