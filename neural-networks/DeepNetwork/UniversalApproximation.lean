module

public import Network
public import ShallowNetwork.UniversalApproximation
public import Mathlib.LinearAlgebra.Matrix.Notation

/-!
# Universal approximation by deep networks

Source: Understanding Deep Learning, Section 4.5.1.

A deep network with two hidden layers can represent the composition of two
shallow networks. If the second network computes the identity, the deep network
replicates a single shallow network, so deep networks inherit universal
approximation from shallow ones.

For ReLU the identity is exact: `y = ReLU(y) - ReLU(-y)`. `deepen Φ K` turns a
shallow network into a deep network with hidden widths `D :: [2, …, 2]` (`K` twos):

1. the first layer is the shallow network's hidden layer;
2. the next layer absorbs the shallow output `y` and computes
   `(ReLU(y), ReLU(-y))`;
3. each further layer maps `(u, v)` to `(ReLU(u - v), ReLU(v - u))`, which keeps
   `u - v = y`;
4. the output layer returns `u - v`.

`network_deepen` shows that the deep network computes the shallow network's
output. `deepUniversalApproximation` then applies the multi-input universal
approximation theorem: for every depth `K ≥ 1`, deep ReLU networks with some
hidden widths approximate any continuous function on a compact set. The compact
set is called `C` here because the book uses `K` for the depth.
-/

@[expose] public section

namespace NeuralNetworks

open Matrix

/-- ReLU recovers the identity: `ReLU(t) - ReLU(-t) = t`. -/
theorem relu_sub_relu_neg (t : ℝ) : relu t - relu (-t) = t := by
  rw [relu_eq_max, relu_eq_max]
  rcases le_total 0 t with h | h
  · rw [max_eq_left h, max_eq_right (neg_nonpos.mpr h), sub_zero]
  · rw [max_eq_right h, max_eq_left (neg_nonneg.mpr h), zero_sub, neg_neg]

namespace Deepen

/-- `K` layers that keep a pair `(u, v)` with `u - v = y`, then output `u - v`. -/
def pairLayers : (K : ℕ) → NetworkParameters ℝ (Architecture.ofList 2 (List.replicate K 2) 1)
  | 0 => .single 0 !![1, -1]
  | K + 1 => .cons 0 !![1, -1; -1, 1] (pairLayers K)

/-- The output `y = b + ∑ d, w d * h d` of a hidden layer `h`, carried through
`K` further layers as the pair `(ReLU(y), ReLU(-y))`. -/
def carry {D : ℕ} (b : ℝ) (w : Fin D → ℝ) :
    (K : ℕ) → NetworkParameters ℝ (Architecture.ofList D (List.replicate K 2) 1)
  | 0 => .single (fun _ => b) (Matrix.of fun _ d => w d)
  | K + 1 => .cons ![b, -b] (Matrix.of ![w, -w]) (pairLayers K)

/-- The pair layers output `u - v` for any input pair `(u, v)`. -/
theorem network_pairLayers (K : ℕ) (p : Fin 2 → ℝ)
    (j : Fin (Architecture.ofList 2 (List.replicate K 2) 1).outputs) :
    network relu (pairLayers K) p j = p 0 - p 1 := by
  induction K generalizing p with
  | zero =>
    revert j
    show ∀ j : Fin 1, (0 : Fin 1 → ℝ) j + ∑ i : Fin 2, !![(1 : ℝ), -1] j i * p i = p 0 - p 1
    intro j
    fin_cases j
    simp [Fin.sum_univ_two, sub_eq_add_neg]
  | succ K ih =>
    rw [pairLayers, network_cons]
    refine (ih _ j).trans ?_
    simp only [Function.comp_apply, zero_add, mulVec, dotProduct, Fin.sum_univ_two, of_apply, cons_val', cons_val_zero, cons_val_one, empty_val',
      cons_val_fin_one, one_mul, neg_mul]
    rw [show -p 0 + p 1 = -(p 0 + -p 1) by ring, ← sub_eq_add_neg]
    exact relu_sub_relu_neg _

/-- `carry b w K` computes `b + ∑ d, w d * h d` from the hidden layer `h`. -/
theorem network_carry {D : ℕ} (b : ℝ) (w : Fin D → ℝ) (K : ℕ) (h : Fin D → ℝ)
    (j : Fin (Architecture.ofList D (List.replicate K 2) 1).outputs) :
    network relu (carry b w K) h j = b + ∑ d, w d * h d := by
  cases K with
  | zero => rfl
  | succ K =>
    rw [carry, network_cons]
    refine (network_pairLayers K _ j).trans ?_
    simp only [Function.comp_apply, Pi.add_apply, mulVec, dotProduct, of_apply, cons_val_zero,
      cons_val_one, Pi.neg_apply, neg_mul, Finset.sum_neg_distrib]
    rw [show -b + -∑ d, w d * h d = -(b + ∑ d, w d * h d) by ring]
    exact relu_sub_relu_neg _

end Deepen

/-- A shallow network with `Dᵢ` inputs and width `D`, deepened by `K` identity layers
of width two. -/
def MultiInput.Parameters.deepen {Dᵢ D : ℕ} (Φ : MultiInput.Parameters ℝ Dᵢ D) (K : ℕ) :
    NetworkParameters ℝ (Architecture.ofList Dᵢ (D :: List.replicate K 2) 1) :=
  .cons Φ.hiddenBiases (Matrix.of Φ.hiddenWeights) (Deepen.carry Φ.outputBias Φ.outputWeights K)

/-- The deepened network computes the shallow network's output. -/
theorem network_deepen {Dᵢ D : ℕ} (Φ : MultiInput.Parameters ℝ Dᵢ D) (K : ℕ)
    (x : Fin Dᵢ → ℝ) (j : Fin (Architecture.ofList Dᵢ (D :: List.replicate K 2) 1).outputs) :
    network relu (Φ.deepen K) x j = MultiInput.network relu x Φ := by
  rw [MultiInput.Parameters.deepen, network_cons]
  refine (Deepen.network_carry _ _ K _ j).trans ?_
  simp [MultiInput.network_eq, MultiInput.hiddenUnit, addBias, mulVec, dotProduct]

/-- Section 4.5.1: for every depth `K ≥ 1`, deep ReLU networks approximate every
continuous function on a compact set `C ⊆ ℝ^Dᵢ` uniformly. -/
def DeepUniversalApproximation : Prop :=
  ∀ (K : ℕ), 1 ≤ K → ∀ (Dᵢ : ℕ) (C : Set (Fin Dᵢ → ℝ)), IsCompact C →
    ∀ (g : (Fin Dᵢ → ℝ) → ℝ), ContinuousOn g C → ∀ ε > 0,
      ∃ widths : List ℕ, widths.length = K ∧
        ∃ f ∈ networkFamily relu (Architecture.ofList Dᵢ widths 1),
          ∀ x ∈ C, ∀ j, |f x j - g x| < ε

/-- **Universal approximation for deep networks**: a shallow approximation, deepened
by identity layers, has any prescribed depth. -/
theorem deepUniversalApproximation : DeepUniversalApproximation := by
  intro K hK Dᵢ C hC g hg ε hε
  obtain ⟨D, Φ, hΦ⟩ := MultiInput.universalApproximation Dᵢ C hC g hg ε hε
  refine ⟨D :: List.replicate (K - 1) 2, by simp; omega,
    network relu (Φ.deepen (K - 1)), ⟨_, rfl⟩, fun x hx j => ?_⟩
  rw [network_deepen]
  exact hΦ x hx

end NeuralNetworks
