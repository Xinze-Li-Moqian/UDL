module

public import ShallowNetwork.Model
public import ShallowNetwork.MultiInput
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.Order.Field.Basic
public import Mathlib.Basic.Real.Basic
public import Mathlib.Tactic.Ring

/-!
# Functions computed by shallow ReLU networks

`ReLURepresentable f` says that some shallow ReLU network computes `f` exactly
at every input. The class of such functions contains every constant and every
shifted ReLU `relu (x - t)`, and is closed under sums, scalar multiples, and
finite sums. These closure properties let the universal approximation proof
assemble its network from simple pieces instead of writing parameters by hand.

`MultiInput.ReLURepresentable` is the same notion for networks with `Dᵢ` inputs.
Besides the closure properties, `MultiInput.ReLURepresentable.comp_linear` shows
that a one-input network applied to `∑ i, w i * x i` is a multi-input network.
-/

@[expose] public section

namespace NeuralNetworks

/-- `f` is computed exactly, at every input, by some shallow ReLU network. -/
def ReLURepresentable (f : ℝ → ℝ) : Prop :=
  ∃ (D : ℕ) (Φ : ShallowNetworkParameters ℝ D), ∀ x, shallowNetwork relu x Φ = f x

namespace ReLURepresentable

/-- A constant is the output bias of a zero-width network. -/
theorem const (c : ℝ) : ReLURepresentable fun _ => c :=
  ⟨0, ⟨c, Fin.elim0, Fin.elim0, Fin.elim0⟩, fun x => by
    simp [shallowNetwork_eq, addBias]⟩

/-- `relu (x - t)` is a single hidden unit with bias `-t` and weight `1`. -/
theorem relu_sub (t : ℝ) : ReLURepresentable fun x => relu (x - t) :=
  ⟨1, ⟨0, fun _ => 1, fun _ => -t, fun _ => 1⟩, fun x => by
    simp [shallowNetwork_eq, hiddenUnit, addBias, sub_eq_neg_add]⟩

/-- Sums of networks: place the hidden units side by side and add the output biases. -/
theorem add {f g : ℝ → ℝ} (hf : ReLURepresentable f) (hg : ReLURepresentable g) :
    ReLURepresentable fun x => f x + g x := by
  obtain ⟨D₁, Φ₁, h₁⟩ := hf
  obtain ⟨D₂, Φ₂, h₂⟩ := hg
  refine ⟨D₁ + D₂, ⟨Φ₁.outputBias + Φ₂.outputBias,
    Fin.append Φ₁.outputWeights Φ₂.outputWeights,
    Fin.append Φ₁.hiddenBiases Φ₂.hiddenBiases,
    Fin.append Φ₁.hiddenWeights Φ₂.hiddenWeights⟩, fun x => ?_⟩
  show _ = f x + g x
  rw [← h₁ x, ← h₂ x]
  simp only [shallowNetwork_eq, addBias, Fin.sum_univ_add, Fin.append_left, Fin.append_right]
  ring

/-- Scaling a network scales its output bias and output weights. -/
theorem mul_left {f : ℝ → ℝ} (c : ℝ) (hf : ReLURepresentable f) :
    ReLURepresentable fun x => c * f x := by
  obtain ⟨D, Φ, h⟩ := hf
  refine ⟨D, ⟨c * Φ.outputBias, fun d => c * Φ.outputWeights d,
    Φ.hiddenBiases, Φ.hiddenWeights⟩, fun x => ?_⟩
  show _ = c * f x
  rw [← h x]
  simp only [shallowNetwork_eq, addBias, mul_add, Finset.mul_sum, mul_assoc]

/-- Finite sums of representable functions are representable. -/
theorem sum {ι : Type*} (s : Finset ι) {f : ι → ℝ → ℝ}
    (hf : ∀ i ∈ s, ReLURepresentable (f i)) :
    ReLURepresentable fun x => ∑ i ∈ s, f i x := by
  classical
  induction s using Finset.induction_on with
  | empty => simpa using const 0
  | insert i s hi ih =>
    simp only [Finset.sum_insert hi]
    exact add (hf i (Finset.mem_insert_self i s))
      (ih fun j hj => hf j (Finset.mem_insert_of_mem hj))

end ReLURepresentable

namespace MultiInput

/-- `f` is computed exactly, at every input, by some shallow ReLU network with `Dᵢ` inputs. -/
def ReLURepresentable {Dᵢ : ℕ} (f : (Fin Dᵢ → ℝ) → ℝ) : Prop :=
  ∃ (D : ℕ) (Φ : Parameters ℝ Dᵢ D), ∀ x, network relu x Φ = f x

namespace ReLURepresentable

variable {Dᵢ : ℕ}

/-- A constant is the output bias of a zero-width network. -/
theorem const (c : ℝ) : ReLURepresentable (Dᵢ := Dᵢ) fun _ => c :=
  ⟨0, ⟨c, Fin.elim0, Fin.elim0, Fin.elim0⟩, fun x => by
    simp [network, addBias]⟩

/-- Sums of networks: place the hidden units side by side. -/
theorem add {f g : (Fin Dᵢ → ℝ) → ℝ} (hf : ReLURepresentable f) (hg : ReLURepresentable g) :
    ReLURepresentable fun x => f x + g x := by
  obtain ⟨D₁, Φ₁, h₁⟩ := hf
  obtain ⟨D₂, Φ₂, h₂⟩ := hg
  refine ⟨D₁ + D₂, ⟨Φ₁.outputBias + Φ₂.outputBias,
    Fin.append Φ₁.outputWeights Φ₂.outputWeights,
    Fin.append Φ₁.hiddenBiases Φ₂.hiddenBiases,
    Fin.append Φ₁.hiddenWeights Φ₂.hiddenWeights⟩, fun x => ?_⟩
  show _ = f x + g x
  rw [← h₁ x, ← h₂ x]
  simp only [network, addBias, Fin.sum_univ_add, Fin.append_left, Fin.append_right]
  ring

/-- Scaling a network scales its output bias and output weights. -/
theorem mul_left {f : (Fin Dᵢ → ℝ) → ℝ} (c : ℝ) (hf : ReLURepresentable f) :
    ReLURepresentable fun x => c * f x := by
  obtain ⟨D, Φ, h⟩ := hf
  refine ⟨D, ⟨c * Φ.outputBias, fun d => c * Φ.outputWeights d,
    Φ.hiddenBiases, Φ.hiddenWeights⟩, fun x => ?_⟩
  show _ = c * f x
  rw [← h x]
  simp only [network, addBias, mul_add, Finset.mul_sum, mul_assoc]

/-- A one-input network applied to `∑ i, w i * x i` is a network with `Dᵢ` inputs:
each hidden weight `θ_d` becomes the weight vector `θ_d • w`. -/
theorem comp_linear {D : ℕ} (Φ : ShallowNetworkParameters ℝ D) (w : Fin Dᵢ → ℝ) :
    ReLURepresentable fun x => shallowNetwork relu (∑ i, w i * x i) Φ :=
  ⟨D, ⟨Φ.outputBias, Φ.outputWeights, Φ.hiddenBiases,
    fun d i => Φ.hiddenWeights d * w i⟩, fun x => by
    simp only [network, shallowNetwork_eq, hiddenUnit, NeuralNetworks.hiddenUnit,
      Finset.mul_sum, mul_assoc]⟩

end ReLURepresentable

end MultiInput

end NeuralNetworks
