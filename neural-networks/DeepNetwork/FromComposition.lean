module

public import ComposedNetwork.Model

/-!
# Composed networks are deep networks

Source: Understanding Deep Learning, Section 4.2, equations (4.5)–(4.6).

The output of a shallow network, `y = φ₀ + ∑ d, φ_d * h_d`, is linear in its
hidden units, and the next network begins with the linear maps `θ'_d'0 + θ'_d'1 * y`.
Substituting one into the other gives a hidden layer acting directly on `h`:

```
ψ_d'0 = θ'_d'0 + θ'_d'1 * φ₀,     ψ_d'd = θ'_d'1 * φ_d.
```

`ComposedNetwork.toNetworkParameters` builds the deep network this way: the first
shallow network's hidden layer stays, each later hidden layer absorbs the previous
network's output layer, and the last output layer stays. Composing shallow
networks of widths `[D₁, …, D_K]` gives a deep network with those hidden widths
(`network_toNetworkParameters`), so the composed family lies inside the deep
family of the same architecture (`composedFamily_subset`).

The middle weight matrices are constrained to the outer products `θ' φᵀ`, while a
general deep network may use any matrix there. The book observes that deep
networks therefore describe a broader family; that strictness is not proved here.
With no networks, the composition is the identity, the affine map with `β = 0`
and `Ω = 1`.
-/

@[expose] public section

namespace NeuralNetworks.ComposedNetwork

open Matrix

variable {Scalar : Type} [Semiring Scalar]

/-- The layers that follow a hidden layer `h` of width `D` whose network output is
`y = b + ∑ d, w d * h d`: each later network's hidden layer absorbs that output, as in
equation (4.6), and the last network's output layer ends the deep network. -/
def tail {D : ℕ} (b : Scalar) (w : Fin D → Scalar) :
    (Ns : List (Σ D, ShallowNetworkParameters Scalar D)) →
      NetworkParameters Scalar (Architecture.ofList D (Ns.map Sigma.fst) 1)
  | [] => .single (fun _ => b) (Matrix.of fun _ d => w d)
  | ⟨_, Φ⟩ :: rest =>
      .cons (fun d' => Φ.hiddenBiases d' + Φ.hiddenWeights d' * b)
        (Matrix.of fun d' d => Φ.hiddenWeights d' * w d)
        (tail Φ.outputBias Φ.outputWeights rest)

/-- Equations (4.5)–(4.6): the deep network computing the composition of `Ns`. -/
def toNetworkParameters :
    (Ns : List (Σ D, ShallowNetworkParameters Scalar D)) →
      NetworkParameters Scalar (Architecture.ofList 1 (Ns.map Sigma.fst) 1)
  | [] => .single 0 (Matrix.of fun _ _ => 1)
  | ⟨_, Φ⟩ :: rest =>
      .cons Φ.hiddenBiases (Matrix.of fun d _ => Φ.hiddenWeights d)
        (tail Φ.outputBias Φ.outputWeights rest)

/-- The tail feeds the output `b + ∑ d, w d * h d` into the remaining networks.
The network has a single output; the statement holds at its only index `j`. -/
theorem network_tail (a : ActivationFunction Scalar) {D : ℕ}
    (Ns : List (Σ D, ShallowNetworkParameters Scalar D))
    (b : Scalar) (w h : Fin D → Scalar) (j : Fin (Architecture.ofList D (Ns.map Sigma.fst) 1).outputs) :
    network a (tail b w Ns) h j = composedNetwork a Ns (b + ∑ d, w d * h d) := by
  induction Ns generalizing D b w h with
  | nil => rfl
  | cons N rest ih =>
    obtain ⟨D', Φ⟩ := N
    rw [tail, network_cons]
    refine (ih _ _ _ j).trans ?_
    simp only [composedNetwork, shallowNetwork_eq, hiddenUnit, addBias, Function.comp_apply,
      Pi.add_apply, mulVec, dotProduct, of_apply, mul_add, Finset.mul_sum, mul_assoc, add_assoc]

/-- **Section 4.2**: the deep network computes the composition of the shallow networks,
at its only output index `j`. -/
theorem network_toNetworkParameters (a : ActivationFunction Scalar)
    (Ns : List (Σ D, ShallowNetworkParameters Scalar D)) (x : Scalar)
    (j : Fin (Architecture.ofList 1 (Ns.map Sigma.fst) 1).outputs) :
    network a (toNetworkParameters Ns) (fun _ : Fin 1 => x) j = composedNetwork a Ns x := by
  cases Ns with
  | nil =>
    show (0 : Scalar) + ∑ _ : Fin 1, (1 : Scalar) * x = x
    simp
  | cons N rest =>
    obtain ⟨D, Φ⟩ := N
    rw [toNetworkParameters, network_cons]
    refine (network_tail a rest _ _ _ j).trans ?_
    simp only [composedNetwork, shallowNetwork_eq, hiddenUnit, addBias, Function.comp_apply,
      Pi.add_apply, mulVec, dotProduct, of_apply, Fin.sum_univ_one]

/-- Composed networks of widths `[D₁, …, D_K]` lie in the family of deep networks with
those hidden widths, one input, and one output. -/
theorem composedFamily_subset (a : ActivationFunction Scalar) (widths : List ℕ) :
    ∀ f ∈ composedFamily a widths,
      ∃ g ∈ networkFamily a (Architecture.ofList 1 widths 1),
        ∀ x j, g (fun _ : Fin 1 => x) j = f x := by
  rintro f ⟨Ns, rfl, rfl⟩
  exact ⟨_, ⟨toNetworkParameters Ns, rfl⟩, network_toNetworkParameters a Ns⟩

end NeuralNetworks.ComposedNetwork
