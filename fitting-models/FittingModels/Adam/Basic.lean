module

public import Adam
public import FittingModels.StochasticGradientDescent.Basic
public import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Shared Adam update

The two batch methods supply different inputs q to the second-moment update.
Both use the same batch gradient g and the same update below, coordinatewise:

    m_next = β m + (1 − β) g,
    v_next = γ v + (1 − γ) q,
    m_hat  = m_next / (1 − β^(t+1)),
    v_hat  = v_next / (1 − γ^(t+1)),
    φ_next = φ − α m_hat / (sqrt(v_hat) + ε).

The denominator uses ε outside the square root, as in Algorithm 1 of
Kingma and Ba, Adam: https://arxiv.org/abs/1412.6980.

Use α > 0, 0 ≤ β, γ < 1, ε > 0, and nonnegative second-moment inputs.
These arithmetic definitions do not assert convergence. Parameters are
finite Euclidean vectors, so squaring and division act on each coordinate.
-/

@[expose] public section

namespace FittingModels.Adam

open Optimization (AdamState)

variable {N : ℕ}

/-- Square each coordinate, rather than taking a squared norm. -/
noncomputable def square (g : EuclideanSpace ℝ (Fin N)) : EuclideanSpace ℝ (Fin N) :=
  WithLp.toLp 2 (fun j => (g j) ^ 2)

/-- Shared one-step update. The caller explicitly supplies q, making the
second-moment convention visible at each call site. Store uncorrected m and v;
use their bias-corrected values only for updating the parameters. -/
noncomputable def stepFromStatistics (g q : EuclideanSpace ℝ (Fin N))
    (α β γ ε : ℝ) (state : AdamState (EuclideanSpace ℝ (Fin N))) :
    AdamState (EuclideanSpace ℝ (Fin N)) :=
  let t_next := state.t + 1
  let m_next := β • state.m + (1 - β) • g
  let v_next := γ • state.v + (1 - γ) • q
  let m_hat := (1 - β ^ t_next)⁻¹ • m_next
  let v_hat := (1 - γ ^ t_next)⁻¹ • v_next
  let φ_next := WithLp.toLp 2 (fun j =>
    state.φ j - α * m_hat j / (Real.sqrt (v_hat j) + ε))
  ⟨φ_next, m_next, v_next, t_next⟩

end FittingModels.Adam
