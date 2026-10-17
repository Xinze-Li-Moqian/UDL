# FinTech, Reinforcement Learning & Deep Learning

A personal learning and experimentation repository for financial technology (FinTech), reinforcement learning (RL), and deep learning (DL).

The aim is to build understanding through theory notes, algorithm implementations, paper reproductions, and experiments. Each area has room for its own foundations, with connections explored as questions and interests develop.

## Areas of Study

- **FinTech:** Financial concepts, data, and computational approaches to financial problems.
- **Reinforcement Learning:** Sequential decision-making, learning algorithms, and agent evaluation.
- **Deep Learning:** Neural networks, optimization, model architectures, and representation learning.
- **Connections:** Applications of DL and RL to financial problems, grounded in clear assumptions and careful evaluation.

## Working Approach

For each topic or experiment, aim to record:

1. The question being explored and the relevant background.
2. The assumptions, mathematical ideas, and method.
3. An implementation, with a simple baseline where useful.
4. The setup and instructions needed to reproduce the results.
5. What worked, what failed, and what remains unclear.

Keep the structure simple and let shared tooling grow out of recurring needs. Progress means being able to explain the ideas, reproduce the results, and revisit earlier work with a clearer understanding.

## Lean Verification

The project pins Lean and Mathlib to version 4.35.0. From the repository root, run
`lake exe cache get` to download Mathlib's compiled dependencies, then `lake build`
to check the definitions and proofs.

`supervised-learning/Verification.lean` groups proofs by topic. The real-valued
training proofs verify the two partial derivatives and a sufficient step-size
condition for nonincreasing least-squares loss; they do not establish convergence
or that a finite training run finds an exact minimizer.
