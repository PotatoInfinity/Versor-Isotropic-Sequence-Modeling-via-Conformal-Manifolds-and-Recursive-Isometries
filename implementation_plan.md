# Implementation Plan: Geo-Llama Hybrid Model Proof of Concept

This document outlines the steps to implement the **Geo-Llama-3.2-1B Hybrid Model**, focusing on the integration of $Cl_{4,1}$ Conformal Geometric Algebra (CGA) with the Llama Transformer architecture.

## Phase 1: Geometric Core (G-Stream) Implementation
- [ ] **Step 1: Python CGA Runtime**
  - Implement `Multivector5D` class in Python using NumPy.
  - Precompute the `GP_MAP` (Result-Centric Cayley table) for $O(1)$ geometric products.
  - Implement `wedge`, `inner_product`, and `rotor` transformations.
- [ ] **Step 2: Lifting and Partitioning**
  - Implement a `LiftingLayer` to project 2048D Llama embeddings into 64 parallel $Cl_{4,1}$ manifolds.
  - Define the mapping of token features to geometric grades (Points, Bivectors, Quad-blades).
- [ ] **Step 3: Recursive Rotor Accumulator**
  - Implement the stateful $\Psi$ (Context Rotor) update: $\Psi_{t+1} = R_t \Psi_t \tilde{R}_t$.
  - Implement "Rotor Re-projection" for numerical stability (Gram-Schmidt).

## Phase 2: Hybrid Integration (Dual-Stream)
- [ ] **Step 4: Llama 3.2 1B Integration**
  - [x] Approval received for Llama 3.2 1B & 3B.
  - [x] Hugging Face Login Successful.
  - [ ] Download weights and run first hybrid pass (In Progress).
- [x] **Step 5: GCA (Geometry-Conditioned Attention)**
  - [x] Implement the GCA layer in PyTorch: $Q_i K_j^T + \lambda \langle \Psi, Q_i \wedge K_j \rangle$.
  - [x] Implement "Monkey-Patch" weight injection for Llama self-attention blocks.
- [x] **Step 6: Manifold Training & Loss**
  - [x] Implement `GeometricConsistencyLoss` (Grade-Sparsity and Hierarchy enforcement).
  - [x] Train the `LiftingLayer` to optimize for geometric consistency.
- [ ] **Step 7: Comparative Tests**
  - Compare logical consistency and context retention between standard Llama and Geo-Llama Hybrid.
  - Benchmark memory usage and inference speed (O(1) context test).

---

### Current Task: Step 7 - Comparative Tests / Project Review
We have successfully implemented the Manifold Training loop. The next step is to finalize the project review against the README.
