# Geo-Llama Codebase Documentation

This document provides a detailed technical breakdown of every file and code block in the **Geo-Llama (Stable Release)** implementation. It serves as a companion to the theoretical "Blue Book" paper.

---

## **1. Root Directory** `Stable/`

### `chat.py`
The main entry point for the user interface.
*   **`chat_interface()`**: The infinite loop that drives the conversation.
    *   **Initialization**: Instantiates `GeoLlamaHybrid`. Sets the initial "Lambda" (geometric influence) to 0.001 to prevent untrained noise from breaking the language model.
    *   **Metrics Display**: Calculates and prints the fixed state size (`g_stream_size = 64 * 32 * 4` bytes), verifying O(1) memory.
    *   **Prompt Construction**: Formats user input into the Llama-3 instruction template.
    *   **Response Parsing**: logic to strip `start_header_id` and other specialized Llama tokens from the raw output.

### `start_demo.sh`
Bash script wrapper.
*   **Environment Detection**: Automatically detects if a `venv` exists in the current or parent directory to use the correct Python interpreter.
*   **Path Setup**: Exports `PYTHONPATH` to ensure the `geo_llama` module is importable.

### `test_mock.py`
Verification script.
*   **`test_mock()`**: Instantiates `GeoLlamaHybrid` with `is_mock=True`. This allows testing the Geometric Algebra logic (Right Brain) without loading the 2GB+ Llama weights (Left Brain), enabling fast architectural validation.

### `README.md`
The "Blue Book" theoretical paper, serving as the primary project description.

### `README_JUDGES.md`
A simplified guide for hackathon judges focusing on how to run the demo and what specific features (like O(1) context) to look for.

---

## **2. Core Logic** `Stable/geo_llama/`

### `hybrid.py`
The orchestrator that binds the Neural (Llama) and Symbolic (Geometric) systems.
*   **`class GeoLlamaHybrid`**:
    *   **`__init__`**:
        *   Loads the pre-trained `meta-llama/Llama-3.2-1B-Instruct`.
        *   Initializes the `SpecializedLiftingLayer` (Encoder) and `GeoLlamaState` (Memory).
        *   Initializes the `ManifoldMixingLayer` (Resolution to Binding Problem).
        *   **`head_lifter`**: A projection layer that maps Llama's attention heads to the 32-dim geometric space for the Attention Bias calculation.
    *   **`gca_bias(Q, K)`**: Calculates the **Geometry-Conditioned Attention**.
        *   `psi_torch`: The current state of the 64 geometric heads.
        *   `bias`: Computed as $\lambda \langle \Psi, Q \wedge K \rangle$.
    *   **`hook_attention()`**: The critical "Monkey Patch". It iterates through every layer of the frozen Llama model and replaces the standard `forward()` method with `create_custom_forward()`, which injects the `gca_bias` into the attention mask.
    *   **`generate_with_geometry()`**:
        *   Runs a pre-generation forward pass to update the geometric state `PSI`.
        *   Calls `self.state.update()` recursively to "digest" the prompt into the geometric manifold.

---

## **3. Geometric Engine** `Stable/geo_llama/core/`

### `cga.py`
The Mathematical Kernel. Implements $Cl_{4,1}$ Conformal Geometric Algebra.
*   **`GP_MAP`**: A precomputed Cayley Table logic that maps basis vector indices (0-31) to their Geometric Product result. This replaces sparse matrix multiplication with a pre-calculated lookup, key to performance.
*   **`batch_geometric_product(A, B)`**: The core algebraic operation. Supports both `numpy` and `torch`. Computes $A B$ utilizing the `GP_MAP`.
*   **`batch_wedge_product(A, B)`**: Computes the Outer Product $A \wedge B$. Used to form "Planes of Thought" (Bivectors) from token points.
*   **`inverse(R)`**: Computes $R^{-1}$ for rotors.
*   **`normalize_rotor(R)`**: Ensures numerical stability by forcing the rotor to unit norm (Manifold Projection).
*   **`batch_point(x, y, z)`**: Converts 3D coordinates into 5D Conformal Points $P = n_o + x + \frac{1}{2}x^2 n_\infty$.

### `layers.py`
High-level Neural Network layers built on top of `cga.py`.
*   **`class SpecializedLiftingLayer`**: 
    *   Implementing **Subspace Partitioning** (Section 2.1).
    *   Splits the embedding dimension into 3 slices: Syntactic (Heads 0-9), Semantic (Heads 10-39), Narrative (Heads 40-63).
    *   Applies distinct initialization scales (0.1, 0.5, 1.5) to each group.
*   **`class GeoLlamaState`**:
    *   **`self.psi`**: The **Tensor Bundle** of 64 rotors. This is the "Infinite Memory".
    *   **`update(rotors, mixing_layer)`**:
        *   Applies the **Recursive Isometry**: $\Psi_{new} = R \Psi_{old} R^{-1}$.
        *   **Manifold Mixing**: Calls the `mixing_layer` to allow information exchange between heads (solving Binding Problem).
        *   **Normalization**: Prevents "Drift".
*   **`class ManifoldMixingLayer`**: A learnable linear map $(64 \times 64)$ that mixes the channels of the rotor bundle.
*   **`geometry_conditioned_attention_bias`**: Implementation of the math $\langle \Psi, Q \wedge K \rangle$.

### `gca_pytorch.py`
Optimization wrapper.
*   **`class GeoAttentionBias`**: A `nn.Module` wrapper for the attention bias calculation.
    *   **`wedge_indices`**: Pre-buffers the wedge product indices so they move to GPU with the model, avoiding CPU overhead during training.
    *   Uses `torch.einsum` for efficient tensor contraction of the bias equation.

### `patch.py`
The "Surgery" module.
*   **`patch_llama_attention()`**: Helper function to locate `LlamaAttention` modules in the HuggingFace model hierarchy.
*   **`gca_attention_forward()`**: A re-implementation of the Llama Attention Forward Pass that accepts an injected `hybrid_wrapper`.
    *   It intercepts $Q$ and $K$.
    *   Computes the standard Softmax Attention.
    *   Adds the **Geometric Bias** before the Softmax.
    *   Returns the modified attention weights.

### `sync.py`
Utility for synchronization.
*   Placeholder for distributed logic (if scaling to multi-GPU), ensuring the PSI state is consistent across devices.

### `loss.py`
*   **Geometric Auxiliary Loss**: (If training) Defines constraints to force the model to respect algebraic grades (e.g., punishing a "Vector" output when a "Bivector" relationship was expected).

### `logic_gate.py`
*   **`GeoLogicGate`**: A learnable, sigmoid-activated gate that determines the value of $\lambda$ (the influence of Geometry). It allows the model to dynamically decide when to use "Logic" vs "Statistics".

---

## **4. Rust Runtime** `Aethelgard-X/`
(Included for reference as the high-performance backend proof)

*   **`main.rs`**: UCI (Universal Chess Interface) loop proving the logic can play games.
*   **`cga.rs`**: Rust implementation of the exact same logic in `cga.py`, but utilizing struct memory layout for CPU cache optimization.
*   **`geometry_tables.rs`**: The hard-coded "Unrolled" multiplication tables for extreme speed.
