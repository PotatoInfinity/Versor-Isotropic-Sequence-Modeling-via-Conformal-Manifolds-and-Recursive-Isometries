import torch
import numpy as np
from geo_llama.core.layers import GeoLlamaState, GeometricLiftingLayer

def test_geo_llama_poc():
    print("Initializing Geo-Llama Hybrid PoC (Torch Version)...")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 1. Setup dimensions
    d_model = 2048
    num_heads = 64
    
    # 2. Setup G-Stream State (Context Rotor PSI)
    state = GeoLlamaState(num_heads=num_heads, d_model=d_model, device=device)
    print(f"PSI State initialized with shape: {state.psi.shape}")
    
    # 3. Setup Lifting Layer
    lifter = GeometricLiftingLayer(d_model=d_model, num_heads=num_heads).to(device)
    
    # 4. Simulate a dummy input hidden state (e.g., from Llama 3.2 1B)
    # Shape: (batch=1, seq_len=5, d_model=2048)
    dummy_hidden_state = torch.randn(1, 5, d_model, device=device)
    print(f"Input Hidden State: {dummy_hidden_state.shape}")
    
    # 5. Lifting: Map to Geometry Heads
    lifted_rotors = lifter.lift(dummy_hidden_state)
    print(f"Lifted Geometric Rotors: {lifted_rotors.shape}")
    
    # 6. Recursive Update (Processing sequence tokens)
    print("Simulating Recursive Rotor Accumulation...")
    for t in range(lifted_rotors.shape[1]):
        # Extract token 't' rotors
        # lifted_rotors shape: (1, seq, heads, 32)
        rotors_t = lifted_rotors[0, t] # (64, 32)
        
        # Update PSI
        state.update(rotors_t)
        
        # Check scalar part consistency (should stay near 1.0 or normalized)
        scalar_mean = state.psi[:, 0].mean().item()
        print(f"Token {t}: PSI Scalar Mean = {scalar_mean:.4f}")

    print("\nPoC Test Complete. Geometric State PSI successfully accumulated.")

if __name__ == "__main__":
    test_geo_llama_poc()
