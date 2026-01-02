import numpy as np
import torch
from geo_llama.hybrid import GeoLlamaHybrid
from geo_llama.visualizer import StructuralVisualizer
from geo_llama.logic_gates import GeometricLogicGate

def run_integrative_demo():
    print("--- Geo-Llama Structural Intelligence Demo ---")
    
    # 1. Setup
    hybrid = GeoLlamaHybrid(is_mock=True)
    viz = StructuralVisualizer()
    gate = GeometricLogicGate(hybrid)
    
    # 2. Process a Logical Sequence
    tokens = ["Atoms", "into", "Molecules", "into", "Cells"]
    print(f"\nProcessing Sequence: {' '.join(tokens)}")
    
    # Simulate lifting for these tokens
    dummy_hidden = torch.randn(1, len(tokens), hybrid.d_model)
    lifted = hybrid.lifter.lift(dummy_hidden.numpy())
    
    # Sync G-Stream
    for t in range(len(tokens)):
        hybrid.state.update(lifted[0, t])
        
    # 3. Visualizer: Show Relationship Planes
    # (Using head 0 tokens: Shape (5, 32))
    viz.analyze_relationships(tokens, lifted[0, :, 0, :], lifted[0, :, 0, :])
    
    # 4. Logic Gate: The 'Topological Veto'
    print("\n--- Testing Topological Veto ---")
    # Simulate a token candidate that contradicts the 'Containment' chain
    # e.g., a token that implies 'Cells are inside Atoms' (impossible logic)
    contradictory_candidate = lifted[0, 0] * -1.5 # Invert 'Atoms' to simulate conflict
    
    fake_logits = np.random.randn(10)
    candidates = [(0, contradictory_candidate)] # Token 0 is the conflict candidate
    
    print("Initial Logits for Candidate 0:", f"{fake_logits[0]:.4f}")
    new_logits = gate.apply_veto(fake_logits, candidates, hybrid.state.psi[0])
    print("Vetoed Logits for Candidate 0:", f"{new_logits[0]:.4f}")

    # 5. PSI Orientation
    viz.plot_psi_state(hybrid.state.psi)

if __name__ == "__main__":
    run_integrative_demo()
