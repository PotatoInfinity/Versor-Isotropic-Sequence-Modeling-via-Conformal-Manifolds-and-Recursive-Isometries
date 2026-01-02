import numpy as np
from .core.cga import batch_wedge_product, BIVECTOR_INDICES

class StructuralVisualizer:
    """
    Translates high-dimensional Cl(4,1) relationship planes into human-readable 
    structural insights.
    """
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        # Basis bivector names for Cl(4,1)
        self.bivector_names = [
            "e12", "e13", "e1+", "e1-", "e23", "e2+", "e2-", "e3+", "e3-", "e+-"
        ]

    def analyze_relationships(self, tokens, Q_manifold, K_manifold):
        """
        Computes the 'Plane of Thought' between all token pairs.
        tokens: list of strings
        Q_manifold, K_manifold: (seq_len, 32)
        """
        seq_len = len(tokens)
        
        # Broadcast for pairwise interaction: (seq, 1, 32) and (1, seq, 32)
        Q_exp = Q_manifold[:, np.newaxis, :]
        K_exp = K_manifold[np.newaxis, :, :]
        
        # Wedge product: (seq_len, seq_len, 32)
        planes = batch_wedge_product(Q_exp, K_exp)
        
        print("\n--- Geometric Relationship Map ---")
        print(f"{'Token A':<15} | {'Token B':<15} | {'Primary Plane':<15} | {'Magnitude'}")
        print("-" * 65)
        
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                # Extract bivector components (indices 6-15)
                bivector_part = planes[i, j, 6:16]
                mag = np.linalg.norm(bivector_part)
                
                if mag > 0.1: # Only show significant relationships
                    max_idx = np.argmax(np.abs(bivector_part))
                    plane_name = self.bivector_names[max_idx]
                    
                    token_a = tokens[i]
                    token_b = tokens[j]
                    
                    print(f"{token_a:<15} | {token_b:<15} | {plane_name:<15} | {mag:.4f}")

    def plot_psi_state(self, psi):
        """
        Visualizes the current Context Rotor's orientation.
        """
        print("\n--- PSI Context State Visualizer ---")
        for h in range(4): # Show first 4 heads
            bivector_energy = np.linalg.norm(psi[h, 6:16])
            trivector_energy = np.linalg.norm(psi[h, 16:26])
            
            print(f"Head {h:02}: Bi-Energy: {bivector_energy:.4f} | Tri-Energy: {trivector_energy:.4f}")
