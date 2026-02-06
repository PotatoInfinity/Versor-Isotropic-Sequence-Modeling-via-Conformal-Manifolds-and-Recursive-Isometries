import torch
import sys
import os

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "gatr"))
sys.path.append(os.path.join(root_dir, "Physics"))

from Physics.run_gatr import GATrAdapter
from Physics.models import StandardTransformer, VersorRotorRNN, GraphNetworkSimulator, HamiltonianNN, MambaSimulator, EquivariantGNN, HamiltonianVersorNN

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    print(f"{'Model':<20} | {'Param Count':<15}")
    print("-" * 40)
    
    models = {
        "Versor (1-ch)": VersorRotorRNN(n_particles=5),
        "Transformer": StandardTransformer(n_particles=5),
        "GNS": GraphNetworkSimulator(n_particles=5),
        "HNN": HamiltonianNN(n_particles=5),
        "Mamba": MambaSimulator(n_particles=5),
        "EGNN": EquivariantGNN(n_particles=5),
        "GATr (Table 2)": GATrAdapter(),
        "Ham-Versor": HamiltonianVersorNN(n_particles=5)
    }
    
    for name, model in models.items():
        count = count_parameters(model)
        print(f"{name:<20} | {count:<15,}")

if __name__ == "__main__":
    main()
