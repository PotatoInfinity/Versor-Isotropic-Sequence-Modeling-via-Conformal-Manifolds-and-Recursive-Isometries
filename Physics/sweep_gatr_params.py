import torch
import sys
import os

# Add paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "gatr"))
sys.path.append(os.path.join(root_dir, "Physics"))

from gatr import GATr, SelfAttentionConfig, MLPConfig

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test_gatr(mv, s, blocks):
    model = GATr(
        in_mv_channels=1, 
        out_mv_channels=1,
        in_s_channels=None,
        out_s_channels=None,
        hidden_mv_channels=mv,
        hidden_s_channels=s,
        num_blocks=blocks, 
        attention=SelfAttentionConfig(), 
        mlp=MLPConfig()
    )
    return count_parameters(model)

def main():
    print(f"{'MV':<5} | {'S':<5} | {'Blocks':<8} | {'Param Count':<15}")
    print("-" * 40)
    
    configs = [
        (16, 32, 10),
        (16, 64, 10),
        (16, 128, 10),
        (32, 128, 10),
        (32, 128, 20),
    ]
    
    for mv, s, b in configs:
        count = test_gatr(mv, s, b)
        print(f"{mv:<5} | {s:<5} | {b:<8} | {count:<15,}")

if __name__ == "__main__":
    main()
