import torch
import torch.nn as nn
import torch.optim as optim
from geo_llama.core.model import GeoLlamaHybrid
from test_integration import MockLlama

class GeometricConsistencyLoss(nn.Module):
    """
    Implements the loss terms from Section 6 with Head Specialization (Section 2.1):
    1. Grade Sparsity: Encourage rotors to occupy only their assigned grades.
    2. Normalization: Ensure rotors remain on the manifold.
    """
    def __init__(self, sparsity_weight=0.1, norm_weight=1.0):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.norm_weight = norm_weight
        
        from geo_llama.core.cga import VECTOR_INDICES, BIVECTOR_INDICES, QUADVECTOR_INDICES
        
        # Create masks for each head group
        # Each mask is (64, 32)
        self.register_buffer('partition_masks', torch.ones(64, 32))
        
        # Syntax (0-9): Only Bivectors + Scalar (0)
        self.partition_masks[0:10, [0] + BIVECTOR_INDICES] = 0.0
        
        # Semantic (10-39): Vector + Bivector + Scalar
        self.partition_masks[10:40, [0] + VECTOR_INDICES + BIVECTOR_INDICES] = 0.0
        
        # Narrative (40-63): Bivector + Quadvector + Scalar
        self.partition_masks[40:64, [0] + BIVECTOR_INDICES + QUADVECTOR_INDICES] = 0.0

    def forward(self, rotors):
        # rotors: (Batch, Seq, Heads, 32)
        
        # 1. Normalization Loss
        norms_sq = torch.sum(rotors**2, dim=-1)
        norm_loss = torch.mean((norms_sq - 1.0)**2)
        
        # 2. Specialized Grade Sparsity Loss
        # Multiply rotors by the partition masks (which contain 1.0 for FORBIDDEN grades)
        # We broadcast the mask over Batch and Seq
        forbidden_content = rotors * self.partition_masks.view(1, 1, 64, 32)
        sparsity_loss = torch.mean(torch.abs(forbidden_content))
        
        return self.norm_weight * norm_loss + self.sparsity_weight * sparsity_loss

def run_phase_1_demo():
    print("Initializing Phase 1 Demo (Frozen T-Stream)...")
    
    device = 'cpu'
    d_model = 128
    num_heads = 64
    vocab_size = 1000
    
    # 1. Setup Mock Model
    mock_base = MockLlama(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads)
    hybrid_model = GeoLlamaHybrid(mock_base, d_model=d_model, num_heads=num_heads).to(device)
    
    # 2. Freeze T-Stream
    for param in hybrid_model.llama.parameters():
        param.requires_grad = False
        
    # 3. Setup Optimizer & Loss
    optimizer = optim.AdamW(list(hybrid_model.lifter.parameters()) + 
                           list(hybrid_model.gate.parameters()), lr=1e-3)
    
    geo_loss_fn = GeometricConsistencyLoss()
    criterion = nn.CrossEntropyLoss()
    
    # 4. Dummy Training Step
    print("Starting training loop...")
    for i in range(5):
        optimizer.zero_grad()
        
        # input: (Batch, Seq)
        input_ids = torch.randint(0, vocab_size, (1, 10))
        target_ids = torch.randint(0, vocab_size, (1, 10))
        
        # Forward pass (this will patch attention and use pre-pass G-stream)
        outputs = hybrid_model(input_ids=input_ids)
        logits = outputs.logits # (1, 10, 1000)
        
        # Calculate standard LM loss on the Vetoed logits
        lm_loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
        
        # Calculate Geometric Loss on the rotors produced by lifter
        # We need to expose rotors or re-run lifter
        rotors = hybrid_model.lifter(hybrid_model.llama.get_input_embeddings()(input_ids))
        geo_loss = geo_loss_fn(rotors)
        
        total_loss = lm_loss + geo_loss
        total_loss.backward()
        optimizer.step()
        
        print(f"Iteration {i}: Total Loss = {total_loss.item():.4f}, LM Loss = {lm_loss.item():.4f}, Geo Loss = {geo_loss.item():.4f}")

    print("\nPhase 1 Demo Complete. G-Stream components successfully updated.")

if __name__ == "__main__":
    run_phase_1_demo()
