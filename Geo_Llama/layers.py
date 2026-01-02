import torch
import torch.nn as nn
from .cga import (
    batch_geometric_product, exp_map, inverse, normalize_rotor, 
    VECTOR_INDICES, BIVECTOR_INDICES, QUADVECTOR_INDICES
)

class ManifoldMixingLayer(nn.Module):
    """
    Implements 'Manifold Mixing' (Section 9.2).
    Allows information to bleed between the 64 parallel rotors to solve the Binding Problem.
    """
    def __init__(self, num_heads=64):
        super().__init__()
        self.num_heads = num_heads
        # Mixing matrix: (64, 64)
        # We process (64, 32) -> (32, 64) -> Linear -> (32, 64) -> (64, 32)
        self.mixer = nn.Linear(num_heads, num_heads, bias=False)
        
        # Initialize near Identity to preserve independent manifolds initially
        with torch.no_grad():
            self.mixer.weight.copy_(torch.eye(num_heads))
            # Add small noise to allow gradient flow
            self.mixer.weight.add_(torch.randn(num_heads, num_heads) * 0.01)

    def forward(self, psi):
        # psi: (num_heads, 32)
        # Transpose to mix heads
        psi_t = psi.t() # (32, 64)
        mixed_t = self.mixer(psi_t)
        return mixed_t.t() # (64, 32)

class GeoLlamaState:
    """
    Maintains the structural context PSI (Context Rotor) for the Geo-Llama stream.
    """
    def __init__(self, num_heads=64, d_model=2048, device='cpu'):
        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device
        # PSI is a collection of 64 rotors, each with 32 components.
        # Initialized to identity (Scalar 1.0)
        self.psi = torch.zeros((num_heads, 32), device=device)
        self.psi[:, 0] = 1.0

    def update(self, rotors, mixing_layer=None):
        """
        Recursive Rotor Accumulation (Sandwich Product): PSI_t = R_t * PSI_t-1 * R_t_inv
        rotors: torch tensor of shape (num_heads, 32)
        mixing_layer: Optional ManifoldMixingLayer
        """
        # 1. Inverse of the incoming rotors
        rotors_inv = inverse(rotors)
        
        # 2. Update PSI: PSI = R * PSI * R_inv
        # This preserves the geometric object's properties as it 'moves'
        temp = batch_geometric_product(rotors, self.psi)
        self.psi = batch_geometric_product(temp, rotors_inv)
        
        # 3. Manifold Correction (Drift Mitigation)
        self.psi = normalize_rotor(self.psi)
        
        # 4. Manifold Mixing (Section 9.2 - Binding Problem Solution)
        if mixing_layer is not None:
            self.psi = mixing_layer(self.psi)
            self.psi = normalize_rotor(self.psi) # Ensure we stay on manifold
        
        # 5. Detach from graph to prevent infinite memory growth during inference
        self.psi = self.psi.detach()

    def normalize(self):
        self.psi = normalize_rotor(self.psi)

class SpecializedLiftingLayer(nn.Module):
    """
    Implements 'Head Specialization' (Section 2.1).
    Partitions 64 heads into different functional manifolds:
    - Heads 0-9: Syntactic (10 Bivectors, Small Scale)
    - Heads 10-39: Semantic (5 Vectors + 10 Bivectors, Medium Scale)
    - Heads 40-63: Narrative (10 Bivectors + 5 Quadvectors, Large Scale)
    """
    def __init__(self, d_model=2048, num_heads=64):
        super().__init__()
        assert num_heads == 64, "Specialization logic tuned for 64 heads"
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Dimensions per group
        self.syntax_heads = slice(0, 10)
        self.semantic_heads = slice(10, 40)
        self.narrative_heads = slice(40, 64)
        
        # Projections
        # Syntax: 10 bivectors
        self.proj_syntax = nn.Linear(d_model, 10 * 10)
        # Semantic: 5 vectors + 10 bivectors = 15
        self.proj_semantic = nn.Linear(d_model, 30 * 15)
        # Narrative: 10 bivectors + 5 quadvectors = 15
        self.proj_narrative = nn.Linear(d_model, 24 * 15)
        
        # Initialization Scales (Section 2.1: short, medium, long scale)
        with torch.no_grad():
            self.proj_syntax.weight *= 0.1  # Short-scale
            self.proj_semantic.weight *= 0.5 # Medium-scale
            self.proj_narrative.weight *= 1.5 # Long-scale (global context)

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        returns: (batch, seq_len, 64, 32) - Rotors
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        dtype = x.dtype
        
        # Project each group
        syn_out = self.proj_syntax(x).view(batch_size, seq_len, 10, 10)
        sem_out = self.proj_semantic(x).view(batch_size, seq_len, 30, 15)
        nar_out = self.proj_narrative(x).view(batch_size, seq_len, 24, 15)
        
        # Map to full 32-lane multivectors
        B_full = torch.zeros(batch_size, seq_len, 64, 32, device=device, dtype=dtype)
        
        # Syntax mapping (Head 0-9): 10 Bivectors
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 0:10, idx] = syn_out[..., i]
            
        # Semantic mapping (Head 10-39): 5 Vectors + 10 Bivectors
        for i, idx in enumerate(VECTOR_INDICES):
            B_full[..., 10:40, idx] = sem_out[..., i]
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 10:40, idx] = sem_out[..., i + 5]
            
        # Narrative mapping (Head 40-63): 10 Bivectors + 5 Quadvectors
        for i, idx in enumerate(BIVECTOR_INDICES):
            B_full[..., 40:64, idx] = nar_out[..., i]
        for i, idx in enumerate(QUADVECTOR_INDICES):
            B_full[..., 40:64, idx] = nar_out[..., i + 10]
            
        # Map to Rotors
        Rotors = exp_map(-B_full / 2.0)
        return Rotors
    
    def lift(self, x):
        return self.forward(x)

class GeometricLiftingLayer(SpecializedLiftingLayer):
    """Alias for backwards compatibility if needed, now specialized."""
    pass

def geometry_conditioned_attention_bias(psi, Q_lifted, K_lifted, lambda_val=0.1):
    """
    Computes the GCA bias: lambda * <PSI, Q ^ K>
    psi: (num_heads, 32)
    Q_lifted: (batch, n_heads, seq_len, 32)
    K_lifted: (batch, n_heads, seq_len, 32)
    """
    from .cga import batch_wedge_product, batch_inner_product
    
    # 1. Pairwise relationship plane
    # (batch, n_heads, seq_len, 1, 32) ^ (batch, n_heads, 1, seq_len, 32)
    rel_plane = batch_wedge_product(Q_lifted.unsqueeze(3), K_lifted.unsqueeze(2))
    
    # 2. Agreement with PSI
    # psi: (batch, n_heads, 32) or (n_heads, 32)
    # Ensure psi matches heads in rel_plane: (batch, n_heads, seq, seq, 32)
    if psi.dim() == 2:
        psi_exp = psi.view(1, psi.shape[0], 1, 1, 32)
    else:
        psi_exp = psi.view(psi.shape[0], psi.shape[1], 1, 1, 32)
        
    bias = batch_inner_product(psi_exp, rel_plane)
    
    return lambda_val * bias
