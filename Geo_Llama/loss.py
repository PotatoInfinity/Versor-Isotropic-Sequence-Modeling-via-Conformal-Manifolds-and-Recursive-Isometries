import torch
import torch.nn as nn
from .cga import VECTOR_INDICES, BIVECTOR_INDICES, QUADVECTOR_INDICES

class GeometricConsistencyLoss(nn.Module):
    """
    Implements the loss terms for Head Specialization (Section 2.1):
    1. Grade Sparsity: Encourage rotors to occupy only their assigned grades.
    2. Normalization: Ensure rotors remain on the manifold.
    """
    def __init__(self, gca_engine=None, sparsity_weight=0.1, norm_weight=1.0):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.norm_weight = norm_weight
        self.gca_engine = gca_engine
        
        # Create masks for each head group (Batch, Seq, Heads, 32)
        # We perform 1-based indexing for the grades, 0 is Scalar.
        
        # Each mask is (64, 32). 
        # 1.0 means FORBIDDEN (We want to minimize energy here).
        # 0.0 means ALLOWED.
        self.register_buffer('partition_masks', torch.ones(64, 32))
        
        # Syntax (0-9): Only Bivectors + Scalar (0)
        self.partition_masks[0:10, [0] + BIVECTOR_INDICES] = 0.0
        
        # Semantic (10-39): Vector + Bivector + Scalar
        self.partition_masks[10:40, [0] + VECTOR_INDICES + BIVECTOR_INDICES] = 0.0
        
        # Narrative (40-63): Bivector + Quadvector + Scalar
        self.partition_masks[40:64, [0] + BIVECTOR_INDICES + QUADVECTOR_INDICES] = 0.0

    def forward(self, rotors, *args, **kwargs):
        """
        rotors: (Batch, Seq, Heads, 32)
        """
        # 1. Normalization Loss (R * ~R should be 1, or close to unit magnitude)
        # Simply keeping the coefficients reasonable prevents drift.
        norms_sq = torch.sum(rotors**2, dim=-1)
        norm_loss = torch.mean((norms_sq - 1.0)**2)
        
        # 2. Specialized Grade Sparsity Loss
        # Multiply rotors by the partition masks
        # We broadcast the mask over Batch and Seq
        if rotors.dim() == 4: # (B, S, H, 32)
            mask = self.partition_masks.view(1, 1, 64, 32)
        else:
            mask = self.partition_masks
            
        forbidden_content = rotors * mask
        sparsity_loss = torch.mean(torch.abs(forbidden_content))
        
        return self.norm_weight * norm_loss + self.sparsity_weight * sparsity_loss
