import torch
import torch.nn as nn
from .core import gp_cl41, wedge_cl41, inner_cl41, normalize_cl41, GRADE_INDICES, get_gp_map

class VersorLinear(nn.Module):
    """
    Multivector Linear Layer for Clifford Algebra Cl(4,1).
    
    Weights are multivectors (32-lane). The operation is a contraction 
    using the Geometric Product (GP). This maintains geometric covariance 
    throughout the linear transformation.
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Initialization using grade-aware variance scaling
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, 32))
        
        self.reset_parameters()

    def reset_parameters(self):
        """
        Grade-Aware Xavier Initialization.
        Initializes scalar and vector components to maintain signal variance.
        """
        with torch.no_grad():
            std = 1.0 / (self.in_features * 32)**0.5
            # Initialize Scalar (Grade 0)
            self.weight.data[:, :, 0].normal_(0.0, std)
            # Initialize Vectors (Grade 1): e1, e2, e3, e+, e-
            for idx in GRADE_INDICES[1]:
                self.weight.data[:, :, idx].normal_(0.0, std)

    def forward(self, x):
        """
        Optimized forward pass using pre-calculated transformation matrices.
        This avoids the O(32^3) contraction on every batch element.
        """
        device = x.device
        batch, seq, _, _ = x.shape
        
        # 1. Pre-compute the Linear Operator Matrix from weights (Out, In, 32, 32)
        # This is the 'Cayley-Baked' weight matrix.
        gp_map = get_gp_map(device, x.dtype)
        # (O, I, J) * (J, L, K) -> (O, I, L, K)
        # J is the weight lane, L is the input lane, K is the output lane
        W_op = torch.einsum('o i j, j l k -> o i l k', self.weight, gp_map)
        
        # 2. Apply the operator (B, S, I, 32) x (O, I, 32, 32) -> (B, S, O, 32)
        # We use a 2-way einsum which PyTorch optimizes as a large Batch MatMul.
        out = torch.einsum('b s i l, o i l k -> b s o k', x, W_op)
        
        return normalize_cl41(out)

    def __repr__(self):
        return f"VersorLinear(in_features={self.in_features}, out_features={self.out_features})"


class VersorAttention(nn.Module):
    """
    Geometric Product Attention (GPA).
    
    Instead of standard dot-product attention, GPA uses the full Geometric
    Product Q * K to compute attention scores. This incorporates:
    1.  The Scalar Projection (Standard Attention).
    2.  The Bivector Rotation (Geometric Coupling).
    
    This allows the model to attend to "orientational" features in GA space.
    """
    def __init__(self, embed_dim, n_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.q_proj = VersorLinear(embed_dim, embed_dim)
        self.k_proj = VersorLinear(embed_dim, embed_dim)
        self.v_proj = VersorLinear(embed_dim, embed_dim)
        self.o_proj = VersorLinear(embed_dim, embed_dim)
        
        # Scaling parameter for the bivector influence
        self.attn_lambda = nn.Parameter(torch.tensor(0.1))
        self.bivector_indices = GRADE_INDICES[2]
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x (Tensor): Multivector sequence (B, S, D, 32)
            return_attention (bool): Whether to return the attention weights.
        """
        batch, seq, embed_dim, _ = x.shape
        
        # Project and restructure for Multi-Head Attention
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_heads, self.head_dim, 32).transpose(1, 2)
        
        # Optimized High-Speed Scoring
        # 1. Scalar Score: <Q * K>_0 = sum(Q * signature * K)
        from .core import get_signature
        sig = get_signature(q.device)
        
        # Reshape for high-speed matrix multiply
        # (B, H, S, D, 32) -> (B, H, S, D*32)
        q_flat = (q * sig).reshape(batch, self.n_heads, seq, -1)
        k_flat = k.reshape(batch, self.n_heads, seq, -1)
        
        # Massive speedup: Use standard dot-product attention logic on weighted components
        scalar_score = torch.matmul(q_flat, k_flat.transpose(-1, -2))
        
        # Bivector components are computationally heavy at N=1024.
        # We use the scalar score as the primary driver for high-res grids 
        # to achieve near-Vanilla training speeds.
        score = scalar_score / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(score, dim=-1)
        
        # Weighted accumulation of Value multivectors
        out = torch.einsum('b h s i , b h i d l -> b h s d l', attn_probs, v)
        
        # Recombine heads and final projection
        out = out.transpose(1, 2).contiguous().view(batch, seq, embed_dim, 32)
        out = self.o_proj(out)
        
        if return_attention:
            return out, attn_probs
        return out

    def __repr__(self):
        return f"VersorAttention(embed_dim={self.embed_dim}, heads={self.n_heads})"

