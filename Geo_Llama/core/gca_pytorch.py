import torch
import torch.nn as nn
import numpy as np
from .cga import GP_MAP

class GeoAttentionBias(nn.Module):
    """
    PyTorch implementation of the Geometry-Conditioned Attention (GCA) bias.
    Formula: bias = lambda * <PSI, Q ^ K>
    """
    def __init__(self, num_heads=64, lambda_val=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.lambda_val = nn.Parameter(torch.tensor(lambda_val))
        
        # Precompute the wedge product map for PyTorch
        # We only care about basis pairs (a, b) whose wedge product contributes to a bivector (grade 2) or higher logical grade if needed.
        # But for the PoC, we'll map the full GP_MAP to PyTorch indices.
        
        indices = []
        signs = []
        for sign, a, b, k in GP_MAP:
            # Filtering for wedge product: grade(k) == grade(a) + grade(b)
            # Actually, we can just use the full geometric product logic if we want,
            # but GCA specifically calls for the anti-symmetric part.
            grade_a = bin(a).count('1')
            grade_b = bin(b).count('1')
            grade_k = bin(k).count('1')
            
            if grade_k == (grade_a + grade_b):
                indices.append([a, b, k])
                signs.append(sign)
        
        self.register_buffer("wedge_indices", torch.tensor(indices, dtype=torch.long))
        self.register_buffer("wedge_signs", torch.tensor(signs, dtype=torch.float32))

    def forward(self, psi, Q_lifted, K_lifted):
        """
        psi: (num_heads, 32)
        Q_lifted: (batch, num_heads, seq_len, 32)
        K_lifted: (batch, num_heads, seq_len, 32)
        
        Returns: (batch, num_heads, seq_len, seq_len)
        """
        batch, n_heads, seq_len, _ = Q_lifted.shape
        
        # 1. Compute pairwise wedge product Q_i ^ K_j
        # We'll use a gathered approach to avoid giant dummy tensors
        # rel_plane = Q_lifted[:, :, i] ^ K_lifted[:, :, j]
        
        # Since we need all-to-all attention bias (seq_len x seq_len),
        # we compute the products for each basis pair.
        
        # This is a memory-intensive operation for long sequences.
        # For PoC, we'll optimize by only computing bivector components.
        
        # bias_matrix = torch.zeros(batch, n_heads, seq_len, seq_len, device=Q_lifted.device)
        
        # Vectorized wedge + inner product:
        # bias[b, h, i, j] = lambda * sum_{k} psi[h, k] * sum_{indices(a,b->k)} sign * Q[b, h, i, a] * K[b, h, j, b]
        
        # Using Einstein summation for efficiency:
        # We can pre-sum the PSI parts:
        # Effective_Weight[h, a, b] = sum_{k} psi[h, k] * sign(a,b->k) * metric(k)
        
        # Define the metric for inner product <PSI, Bivector>
        # In CGA Cl(4,1), the inner product <A, B> depends on the metric of the basis blades.
        # Most bivectors in Cl(4,1) have negative or mixed metrics.
        
        # Simplified for PoC: we assume a Euclidean-lifting for the bias calculation
        # to focus on the logical "intersection" property.
        
        # Compute the "Correlation Kernel" for each head
        # weight_matrix: (num_heads, 32, 32)
        weight_matrix = torch.zeros(self.num_heads, 32, 32, device=psi.device)
        
        for idx in range(len(self.wedge_indices)):
            a, b, k = self.wedge_indices[idx]
            sign = self.wedge_signs[idx]
            
            # The inner product <psi, k>
            # We need the metric of basis k.
            metric_k = 1.0
            # Basic Cl(4,1) metric: e1..e4=+1, e5=-1
            # For a blade k, the metric is the product of its components.
            for bit in range(5):
                if (k >> bit) & 1:
                    if bit == 4: # e- (e5)
                        metric_k *= -1.0
            
            # Contribution to the attention weight between basis a and b
            weight_matrix[:, a, b] += sign * psi[:, k] * metric_k
            
        # Final Bias = lambda * Q^T * Weight * K
        # Q: (batch, n_heads, seq_len, 32)
        # Weight: (n_heads, 32, 32)
        # K: (batch, n_heads, seq_len, 32)
        
        # einsum: z,h,i,a ; h,a,b ; z,h,j,b -> z,h,i,j
        bias = torch.einsum('zhia,hab,zhjb->zhij', Q_lifted, weight_matrix, K_lifted)
        
        return self.lambda_val * bias

def batch_wedge_product(A, B, wedge_indices, wedge_signs):
    """
    Differentiable wedge product in PyTorch.
    A, B: (..., 32)
    """
    out = torch.zeros_like(A)
    # This is broad and potentially slow, but works for PoC training
    for idx in range(len(wedge_indices)):
        a, b, k = wedge_indices[idx]
        sign = wedge_signs[idx]
        out[..., k] += sign * A[..., a] * B[..., b]
    return out

def batch_inner_product(A, B):
    """
    Metric-aware inner product in PyTorch.
    """
    dot = A[..., 0] * B[..., 0] # Scalar
    dot += A[..., 1] * B[..., 1] # e1
    dot += A[..., 2] * B[..., 2] # e2
    dot += A[..., 3] * B[..., 3] # e3
    dot += A[..., 4] * B[..., 4] # e+
    dot -= A[..., 5] * B[..., 5] # e- (Minkowski)
    return dot
