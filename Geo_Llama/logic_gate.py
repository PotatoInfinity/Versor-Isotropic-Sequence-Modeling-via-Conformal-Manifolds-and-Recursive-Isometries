import torch
import torch.nn as nn

class LogicGate(nn.Module):
    """
    Implements the 'Topological Veto' mechanism (Section 3 & 10.3).
    Merges statistical logits from the LLM with geometric validity scores from the G-Stream.
    """
    def __init__(self, lambda_val=0.5):
        super().__init__()
        # lambda can be learned or fixed
        self.lambda_val = nn.Parameter(torch.tensor(lambda_val))

    def forward(self, llama_logits, geometric_validity_scores):
        """
        llama_logits: [Batch, Vocab] (Statistical prediction)
        geometric_validity_scores: [Batch, Vocab] (Topological feasibility)
        
        Geometric validity scores should be:
        - Near 0 for valid transitions
        - Highly negative (-inf) for impossible transitions
        
        Formula: P(x) = Softmax(Logits + lambda * GeoValidity)
        """
        # Ensure scores are on same device
        if geometric_validity_scores.device != llama_logits.device:
            geometric_validity_scores = geometric_validity_scores.to(llama_logits.device)
            
        integrated_logits = llama_logits + (self.lambda_val * geometric_validity_scores)
        
        return integrated_logits

def compute_topological_veto(context_psi, candidate_embeddings, lifting_layer):
    """
    Helper to compute geometric validity scores for a set of candidates.
    context_psi: (num_heads, 32)
    candidate_embeddings: (vocab_size, d_model)
    lifting_layer: GeometricLiftingLayer
    
    Returns: (vocab_size,) validity scores
    """
    # 1. Lift candidates to rotors
    # candidate_rotors: (vocab_size, num_heads, 32)
    candidate_rotors = lifting_layer.lift(candidate_embeddings.unsqueeze(0)).squeeze(0)
    
    # 2. Check consistency: <Context_PSI, Candidate_Rotor>
    # Validity is the 'agreement' between the current geometric context and the candidate action.
    from .cga import batch_inner_product
    
    # validity: (vocab_size, num_heads)
    # We sum or average over heads
    validity = batch_inner_product(context_psi.unsqueeze(0), candidate_rotors)
    
    # 3. Aggregate across heads (consensus)
    # If any head strongly disagrees (veto), the total should reflect it.
    validity_scores = torch.mean(validity, dim=-1)
    
    return validity_scores
