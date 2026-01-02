import torch
import numpy as np
from .core.cga import batch_inner_product, batch_wedge_product, hodge_dual

class GeometricLogicGate:
    """
    Enforces 'Topological Veto' over probabilistic outputs using 
    Conformal Geometric Algebra constraints.
    """
    def __init__(self, hybrid):
        self.hybrid = hybrid

    def check_logical_consistency(self, candidate_rotor, context_psi):
        """
        Calculates the consistency between a candidate thought (Rotor)
        and the global structural context (PSI).
        
        Consistency = ScalarPart(Candidate * PSI)
        In CGA, valid relationships align with the current manifold orientation.
        """
        # We check how much the candidate 'rotates' the context toward 
        # destructive interference (negative scalar part).
        from .core.cga import batch_geometric_product
        
        # res = R * PSI
        res = batch_geometric_product(candidate_rotor, context_psi)
        
        # A high positive scalar component (lane 0) indicates alignment.
        # A negative scalar component indicates a logical contradiction.
        return res[..., 0]

    def check_intersection_validity(self, entity_a, entity_b):
        """
        Checks if two entities have a valid intersection.
        In CGA: Intersection = (A* ^ B*)*
        If the intersection result is zero, they are geometrically disjoint.
        """
        a_dual = hodge_dual(entity_a)
        b_dual = hodge_dual(entity_b)
        
        intersection_dual = batch_wedge_product(a_dual, b_dual)
        intersection = hodge_dual(intersection_dual)
        
        return np.linalg.norm(intersection)

    def apply_veto(self, logits, token_candidates, context_psi):
        """
        logits: (vocab_size)
        token_candidates: list of (token_id, lifted_representation)
        """
        modified_logits = logits.copy()
        
        for token_id, lifted in token_candidates:
            # We treat the 'lifted' representation as a candidate Rotor
            # If it's multi-head, we check the primary head (0)
            candidate = lifted[0] if lifted.ndim > 1 else lifted
            
            # 1. Check Alignment with Global Context
            consistency = self.check_logical_consistency(candidate, context_psi)
            
            # 2. Veto Logic
            if consistency < 0.0: # Negative alignment = Logical Contradiction
                penalty = abs(consistency) * 15.0 # Scalable penalty
                print(f"VETO [Head 0]: Token {token_id} inconsistent. Consistency: {consistency:.4f} | Penalty: -{penalty:.2f}")
                modified_logits[token_id] -= penalty
                
        return modified_logits
