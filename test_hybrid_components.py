import torch
from geo_llama.core.layers import GeoLlamaState, GeometricLiftingLayer
from geo_llama.core.logic_gate import LogicGate, compute_topological_veto
from geo_llama.core.sync import FrameSynchronization, calculate_invariant_distance
from geo_llama.core.cga import normalize_rotor

def test_hybrid_components():
    print("Verifying LogicGate and FrameSynchronization...")
    
    device = 'cpu'
    d_model = 2048
    num_heads = 64
    vocab_size = 1000
    
    # 1. Setup
    state = GeoLlamaState(num_heads=num_heads, d_model=d_model, device=device)
    lifter = GeometricLiftingLayer(d_model=d_model, num_heads=num_heads).to(device)
    gate = LogicGate(lambda_val=10.0).to(device) # High lambda for visible effect
    sync = FrameSynchronization(num_heads=num_heads, sync_rate=0.5).to(device)
    
    # 2. Simulate Frame Sync
    print("\n--- Frame Synchronization Test ---")
    # Introduce drift to 32 heads (half)
    drifted_psi = state.psi.clone()
    drifted_psi[:32, 6] = 1.0 # Bivector e12
    drifted_psi[:32, 7] = 1.0 # Bivector e13
    
    # Normalize drifted rotors to manifold
    drifted_psi = normalize_rotor(drifted_psi)
    
    consensus_psi = normalize_rotor(torch.mean(drifted_psi, dim=0, keepdim=True))
    dist_before = calculate_invariant_distance(drifted_psi[0:1], consensus_psi)
    print(f"Distance to consensus before sync (Head 0): {dist_before.item():.4f}")
    
    synced_psi = sync(drifted_psi)
    new_consensus = normalize_rotor(torch.mean(synced_psi, dim=0, keepdim=True))
    dist_after = calculate_invariant_distance(synced_psi[0:1], new_consensus)
    print(f"Distance to consensus after sync (Head 0): {dist_after.item():.4f}")
    
    if dist_after < dist_before:
        print("SUCCESS: FrameSynchronization reduced manifold divergence.")
    else:
        print("FAILURE: FrameSynchronization did not reduce divergence.")

    # 3. Simulate Logic Gate / Topological Veto
    print("\n--- Logic Gate (Topological Veto) Test ---")
    llama_logits = torch.randn(1, vocab_size, device=device)
    candidate_embeddings = torch.randn(vocab_size, d_model, device=device)
    
    # Compute validity scores
    # validity scores are <PSI, Candidate_Rotor>
    validity_scores = compute_topological_veto(state.psi, candidate_embeddings, lifter)
    print(f"Validity Scores (Sample): {validity_scores[:5]}")
    
    # Integrate
    integrated_logits = gate(llama_logits, validity_scores.unsqueeze(0))
    
    # Check if a low-validity candidate gets penalized
    min_idx = torch.argmin(validity_scores).item()
    max_idx = torch.argmax(validity_scores).item()
    
    print(f"Candidate {min_idx} validity: {validity_scores[min_idx].item():.4f}, logit change: {integrated_logits[0, min_idx].item() - llama_logits[0, min_idx].item():.4f}")
    print(f"Candidate {max_idx} validity: {validity_scores[max_idx].item():.4f}, logit change: {integrated_logits[0, max_idx].item() - llama_logits[0, max_idx].item():.4f}")

    if (integrated_logits[0, max_idx] - llama_logits[0, max_idx]) > (integrated_logits[0, min_idx] - llama_logits[0, min_idx]):
        print("SUCCESS: LogicGate correctly favors geometrically valid tokens.")
    else:
        print("FAILURE: LogicGate did not bias logits correctly.")

if __name__ == "__main__":
    test_hybrid_components()
