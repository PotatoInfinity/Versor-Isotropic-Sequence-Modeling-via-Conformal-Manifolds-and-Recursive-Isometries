import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .layers import geometry_conditioned_attention_bias

def patch_llama_attention(model, hybrid_wrapper):
    """
    Patches all LlamaAttention layers in the model to include GCA bias.
    """
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaAttention" or module.__class__.__name__ == "LlamaSdpaAttention":
            # Store the original forward
            module.original_forward = module.forward
            # Assign the new forward with a closure to capture 'module' correctly
            def gca_forward_wrapper(*args, m=module, **kwargs):
                return gca_attention_forward(m, hybrid_wrapper, *args, **kwargs)
            module.forward = gca_forward_wrapper
            print(f"Patched {name} with GCA bias.")

def gca_attention_forward(
    self,
    hybrid_wrapper,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # 1. Run Original Projection (Q, K, V)
    # This is tricky because we need the internal variables.
    # A cleaner way is to wrap the original forward but intercept the scores.
    # However, HF forward doesn't easily expose scores before softmax.
    
    # Let's assume we can re-implement the core logic or use a simpler hook.
    # For PoC, let's substitute the forward entirely with a GCA-aware version.
    
    # B, L, D
    bsz, q_len, _ = hidden_states.size()
    
    # query_states shape: [bsz, q_len, q_hidden_dim]
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    q_hidden_dim = query_states.shape[-1]
    k_hidden_dim = key_states.shape[-1]
    
    # Robustly infer heads from dimensions
    # We assume 'head_dim' is correct (usually d_model / num_heads)
    # If head_dim is missing, we must infer it from Q.
    num_heads = getattr(self, 'num_heads', getattr(self, 'num_attention_heads', 32)) 
    
    if hasattr(self, 'head_dim'):
        head_dim = self.head_dim
    else:
        head_dim = q_hidden_dim // num_heads
        
    num_kv_heads = k_hidden_dim // head_dim
    
    # Reshape
    # (B, L, H*D) -> (B, L, H, D) -> (B, H, L, D)
    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_kv_heads, head_dim).transpose(1, 2)
    
    # 2b. Repeat KV for GQA to match num_heads
    if num_kv_heads != num_heads:
        # (B, n_kv, L, D) -> (B, n_heads, L, D)
        key_states = torch.repeat_interleave(key_states, dim=1, repeats=num_heads // num_kv_heads)
        value_states = torch.repeat_interleave(value_states, dim=1, repeats=num_heads // num_kv_heads)
    
    # --- THE FIX: APPLY ROPE ---
    cos, sin = self.rotary_emb(value_states, position_ids)
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    # ---------------------------
    
    # 2. Standard Attention Scores
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / (self.head_dim**0.5)
    
    # 3. Inject GCA Bias
    # We need the current PSI from the hybrid wrapper.
    # During parallel forward, we might use the 'current_psi' accumulated so far or 
    # a sequence-parallel version.
    if hybrid_wrapper.state is not None:
        # Lift Q and K
        # Q_lifted: (B, H, L, 32)
        Q_lifted = hybrid_wrapper.lifter(hidden_states) # Using hidden_states as proxy for Q/K projection
        K_lifted = hybrid_wrapper.lifter(hidden_states)
        
        # Reshape to match heads
        Q_lifted = Q_lifted.view(bsz, q_len, hybrid_wrapper.num_heads, 32).transpose(1, 2)
        K_lifted = K_lifted.view(bsz, q_len, hybrid_wrapper.num_heads, 32).transpose(1, 2)
        
        # Compute Bias: lambda * <PSI, Q^K>
        # Note: We use the last PSI or per-token PSI if available.
        # For prefill/training, we might approximate with the global context.
        bias = geometry_conditioned_attention_bias(
            hybrid_wrapper.state, Q_lifted, K_lifted, lambda_val=hybrid_wrapper.gate.lambda_val
        )
        
        # Project 64 Geo-Heads -> 32 Llama-Heads
        # bias: (B, 64, L, L)
        # attn_weights: (B, 32, L, L)
        if bias.shape[1] != attn_weights.shape[1]:
            ratio = bias.shape[1] // attn_weights.shape[1]
            if ratio > 1:
                # Average adjacent logic planes
                bias = bias.view(bsz, attn_weights.shape[1], ratio, q_len, q_len).mean(dim=2)
        
        attn_weights = attn_weights + bias
        
    # 4. Standard Softmax & V
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    
    attn_output = attn_output.transpose(1, 2).contiguous()
    hidden_size = getattr(self, 'hidden_size', self.q_proj.in_features)
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    
    attn_output = self.o_proj(attn_output)
    
    if output_attentions:
        return attn_output, attn_weights, past_key_values
    else:
        # Caller expects 2 values: (output, past_key_values) or (output, weights) 
        # based on 'hidden_states, _ = ...' pattern.
        # usually (hidden_states, past_key_values) or (hidden_states, weights)
        # Given standard SDPA/Llama logic:
        return attn_output, past_key_values
