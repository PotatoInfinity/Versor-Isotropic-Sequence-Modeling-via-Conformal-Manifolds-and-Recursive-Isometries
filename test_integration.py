import torch
import torch.nn as nn
from geo_llama.core.model import GeoLlamaHybrid

class MockAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_heads
        self.head_dim = d_model // num_heads
        self.hidden_size = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

    def forward(self, hidden_states, **kwargs):
        # This will be replaced by the patcher
        return self.o_proj(hidden_states), None, None

class MockLlama(nn.Module):
    def __init__(self, vocab_size=1000, d_model=128, num_heads=8):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({"self_attn": MockAttention(d_model, num_heads)})
        ])
        # Manually set class names to trick the patcher
        self.layers[0]["self_attn"].__class__.__name__ = "LlamaAttention"
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if inputs_embeds is None:
            hidden = self.embed(input_ids)
        else:
            hidden = inputs_embeds
            
        # Run layers
        for layer in self.layers:
            hidden, _, _ = layer["self_attn"](hidden)
            
        logits = self.proj(hidden)
        from types import SimpleNamespace
        return SimpleNamespace(
            logits=logits,
            hidden_states=[hidden]
        )

    def get_input_embeddings(self):
        return self.embed

def test_hybrid_integration():
    print("Verifying GeoLlamaHybrid Integration...")
    
    vocab_size = 1000
    d_model = 2048 # Standard for 1B
    num_heads = 64
    seq_len = 10
    
    mock_base = MockLlama(vocab_size=vocab_size, d_model=d_model, num_heads=num_heads)
    hybrid = GeoLlamaHybrid(mock_base, d_model=d_model, num_heads=num_heads)
    
    dummy_input = torch.randint(0, vocab_size, (1, seq_len))
    
    print(f"Running forward pass with seq_len={seq_len}...")
    outputs = hybrid(input_ids=dummy_input)
    
    print(f"Logits shape: {outputs.logits.shape}")
    
    if outputs.logits.shape == (1, seq_len, vocab_size):
        print("SUCCESS: Hybrid model forward pass completed.")
    else:
        print("FAILURE: Incorrect logits shape.")

if __name__ == "__main__":
    test_hybrid_integration()
