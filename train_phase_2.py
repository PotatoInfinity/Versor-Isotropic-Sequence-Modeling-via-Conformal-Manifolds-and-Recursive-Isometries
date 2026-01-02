import torch
import torch.optim as optim
from transformers import AutoModelForCausalLM
from geo_llama.core.model import GeoLlamaHybrid

def train_phase_2(model_name="unsloth/Llama-3.2-1B"):
    """
    Phase 2 (Gated Finetuning):
    Unfreeze the T-Stream but initialize lambda (mixing constant) to 0.
    Slowly ramp up lambda, allowing the model to lean on the 'Logical Crutch'.
    """
    print(f"Starting Training Phase 2 with {model_name}...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Base Llama
    base_llama = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 2. Wrap in Hybrid
    hybrid_model = GeoLlamaHybrid(base_llama, d_model=2048, num_heads=64).to(device)
    
    # 3. Unfreeze T-Stream
    for param in hybrid_model.llama.parameters():
        param.requires_grad = True
    
    # 4. Initialize lambda to 0 (G-Stream starts with no influence)
    hybrid_model.gate.lambda_val.data.fill_(0.0)
    
    # 5. Optimizer for everything
    optimizer = optim.AdamW(hybrid_model.parameters(), lr=1e-5)
    
    # 6. Training Loop with Lambda Normalization/Ramping
    step = 0
    max_steps = 1000
    
    def step_training():
        nonlocal step
        # Ramping up lambda
        lambda_target = 0.5
        current_lambda = min(lambda_target, (step / max_steps) * lambda_target)
        # We can update it manually or let the optimizer handle it after initialization
        
        # training logic...
        step += 1
    
    print("Phase 2 setup complete.")
    return hybrid_model

if __name__ == "__main__":
    try:
        train_phase_2()
    except Exception as e:
        print(f"Skipping Phase 2 run: {e}")
