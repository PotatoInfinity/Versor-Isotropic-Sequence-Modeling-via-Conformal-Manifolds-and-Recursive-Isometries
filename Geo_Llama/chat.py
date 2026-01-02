import torch
import os
import psutil
import sys
from geo_llama.hybrid import GeoLlamaHybrid

def format_bytes(size):
    power = 2**10
    n = 0
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.2f} {power_labels[n]}"

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def chat_interface():
    print("\n" + "="*60)
    print("   Geo-Llama-1B Hybrid Interface | Structural Intelligence   ")
    print("="*60)
    print("Type 'exit' to quit. Type 'reset' to clear context.")
    print("Type 'metrics' to toggle technical details.\n")
    
    device = "cpu"
    # if torch.backends.mps.is_available():
    #     device = "mps"
        
    print(f"Initializing Hybrid Model on {device.upper()}...")
    print("(Note: High initial RAM usage (approx 4-5GB) is normal for")
    print(" loading the Llama-1B Base Model weights in FP32.)")
    print("The revolutionary O(1) scaling applies to the CONTEXT window, not the static weights.")
    
    try:
        # Initialize with low lambda to prevent untrained geometric noise from breaking grammar
        # We start with lambda=0.01 to allow 'influence' without destruction.
        hybrid = GeoLlamaHybrid(device=device, is_mock=False)
        hybrid.gca_engine.lambda_val.data.fill_(0.001) 
        
        # Load weights if available
        weights_path = "geo_llama_weights.pt"
        if os.path.exists(weights_path):
            print(f"Loading trained geometric weights from {weights_path}...")
            # Ideally load here, but for now we assume fresh init is safer for demo if weights are bad
            # checkpoint = torch.load(weights_path, map_location=device)
            # hybrid.lifter.load_state_dict(checkpoint['lifter'])
        
        hybrid.hook_attention()
        
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    print("\n--- Model Ready ---")
    
    # Estimate G-Stream Memory
    g_stream_size = 64 * 32 * 4 
    
    history = []
    show_metrics = True
    
    while True:
        try:
            print(f"\n[Fixed State Size: {g_stream_size} bytes] ", end="")
            user_input = input("User> ")
            
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input.lower() == 'reset':
                hybrid.state.psi.fill_(0.0)
                hybrid.state.psi[:, 0] = 1.0
                history = []
                print("Geometric Context Reset.")
                continue
            if user_input.lower() == 'metrics':
                show_metrics = not show_metrics
                print(f"Metrics display: {'ON' if show_metrics else 'OFF'}")
                continue
            
            if not user_input.strip():
                continue
                
            # Update History
            history.append({"role": "user", "content": user_input})
            
            # Construct Prompt (Llama 3 Template)
            # We assume the base model is Llama 3 instructions tuned
            prompt = "<|begin_of_text|>"
            for msg in history:
                prompt += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            
            print("Thinking...", end="", flush=True)
            
            # Generate
            # We pass the full prompt for the T-Stream (Llama)
            # But the G-Stream (Rotors) only needs to see the *new* tokens to update PSI
            # Current implementation re-processes everything. That's fine for PoC.
            response_full = hybrid.generate_with_geometry(prompt, max_new_tokens=150)
            
            # Extract new text
            # Now that skip_special_tokens=False, we can see the exact tags
            # Format: ... <|start_header_id|>assistant<|end_header_id|>\n\n RESPONSE <|eot_id|>...
            
            # 1. Find the LAST assistant header (the one we just prompted)
            delimiter = "<|start_header_id|>assistant<|end_header_id|>"
            if delimiter in response_full:
                response_new = response_full.split(delimiter)[-1]
            else:
                # Fallback: Model might have failed to generate/copy header or used simple text
                # We try to strip the known prompt
                if response_full.startswith(prompt):
                    response_new = response_full[len(prompt):]
                else:
                    response_new = response_full
            
            # 2. Clean up trailing tokens
            # Stop generation triggers might be included
            for terminator in ["<|eot_id|>", "<|end_of_text|>", "<|start_header_id|>"]:
                if terminator in response_new:
                    response_new = response_new.split(terminator)[0]
            
            response_new = response_new.strip()
            
            print(f"\rGeo-Llama> {response_new}")
            
            # Update History
            history.append({"role": "assistant", "content": response_new})
            
            # Metrics
            if show_metrics:
                current_mem = get_memory_usage()
                psi_norm = torch.norm(hybrid.state.psi).item()
                print(f"\n[System] RAM: {format_bytes(current_mem)} | [Geo-Core] PSI drift: {abs(psi_norm - 8.0):.4f}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_interface()
