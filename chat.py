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
    
    device = "cpu"  # Force CPU for stability on Mac unless MPS is verified
    # if torch.backends.mps.is_available():
    #     print("Metal Performance Shaders (MPS) detected. Using GPU acceleration.")
    #     device = "mps"
        
    print(f"\nInitializing Model on {device.upper()}...")
    
    try:
        hybrid = GeoLlamaHybrid(device=device, is_mock=False)
        
        # Load weights
        weights_path = "geo_llama_weights.pt"
        if os.path.exists(weights_path):
            print(f"Loading trained geometric weights from {weights_path}...")
            checkpoint = torch.load(weights_path, map_location=device)
            hybrid.lifter.load_state_dict(checkpoint['lifter'])
            hybrid.gca_engine.load_state_dict(checkpoint['gca_engine'])
            print("Weights loaded successfully.")
        else:
            print("Warning: No trained weights found. Using initialized weights.")
            
        hybrid.hook_attention()
        
    except Exception as e:
        print(f"Initialization Failed: {e}")
        return

    print("\n--- Model Ready ---")
    initial_mem = get_memory_usage()
    print(f"System Memory (RSS): {format_bytes(initial_mem)}")
    
    # Estimate G-Stream Memory
    # PSI state: 64 heads * 32 floats * 4 bytes
    g_stream_size = 64 * 32 * 4 
    print(f"G-Stream Manifold Size (Fixed): {g_stream_size} bytes (Constant O(1))")
    
    history = []
    
    while True:
        try:
            user_input = input("\nUser> ")
            if user_input.lower() in ['exit', 'quit']:
                break
            if user_input.lower() == 'reset':
                hybrid.state = None # Reset rotor
                history = []
                print("Geometric Context Reset.")
                continue
            
            if not user_input.strip():
                continue
                
            # Full prompt with simple template
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            print("Thinking...", end="", flush=True)
            
            # Generate
            response = hybrid.generate_with_geometry(prompt, max_new_tokens=100)
            
            # Extract just the new text if possible (tokenizer decode usually gives full)
            # Simple heuristic split for Llama 3 prompt format
            if "<|start_header_id|>assistant<|end_header_id|>" in response:
                response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            
            print(f"\rGeo-Llama> {response}")
            
            # Metrics
            current_mem = get_memory_usage()
            
            # Calculate PSI metrics
            psi_norm = torch.norm(hybrid.state.psi).item() if hybrid.state else 0
            
            print(f"\n[Metrics] Memory: {format_bytes(current_mem)} | PSI Norm: {psi_norm:.4f}")
            print(f"[G-Stream] Context efficiently encoded in {g_stream_size} bytes.")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"\nError: {e}")

if __name__ == "__main__":
    chat_interface()
