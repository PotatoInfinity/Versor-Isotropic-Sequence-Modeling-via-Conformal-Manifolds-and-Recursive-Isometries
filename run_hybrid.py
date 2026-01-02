from geo_llama.hybrid import GeoLlamaHybrid
import sys

import argparse

def main():
    parser = argparse.ArgumentParser(description="Geo-Llama Hybrid Runner")
    parser.add_argument("--mock", action="store_true", help="Run in mock mode without loading Llama weights")
    args = parser.parse_args()

    print("--- Geo-Llama Hybrid Model Boot Sequence ---")
    
    # Initialize the Hybrid Model
    try:
        hybrid_model = GeoLlamaHybrid(model_id="meta-llama/Llama-3.2-1B-Instruct", device="cpu", is_mock=args.mock)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nMake sure you have requested access at: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        return

    # test prompt to verify dual-stream synchronization
    prompt = "Explain the relationship between a circle and a sphere in geometric terms."
    
    print(f"\nPrompt: {prompt}")
    print("Processing (G-Stream synchronization in progress)...")
    
    # This will trigger the download (first time) and then the geometric forward pass
    response = hybrid_model.generate_with_geometry(prompt, max_new_tokens=100)
    
    print("\n--- Model Output ---")
    print(response)
    print("\n--- G-Stream Status ---")
    # Check the norm of one of the geometric heads to verify structural activity
    norm = float(hybrid_model.state.psi[0, 0])
    print(f"Manifold 0 Scalar Norm: {norm:.4f} (Active)")

if __name__ == "__main__":
    main()
