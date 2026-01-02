
import sys
import os

# Add the current directory to path so we can import geo_llama
sys.path.append("/Users/mac/Desktop/Geo-llama/Stable")

from geo_llama.hybrid import GeoLlamaHybrid

def test_mock():
    print("Initializing Mock Geo-Llama...")
    try:
        hybrid = GeoLlamaHybrid(device="cpu", is_mock=True)
        print("Model Initialized.")
        
        # Test Hook
        hybrid.hook_attention()
        print("Attention Hooked.")
        
        # Test Generation Loop logic
        prompt = "Define a sphere."
        print(f"Testing generation with prompt: {prompt}")
        response = hybrid.generate_with_geometry(prompt, max_new_tokens=20)
        print(f"Response: {response}")
        
        # Test Metrics
        psi_norm = hybrid.state.psi.norm()
        print(f"PSI Norm: {psi_norm}")
        
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_mock()
