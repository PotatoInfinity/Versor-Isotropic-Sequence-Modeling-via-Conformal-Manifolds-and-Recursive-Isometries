import torch
import torch.optim as optim
from geo_llama.hybrid import GeoLlamaHybrid
from geo_llama.core.loss import GeometricConsistencyLoss
import argparse
import time

def generate_structural_data():
    """
    Generates a synthetic dataset focused on logical hierarchies.
    This helps the model learn that concepts follow geometric containment.
    """
    data = [
        "A circle is a subset of a plane. A sphere is a subset of 3D space. A circle is a projection of a sphere.",
        "Mammals include dogs, cats, and humans. A dog is a mammal. An animal is a category that includes mammals.",
        "The King lives in the Castle. The Castle is in the Kingdom. Therefore, the King is in the Kingdom.",
        "Red is a color. Crimson is a type of red. All crimson objects are red objects.",
        "Berlin is a city in Germany. Germany is a country in Europe. Berlin is in Europe."
    ]
    return data

def train_geo_llama(is_mock=True):
    print(f"--- Geo-Llama Manifold Refinement (Mode: {'MOCK' if is_mock else 'REAL'}) ---")
    
    # 1. Initialize Hybrid Model
    hybrid = GeoLlamaHybrid(device="cpu", is_mock=is_mock)
    
    # We train the LiftingLayer and the GCA parameters
    # The Llama weights (if real) are frozen to preserve semantic knowledge
    trainable_params = list(hybrid.gca_engine.parameters()) + list(hybrid.lifter.parameters())
    
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    criterion = GeometricConsistencyLoss(hybrid.gca_engine)
    
    # 2. Prepare Data
    corpus = generate_structural_data()
    
    print("\nStarting Structural Distillation...")
    for epoch in range(3):
        total_loss = 0
        for i, text in enumerate(corpus):
            optimizer.zero_grad()
            
            # 1. Forward pass to get hidden states
            if is_mock:
                # Mock hidden states: (batch, seq, d_model)
                hidden_states = torch.randn(1, 10, hybrid.d_model)
            else:
                # Real forward pass logic
                inputs = hybrid.tokenizer(text, return_tensors="pt")
                with torch.no_grad():
                    outputs = hybrid.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]

            # 2. Differentiable Lifting to Geometry
            # We now use the lifter directly as a module to maintain the grad_fn
            lifted_tensor = hybrid.lifter(hidden_states)
            
            # 3. Compute Loss
            # We want to minimize "Geometric Entropy" (Geometric Soup)
            loss = criterion(lifted_tensor)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} | Total Structural Loss: {total_loss:.6f}")

    print("\nTraining Complete. Lifting Layer is now 'Geometry-Aware'.")
    
    # Save the trained weights
    save_path = "geo_llama_weights.pt"
    torch.save({
        'lifter': hybrid.lifter.state_dict(),
        'gca_engine': hybrid.gca_engine.state_dict()
    }, save_path)
    print(f"Geometric weights saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run in mock mode")
    args = parser.parse_args()
    
    train_geo_llama(is_mock=args.mock)
