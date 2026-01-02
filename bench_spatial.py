import torch
import numpy as np
from geo_llama.core.cga import batch_point, batch_sphere, batch_plane, compute_cga_distance
from geo_llama.core.model import GeoLlamaHybrid

class MockLlama(torch.nn.Module):
    def __init__(self, d_model=2048):
        super().__init__()
        self.d_model = d_model
        self.embed = torch.nn.Embedding(1000, d_model)
    def get_input_embeddings(self):
        return self.embed
    def forward(self, **kwargs):
        from types import SimpleNamespace
        return SimpleNamespace(logits=torch.randn(1, 10, 1000), hidden_states=[torch.randn(1, 10, 2048)])

def encode_spatial_scene(scene):
    cga_objects = []
    for obj in scene:
        if obj['type'] == 'sphere':
            cga_objects.append(batch_sphere(obj['x'], obj['y'], obj['z'], obj['r']))
    return np.concatenate(cga_objects, axis=0)

def benchmark_spatial_intelligence():
    print("--- Geo-Llama Spatial Reasoning Benchmark ---")
    
    # 1. Setup
    mock_base = MockLlama()
    hybrid = GeoLlamaHybrid(mock_base, d_model=2048, num_heads=64)
    
    # 2. Define a Scene: A Sphere at (0, 0, 0) with radius 5
    scene = [
        {'type': 'sphere', 'x': 0.0, 'y': 0.0, 'z': 0.0, 'r': 5.0}
    ]
    sphere_cga = encode_spatial_scene(scene)
    
    # 3. Test Points
    # Point A: (1, 1, 1) - Inside
    # Point B: (10, 10, 10) - Outside
    point_a = batch_point(np.array([1.0]), np.array([1.0]), np.array([1.0]))
    point_b = batch_point(np.array([10.0]), np.array([10.0]), np.array([10.0]))
    
    # 4. Logical Check via Inner Product
    # In CGA, for a sphere S and point P:
    # <P, S> > 0 -> Point is INSIDE
    # <P, S> < 0 -> Point is OUTSIDE
    # (Note: This depends on the normalization of ninf and no, 
    #  but we'll check the sign relative to distance).
    
    # Convert to torch for computation
    point_a = torch.from_numpy(point_a)
    point_b = torch.from_numpy(point_b)
    sphere_cga = torch.from_numpy(sphere_cga)
    
    dist_a = compute_cga_distance(point_a, sphere_cga)
    dist_b = compute_cga_distance(point_b, sphere_cga)
    
    print(f"\nSphere Center: (0,0,0), Radius: 5.0")
    print(f"Point A (1,1,1) -> Conformal Metric Score: {dist_a[0]:.4f}")
    print(f"Point B (10,10,10) -> Conformal Metric Score: {dist_b[0]:.4f}")
    
    # Interpretation:
    # d^2 - r^2 < 0 -> INSIDE
    # d^2 - r^2 > 0 -> OUTSIDE
    if dist_a[0] < 0:
        print("Result: Point A recognized as INTERNAL (Correct).")
    else:
        print("Result: Point A recognized as EXTERNAL.")
        
    if dist_b[0] > 0:
        print("Result: Point B recognized as EXTERNAL (Correct).")
    else:
        print("Result: Point B recognized as INTERNAL.")

if __name__ == "__main__":
    benchmark_spatial_intelligence()
