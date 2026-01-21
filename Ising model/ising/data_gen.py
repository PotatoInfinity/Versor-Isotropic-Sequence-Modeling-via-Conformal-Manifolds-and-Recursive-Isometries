import numpy as np
import torch
import os

def generate_ising_2d(size, temp, steps=1000):
    """
    Checkerboard Metropolis Algorithm (Vectorized).
    Updates half the grid in parallel for extreme speedup.
    Steps here refer to 'Full Grid Sweeps'.
    """
    grid = torch.ones((size, size), dtype=torch.float32)
    # Start with a random configuration if temp is high
    if temp > 1.5:
        grid = torch.where(torch.rand((size, size)) > 0.5, 1.0, -1.0)
    
    # Create masks for checkerboard
    indices = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij'))
    mask_even = (indices.sum(dim=0) % 2 == 0)
    mask_odd = ~mask_even
    
    beta = 1.0 / temp
    
    for _ in range(steps):
        for mask in [mask_even, mask_odd]:
            # Roll to find neighbors
            n = torch.roll(grid, shifts=1, dims=0) + torch.roll(grid, shifts=-1, dims=0) + \
                torch.roll(grid, shifts=1, dims=1) + torch.roll(grid, shifts=-1, dims=1)
            
            dE = 2 * grid * n
            # Metropolis probability
            prob = torch.exp(-dE * beta)
            accept = (torch.rand((size, size)) < prob) | (dE <= 0)
            
            # Apply update only to the active mask
            update = torch.where(mask & accept, -grid, grid)
            grid = update
            
    return grid.numpy()

def generate_dataset(n_samples_per_class, size=8):
    """
    Generates a dataset of Ising configurations for three distinct physical phases.
    """
    temps = [1.0, 2.269, 5.0]
    data = []
    labels = []
    
    for label, T in enumerate(temps):
        print(f"Generating Phase {label} (T={T})...")
        for i in range(n_samples_per_class):
            if i % 50 == 0:
                print(f"  {i}/{n_samples_per_class}...")
            grid = generate_ising_2d(size, T)
            data.append(grid)
            labels.append(label)
            
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    return data, labels

def generate_sweep_dataset(samples_per_temp, size=8):
    """
    Generates data across a continuous temperature range [1.0, 4.0].
    """
    temps = np.linspace(1.0, 4.0, 20)
    data = []
    labels = [] # We still use 3 classes for labeling based on T
    
    for T in temps:
        print(f"Generating Sweep T={T:.3f}...")
        for _ in range(samples_per_temp):
            grid = generate_ising_2d(size, T, steps=500) # Full sweeps
            data.append(grid)
            # Labeling for reference
            if T < 2.1: label = 0
            elif T > 2.5: label = 2
            else: label = 1
            labels.append(label)
    return np.array(data, dtype=np.float32), np.array(labels, dtype=np.int64), temps

if __name__ == "__main__":
    # Settings for a publishable-grade benchmark
    SAMPLES_PER_CLASS = 750 # Total 2250 samples (Deep Generalization Challenge)
    GRID_SIZE = 32  # 32x32 grid
    DATA_DIR = "/Users/mac/Desktop/Versor/Research/data"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"--- Ising Data Generation (Size={GRID_SIZE}x{GRID_SIZE}) ---")
    data, labels = generate_dataset(n_samples_per_class=SAMPLES_PER_CLASS, size=GRID_SIZE)
    
    # Generate Sweep Data
    sweep_data, sweep_labels, sweep_temps = generate_sweep_dataset(samples_per_temp=15, size=GRID_SIZE)
    
    data_path = os.path.join(DATA_DIR, "ising_data.pt")
    torch.save({
        'data': torch.from_numpy(data),
        'labels': torch.from_numpy(labels),
        'sweep_data': torch.from_numpy(sweep_data),
        'sweep_labels': torch.from_numpy(sweep_labels),
        'sweep_temps': torch.from_numpy(sweep_temps),
        'metadata': {
            'size': GRID_SIZE,
            'temps': [1.0, 2.269, 5.0],
            'steps_per_sample': 10000
        }
    }, data_path)
    
    print(f"Successfully generated {len(data)} samples.")
    print(f"Saved to: {data_path}")



