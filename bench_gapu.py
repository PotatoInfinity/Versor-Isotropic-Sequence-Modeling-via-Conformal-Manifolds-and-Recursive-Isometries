import numpy as np
import time
import geo_llama_core
from geo_llama.core.cga import batch_geometric_product as python_gp

def benchmark_rust_vs_python():
    print("--- Geo-Llama GAPU Acceleration Benchmark ---")
    
    # Create batch of 1000 multivectors
    n = 1000
    A = np.random.randn(n, 32).astype(np.float32)
    B = np.random.randn(n, 32).astype(np.float32)
    
    # 1. Python Implementation
    start = time.time()
    res_py = python_gp(A, B)
    py_time = time.time() - start
    print(f"Python GP Time ({n} ops): {py_time:.4f}s")
    
    # 2. Rust Implementation
    start = time.time()
    res_rust = geo_llama_core.rust_batch_geometric_product(A, B)
    rust_time = time.time() - start
    print(f"Rust GAPU Time ({n} ops): {rust_time:.4f}s")
    
    # 3. Accuracy Check
    diff = np.abs(res_py - res_rust).max()
    print(f"Numerical Difference: {diff:.2e}")
    
    speedup = py_time / rust_time
    print(f"\nGAPU Speedup: {speedup:.1f}x faster than Python")

if __name__ == "__main__":
    benchmark_rust_vs_python()
