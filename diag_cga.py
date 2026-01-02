import numpy as np
from geo_llama.core.cga import batch_inner_product, batch_point, batch_sphere

def diagnostic():
    # Basis
    e4 = np.zeros(32)
    e4[8] = 1.0 # EP_IDX
    e5 = np.zeros(32)
    e5[16] = 1.0 # EM_IDX
    
    print(f"<e4, e4>: {batch_inner_product(e4, e4)}")
    print(f"<e5, e5>: {batch_inner_product(e5, e5)}")
    
    no = 0.5 * (e5 - e4)
    ninf = e5 + e4
    
    print(f"<no, no>: {batch_inner_product(no, no)}")
    print(f"<ninf, ninf>: {batch_inner_product(ninf, ninf)}")
    print(f"<no, ninf>: {batch_inner_product(no, ninf)}")
    
    # Point at origin
    p0 = batch_point(np.array([0.0]), np.array([0.0]), np.array([0.0]))[0]
    print(f"Point(0,0,0) lane 4: {p0[4]}, lane 5: {p0[5]}")
    
    # Sphere at origin radius 5
    s5 = batch_sphere(np.array([0.0]), np.array([0.0]), np.array([0.0]), np.array([5.0]))[0]
    print(f"Sphere(0,0,0,5) lane 4: {s5[4]}, lane 5: {s5[5]}")
    
    # Inner product <P0, S5>
    ip = batch_inner_product(p0, s5)
    print(f"<P0, S5>: {ip}")
    print(f"-2 * <P0, S5>: {-2. * ip}")

if __name__ == "__main__":
    diagnostic()
