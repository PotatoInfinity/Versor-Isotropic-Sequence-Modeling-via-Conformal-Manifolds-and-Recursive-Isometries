import numpy as np
from .core.cga import batch_point, batch_sphere, batch_plane, exp_map

class GeoMultimodalBridge:
    """
    Bridges non-linguistic spatial data (3D point clouds, sensor data)
    into the Geo-Llama G-Stream manifold.
    """
    def __init__(self, hybrid):
        self.hybrid = hybrid

    def inject_points(self, points_3d):
        """
        Injects 3D coordinates as Conformal Points directly into the state.
        points_3d: (N, 3) numpy array
        """
        x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
        cga_points = batch_point(x, y, z)
        
        # We treat points as 'Identity' thoughts for the rotor update,
        # or we transform them into rotors if they represent movements.
        # For pure injection, we simply 'Observe' them.
        
        # In a full multimodal setup, we would use these points to 
        # condition the context PSI.
        print(f"Injected {len(points_3d)} spatial points into the G-Stream.")
        return cga_points

    def encode_spatial_scene(self, objects):
        """
        objects: list of dicts with type ('sphere', 'plane') and params.
        """
        cga_objects = []
        for obj in objects:
            if obj['type'] == 'sphere':
                cga_objects.append(batch_sphere(
                    obj['x'], obj['y'], obj['z'], obj['r']
                ))
            elif obj['type'] == 'plane':
                cga_objects.append(batch_plane(
                    obj['nx'], obj['ny'], obj['nz'], obj['d']
                ))
        
        return np.concatenate(cga_objects, axis=0) if cga_objects else None

    def query_spatial_knowledge(self, query_point_3d):
        """
        Checks the current G-Stream context against a new spatial 3D point.
        """
        # 1. Lift point to CGA
        p = batch_point(
            np.array([query_point_3d[0]]), 
            np.array([query_point_3d[1]]), 
            np.array([query_point_3d[2]])
        )
        
        # 2. Check against PSI
        # We can treat the context PSI as a generalized geometric 'Volume'
        from .core.cga import batch_inner_product
        consistency = batch_inner_product(self.hybrid.state.psi[0], p[0])
        
        return float(consistency)
