from scipy.spatial import Delaunay
import numpy as np
import torch
import vtk

from npfem import global_variables as gv

# Completely disables all C++ VTK warnings and error prints to the terminal
vtk.vtkObject.GlobalWarningDisplayOff()

class MeshGenerator:
    """
    A class for generating meshes using Delaunay triangulation and alpha-shape filtering.
    """

    def __init__(self):
        pass

    def _circumradius_3d(self, pts: torch.Tensor) -> torch.Tensor:
        """GPU-accelerated circumradius using PyTorch tensor math. pts: [M, 4, 3]"""
        A, B, C, D = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        u = B - A
        v = C - A
        w = D - A

        # PyTorch cross products
        v_x_w = torch.linalg.cross(v, w, dim=1)
        w_x_u = torch.linalg.cross(w, u, dim=1)
        u_x_v = torch.linalg.cross(u, v, dim=1)

        # Volume denominator
        denom = 2.0 * torch.sum(u * v_x_w, dim=1)

        # Squared norms
        u2 = torch.sum(u * u, dim=1).unsqueeze(1)
        v2 = torch.sum(v * v, dim=1).unsqueeze(1)
        w2 = torch.sum(w * w, dim=1).unsqueeze(1)

        num = u2 * v_x_w + v2 * w_x_u + w2 * u_x_v

        valid = torch.abs(denom) > 1e-14
        R = torch.full((pts.shape[0],), float('inf'), device=pts.device, dtype=pts.dtype)

        if valid.any():
            x = num[valid] / denom[valid].unsqueeze(1)
            R[valid] = torch.linalg.norm(x, dim=1)

        return R

    def _boundary_faces_3d(self, simplices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Ultra-fast GPU boundary face extraction using 64-bit tensor hashing."""
        # 1. Stack all faces
        faces = torch.cat([
            simplices[:, [0, 1, 2]],
            simplices[:, [0, 1, 3]],
            simplices[:, [0, 2, 3]],
            simplices[:, [1, 2, 3]]
        ], dim=0)
        
        # 2. Sort rows (torch.sort returns a tuple of values and indices, we just want values)
        faces, _ = torch.sort(faces, dim=1)
        faces = faces.to(torch.int64) # Prevent overflow!
        
        base = num_nodes + 1
        
        # 3. Hash into 1D
        hashed_faces = faces[:, 0] + faces[:, 1] * base + faces[:, 2] * (base ** 2)
        
        # 4. 1D Unique on GPU
        unique_hashes, counts = torch.unique(hashed_faces, return_counts=True)
        boundary_hashes = unique_hashes[counts == 1]
        
        # 5. Unhash back to 3D
        f0 = boundary_hashes % base
        f1 = (boundary_hashes // base) % base
        f2 = boundary_hashes // (base ** 2)
        
        return torch.stack((f0, f1, f2), dim=1)

    def generate_mesh(self, position: torch.Tensor, 
                      alpha: float = 150.0, 
                      apply_node_rules: bool = True):
        """3D Delaunay + alpha-shape filtering."""
        num_nodes = position.shape[0]
        device = position.device

        # 1. Delaunay
        tri = Delaunay(position.detach().cpu().numpy(), qhull_options="Qt Qbb Qc")
        simplices = torch.as_tensor(tri.simplices, dtype=torch.long, device=device)

        # 2. Alpha-shape filter
        radius = self._circumradius_3d(position[simplices])
        mask = radius <= (1.0 / alpha)

        # 3. Node rules
        if apply_node_rules:
            nozzle_nodes = torch.arange(num_nodes, device=device) < gv.n_nozzle_nodes
            new_nodes = torch.arange(num_nodes, device=device) >= num_nodes - gv.n_nozzle_nodes
            others = ~(nozzle_nodes | new_nodes)
            has_euclidean = nozzle_nodes[simplices].any(dim=1)
            internal_cells = others[simplices].any(dim=1)
            mask |= ~internal_cells
            mask &= ~(has_euclidean & internal_cells)

        cells = simplices[mask]

        if cells.numel() == 0:
            raise ValueError(f"No elements survive alpha filter (alpha={alpha}, threshold={1.0 / alpha:.4f}, min circumradius={radius.min().item():.4f}).")

        boundary_nodes = torch.unique(self._boundary_faces_3d(cells, num_nodes=num_nodes))
        return cells, boundary_nodes

    '''
    def calculate_shape_functions(self, elem_coords: torch.Tensor, point: torch.Tensor):
        """
        elem_coords: [K, D] (triangle or tetrahedron node coordinates)
        point: [D]   target point
        Returns: [K] barycentric coordinates (shape functions)
        """
        D = elem_coords.shape[1]
        K = elem_coords.shape[0]

        if D == 2 and K == 3:
            # --- Triangle (2D) ---
            x1, y1 = elem_coords[0]
            x2, y2 = elem_coords[1]
            x3, y3 = elem_coords[2]
            x, y   = point

            det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            N1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / det
            N2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / det
            N3 = 1.0 - N1 - N2
            return torch.tensor([N1, N2, N3], dtype=elem_coords.dtype, device=elem_coords.device)

        elif D == 3 and K == 4:
            # --- Tetrahedron (3D) ---
            v0, v1, v2, v3 = elem_coords

            def volume(a, b, c, d):
                return torch.abs(torch.det(torch.stack([b - a, c - a, d - a], dim=-1))) / 6.0

            vol_total = volume(v0, v1, v2, v3)
            N0 = volume(point, v1, v2, v3) / vol_total
            N1 = volume(v0, point, v2, v3) / vol_total
            N2 = volume(v0, v1, point, v3) / vol_total
            N3 = volume(v0, v1, v2, point) / vol_total
            return torch.stack([N0, N1, N2, N3])

        else:
            raise ValueError(f"Unsupported element shape with {K} nodes in {D}D")
    
    def find_elements(self, points: torch.Tensor, nodes: torch.Tensor, elements: torch.Tensor):
        """
        Vectorized element finder for 2D (triangles) and 3D (tetrahedra).

        Args:
            points:   [N, D] query points
            nodes:    [N_nodes, D] node positions
            elements: [M, K] element connectivity (K=3 tri, K=4 tet)

        Returns:
            elem_idx: [N] element index for each point (-1 if not inside any)
            bary:     [N, K] barycentric coords (valid only if inside)
        """
        device = points.device
        D = points.shape[1]
        K = elements.shape[1]
        assert (D == 2 and K == 3) or (D == 3 and K == 4), "Only supports tri(2D)/tet(3D)."

        # Gather element node coords [M, K, D]
        elem_coords = nodes[elements]  # [M, K, D]

        # Expand dims for broadcasting
        pts = points[:, None, :]       # [N, 1, D]
        elems = elem_coords[None, :, :, :]  # [1, M, K, D]

        if D == 2:
            # --- Triangles ---
            A = elems[:, :, 0, :]  # [1, M, 2]
            B = elems[:, :, 1, :]
            C = elems[:, :, 2, :]
            P = pts

            detT = (B[..., 1] - C[..., 1]) * (A[..., 0] - C[..., 0]) + \
                (C[..., 0] - B[..., 0]) * (A[..., 1] - C[..., 1])  # [1, M]

            l1 = ((B[..., 1] - C[..., 1]) * (P[..., 0] - C[..., 0]) + \
                (C[..., 0] - B[..., 0]) * (P[..., 1] - C[..., 1])) / detT
            l2 = ((C[..., 1] - A[..., 1]) * (P[..., 0] - C[..., 0]) + \
                (A[..., 0] - C[..., 0]) * (P[..., 1] - C[..., 1])) / detT
            l3 = 1.0 - l1 - l2

            bary = torch.stack([l1, l2, l3], dim=-1)  # [N, M, 3]

            inside = (bary >= -1e-8).all(dim=-1) & (bary <= 1+1e-8).all(dim=-1)  # [N, M]

        else:
            # --- Tetrahedra ---
            A = elems[:, :, 0, :]  # [1, M, 3]
            B = elems[:, :, 1, :]
            C = elems[:, :, 2, :]
            Dv = elems[:, :, 3, :]
            P = pts

            def tet_vol(a, b, c, d):
                return torch.sum(torch.cross(b - a, c - a, dim=-1) * (d - a), dim=-1)

            volT = tet_vol(A, B, C, Dv)  # [1, M]

            v1 = tet_vol(P, B, C, Dv) / volT
            v2 = tet_vol(A, P, C, Dv) / volT
            v3 = tet_vol(A, B, P, Dv) / volT
            v4 = tet_vol(A, B, C, P) / volT

            bary = torch.stack([v1, v2, v3, v4], dim=-1)  # [N, M, 4]

            inside = (bary >= -1e-8).all(dim=-1) & (bary <= 1+1e-8).all(dim=-1)  # [N, M]

        # Pick first valid element per point
        elem_idx = torch.full((points.shape[0],), -1, device=device, dtype=torch.long)
        bary_out = torch.zeros((points.shape[0], K), device=device, dtype=points.dtype)

        any_inside = inside.any(dim=1)  # [N]
        idx_inside = any_inside.nonzero(as_tuple=True)[0]

        if idx_inside.numel() > 0:
            first_hit = inside[idx_inside].float().argmax(dim=1)  # pick first element index
            elem_idx[idx_inside] = first_hit
            bary_out[idx_inside] = bary[idx_inside, first_hit, :]

        return elem_idx, bary_out
        '''