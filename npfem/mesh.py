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
        valid = torch.abs(denom) > 1e-30
        R = torch.full((pts.shape[0],), float('inf'), device=pts.device, dtype=pts.dtype)

        if valid.any():
            x = num[valid] / denom[valid].unsqueeze(1)
            R[valid] = torch.linalg.norm(x, dim=1)

        return R

    def _boundary_faces_3d(self, simplices: torch.Tensor) -> torch.Tensor:
        """Robust boundary face extraction using native PyTorch 2D unique."""
        # 1. Stack all 4 faces of each tetrahedron
        faces = torch.cat([
            simplices[:, [0, 1, 2]],
            simplices[:, [0, 1, 3]],
            simplices[:, [0, 2, 3]],
            simplices[:, [1, 2, 3]]
        ], dim=0)
        
        # 2. Sort the node indices of each face so identical faces match perfectly
        faces, _ = torch.sort(faces, dim=1)
        
        # 3. Find unique faces and their counts using PyTorch native 2D unique
        unique_faces, counts = torch.unique(faces, dim=0, return_counts=True)
        
        # 4. A face is a boundary face if it is not shared by another tetrahedron (count == 1)
        boundary_faces = unique_faces[counts == 1]
        
        return boundary_faces

    def generate_initial_mesh(self, position: torch.Tensor, apply_node_rules: bool = True):
        pts = position.detach().cpu().numpy()

        # unique xy positions and unique z layers
        xy, inv = np.unique(pts[:, :2].round(6), axis=0, return_inverse=True)
        inv = inv.ravel()
        zs = np.unique(pts[:, 2].round(6))
        layer = np.searchsorted(zs, pts[:, 2].round(6))

        # idx[layer, xy_id] -> row index in pts
        idx = -np.ones((len(zs), len(xy)), dtype=int)
        idx[layer, inv] = np.arange(len(pts))
        assert (idx >= 0).all(), "every layer must contain every xy point"

        tri2d = Delaunay(xy).simplices

        tets = []
        for k in range(len(zs) - 1):
            for t in tri2d:
                a, b, c = np.sort(t)          # global ordering => conforming faces
                a0, b0, c0 = idx[k,     [a, b, c]]
                a1, b1, c1 = idx[k + 1, [a, b, c]]
                tets += [[a0, b0, c0, c1],
                        [a0, b0, b1, c1],
                        [a0, a1, b1, c1]]
        cells = torch.as_tensor(tets, dtype=torch.long, device=position.device)
        boundary_nodes = torch.unique(self._boundary_faces_3d(cells))
        return cells, boundary_nodes

    def generate_mesh(self, position: torch.Tensor, alpha: float = 400.0, return_boundary_nodes: bool = True):
        """3D Delaunay + alpha-shape filtering.

        Delaunay itself remains CPU/SciPy. When boundary nodes are not needed,
        skip the extra GPU boundary-face reduction.
        """
        device = position.device

        tri = Delaunay(position.detach().cpu().numpy(), qhull_options="Qt Qbb Qc")
        simplices = torch.as_tensor(tri.simplices, dtype=torch.long, device=device)

        radius = self._circumradius_3d(position[simplices])
        
        # Only apply the alpha filter here
        mask = radius <= (1.0 / alpha)
        cells = simplices[mask]

        if cells.numel() == 0:
            raise ValueError(f"No elements survive alpha filter.")

        if return_boundary_nodes:
            boundary_nodes = torch.unique(self._boundary_faces_3d(cells))
            return cells, boundary_nodes
        return cells

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