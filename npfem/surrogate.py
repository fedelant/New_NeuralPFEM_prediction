"""
Deep learning surrogate for 3DCP simulation.
"""

# === Dependencies ===
import numpy as np
import torch
import torch.nn as nn

from npfem import encoder_processor_decoder
from npfem import global_variables as gv
from npfem import mesh


# === Constants ===
# Ratio beyond which mesh is considered
# considered sretched enough to add new nodes.
_ADD_NODES_CONST = 0.17

class Surrogate(nn.Module):
    """Encoder-processor-decoder surrogate model for the printing simulation."""

    def __init__(
        self,
        n_in_features: int,
        latent_dim: int,
        n_mlp_layers: int,
        n_attn_heads: int,
        n_attn_layers: int,
        attn_dropout: float,
    ):
        super().__init__()
        self.mesher = mesh.MeshGenerator()

        self._encode = encoder_processor_decoder.Encoder(
            n_in_features=n_in_features,
            n_out_features=latent_dim,
            nmlp_layers=n_mlp_layers,
            mlp_hidden_dim=latent_dim,
        ).to(gv.device)

        self._process = encoder_processor_decoder.Processor(
            n_in_features=latent_dim,
            mlp_hidden_dim=latent_dim,
            nhead=n_attn_heads,
            nlayers=n_attn_layers,
            dropout=attn_dropout,
        ).to(gv.device)

        self._decode = encoder_processor_decoder.Decoder(
            n_in_features=latent_dim,
            nmlp_layers=n_mlp_layers,
            mlp_hidden_dim=latent_dim,
            output_dim_vel=3,
        ).to(gv.device)

    # ------------------------------------------------------------------
    # Top-level update dispatch
    # ------------------------------------------------------------------
    def learned_update(self):
        """Advance the simulation by one step."""
        if gv.phase1:
            self.free_fall_update()
        else:
            self.deep_learning_update()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _boundary_nodes(cells: np.ndarray, n_nozzle: int):
        """
        Return the nodes at the nozzle boundary, split into 
        (nozzle-side, non-nozzle-side).
        """
        mask = np.any(cells < n_nozzle, axis=1)
        nodes = np.unique(cells[mask])
        nozzle_side = nodes[nodes < n_nozzle]
        deposition_side = nodes[nodes >= n_nozzle]
        return nozzle_side, deposition_side
 
    @staticmethod
    def _add_nodes_check(position: torch.Tensor, nozzle_side, other_side):
        """
        Decide whether the gap between the nozzle-side and the rest of the
        material is large enough to add a new row of nodes. 
        Returns (add_check, mean_z_nozzle, mean_z_other).
        """
        mean_z_nozzle = position[nozzle_side, 2].mean()
        mean_z_other = position[other_side, 2].mean()
        min_x = position[nozzle_side, 0].min()
        max_x = position[nozzle_side, 0].max()
        gap = (mean_z_nozzle - mean_z_other).abs()
        add_check = gap >= _ADD_NODES_CONST * (max_x - min_x)
        return add_check, mean_z_nozzle, mean_z_other
 
    # ------------------------------------------------------------------
    # Phase 1: analytic free-fall update
    # ------------------------------------------------------------------
    def free_fall_update(self):
        tp = gv.toolpath
 
        # Stop phase 1 once any non-nozzle node has reached the floor.
        non_nozzle_z = gv.position[gv.n_nozzle_nodes:, 2]
        if non_nozzle_z.numel() > 0 and (non_nozzle_z <= 0).any():
            gv.phase1 = False
            # Node state initialization: 1 = inside the sphere --> active, 0 = outside the sphere --> frozen
            gv.active = torch.zeros(gv.position.shape[0], dtype=torch.long, device=gv.device)
            return
 
        nozzle_vel_world = torch.tensor(
            [tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity],
            dtype=torch.float32, device=gv.device,
        )
 
        self._add_nodes_free_fall(nozzle_vel_world)
        self._remesh_global()
 
        next_pos = gv.position + nozzle_vel_world * gv.cfg.dt
        next_pos[next_pos[:, 2] < 0, 2] = 0.0
        next_pos[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
 
        # Velocity fades linearly from the floor (0) up to the nozzle height.
        z_top = gv.position[0, 2].clamp(min=1e-6)
        alpha = (next_pos[:, 2].clamp(min=0.0) / z_top).clamp(0.0, 1.0)
        velocity = alpha[:, None] * nozzle_vel_world[None, :]
        velocity[gv.nozzle_ids] = nozzle_vel_world
 
        gv.prev_velocities = velocity[:, None, :].expand(-1, gv.prev_velocities.shape[1], -1) \
            .clone().to(gv.prev_velocities.dtype)
        gv.position = next_pos
        gv.velocity = velocity
        gv.pressure = torch.zeros(gv.position.shape[0], device=gv.device, dtype=torch.float16)
        gv.was_z_transition = tp.is_z_transition
 
    def _add_nodes_free_fall(self, nozzle_vel_world: torch.Tensor):
        """Add a new nodes between nozzle and material if the gap is too large."""
        nozzle_cells = gv.cells[np.any(gv.cells < gv.n_nozzle_nodes, axis=1)]
        nozzle_side, deposition_side = self._boundary_nodes(nozzle_cells, gv.n_nozzle_nodes)
        if deposition_side.size == 0:
            return
 
        add_check, mean_z_nozzle, mean_z_other = self._add_nodes_check(
            gv.position, nozzle_side, deposition_side
        )
        if not add_check:
            return
 
        new_nodes = gv.position[nozzle_side].clone()
        new_nodes[:, 2] = 0.5 * (mean_z_nozzle + mean_z_other)
        n_new = new_nodes.shape[0]
 
        new_tags = np.arange(np.max(gv.tags) + 1, np.max(gv.tags) + 1 + n_new)
        new_velocities = nozzle_vel_world[None, None, :].expand(
            n_new, gv.prev_velocities.shape[1], -1
        ).clone()
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device)
 
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.tags = np.concatenate([gv.tags, new_tags])
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
 
    def _remesh_global(self):
        """Remesh the full point cloud (used in free-falling phase)."""
        tags = np.arange(gv.position.shape[0])
        cells, fs_tags = self.mesher.generate_mesh(
            gv.position
        )
        gv.cells = cells.cpu().numpy()
        gv.free_surf = np.isin(tags, fs_tags.cpu().numpy()).astype(np.int32)
        gv.tags = tags
 
    # ------------------------------------------------------------------
    # Feature preparation (shared by phase 2 forward pass)
    # ------------------------------------------------------------------
    def preprocessor(
        self,
        most_recent_position: torch.Tensor,
        velocity_sequence: torch.Tensor,
        pressure_sequence: torch.Tensor,
        z_floor: torch.Tensor = None,
    ):
        """Build per-node and global feature tensors for the encoder."""
        n_particles = most_recent_position.shape[0]
 
        scaled_vel = (velocity_sequence - gv.vel_mean) / gv.vel_std
        scaled_press = (pressure_sequence - gv.press_mean) / gv.press_std
        velpress_hist = torch.cat([scaled_vel, scaled_press.unsqueeze(-1)], dim=2)
 
        node_features = torch.cat(
            [
                velpress_hist.view(n_particles, -1),
                ((most_recent_position[:, 2].unsqueeze(-1) - z_floor) / 0.005).clamp(-1, 1),
            ],
            dim=-1,
        )
 
        tau0_star = gv.yield_stress / (gv.density * 9.81 * 2 * gv.nozzle_radius)
        p1 = gv.viscosity / 100
        global_features = torch.tensor([tau0_star, p1], device=gv.device)
 
        return node_features, global_features
 
    # ------------------------------------------------------------------
    # Phase 2: deep learning update
    # ------------------------------------------------------------------
    def deep_learning_update(self):
        tp = gv.toolpath
 
        new_node_indices = self.check_and_insert_nodes()
        nozzle_center = gv.position[gv.nozzle_ids].mean(dim=0)
 
        self._update_active_region(nozzle_center, radius=3 * gv.nozzle_radius)
        self._local_remesh(new_node_indices)
 
        spatial_mask = gv.active == 1
        self._forward_predict(spatial_mask, nozzle_center, tp)
 
    # -- adding new nodes ---------------------------------------------------
    def check_and_insert_nodes(self):
        """
        If the gap between the nozzle nodes and the
        rest of the material is too large add new nodes to ensure continous material flow,
        interpolating the state of the new nodes from the surrounding mesh.
 
        Returns `new_node_indices`, the indices of new nodes
        (empty if the gap wasn't large enough to add).
        """
        prev_cells = gv.cells
        mask_contains_eucl = np.any(prev_cells < gv.n_nozzle_nodes, axis=1)
        relevant_cells = prev_cells[mask_contains_eucl]
        connected_eucl, connected_non = self._boundary_nodes(relevant_cells, gv.n_nozzle_nodes)
 
        add_check, mean_z_eucl, mean_z_non = self._add_nodes_check(
            gv.position, connected_eucl, connected_non
        )
 
        if not add_check:
            return torch.zeros(0, dtype=torch.long, device=gv.device)
 
        return self.interpolate_on_new_nodes(
            relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non
        )
 
    def interpolate_on_new_nodes(self, relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non):
        """Add new nodes at the midplane and interpolate velocity and pressure."""
        new_z = 0.5 * (mean_z_eucl + mean_z_non)
        new_nodes = gv.position[connected_eucl].clone()
        new_nodes[:, 2] = new_z
        n_new = new_nodes.shape[0]
        new_tags = np.arange(np.max(gv.tags) + 1, np.max(gv.tags) + 1 + n_new)
 
        cells_old = torch.from_numpy(relevant_cells).long().to(gv.device)
        verts = gv.position[cells_old]
        x0, x1, x2, x3 = verts[:, 0], verts[:, 1], verts[:, 2], verts[:, 3]
        T = torch.stack([x1 - x0, x2 - x0, x3 - x0], dim=-1)
        valid_tets = torch.abs(torch.linalg.det(T)) > 1e-12
        T_valid = T[valid_tets]
        cells_valid = cells_old[valid_tets]
        x0_valid = x0[valid_tets]
        T_inv = torch.linalg.inv(T_valid)
 
        diff = new_nodes[:, None, :] - x0_valid[None, :, :]
        lambdas = torch.einsum("cij,ncj->nci", T_inv, diff)
        N1, N2, N3 = lambdas[..., 0], lambdas[..., 1], lambdas[..., 2]
        bary = torch.stack([1.0 - N1 - N2 - N3, N1, N2, N3], dim=-1)
 
        inside = (bary >= -1e-8).all(dim=-1)
        has_tet = inside.any(dim=1)
        tet_ids = torch.zeros(n_new, dtype=torch.long, device=gv.device)
        tet_ids[has_tet] = inside[has_tet].float().argmax(dim=1)
 
        n_steps = gv.prev_velocities.shape[1]
        new_velocities = torch.zeros(n_new, n_steps, gv.prev_velocities.shape[2], device=gv.device) \
            + torch.tensor([gv.printing_velocity, 0, -gv.flow_velocity], device=gv.device)
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device)
 
        if has_tet.any():
            rows = torch.arange(n_new, device=gv.device)[has_tet]
            bary_sel = bary[rows, tet_ids[has_tet]]
            tet_vels = gv.prev_velocities[cells_valid[tet_ids[has_tet]]]
            tet_press = gv.prev_pressures[cells_valid[tet_ids[has_tet]]]
            new_velocities[has_tet] = (bary_sel[:, :, None, None] * tet_vels).sum(dim=1)
            new_pressures[has_tet] = (bary_sel[:, :, None] * tet_press).sum(dim=1)
 
        if (~has_tet).any():
            connected_nodes_t = torch.from_numpy(np.union1d(connected_eucl, connected_non)).long().to(gv.device)
            nearest = torch.cdist(new_nodes[~has_tet], gv.position[connected_nodes_t]).argmin(dim=1)
            new_velocities[~has_tet] = gv.prev_velocities[connected_nodes_t[nearest]].to(new_velocities.dtype)
            new_pressures[~has_tet] = gv.prev_pressures[connected_nodes_t[nearest]].to(new_pressures.dtype)
 
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.tags = np.concatenate([gv.tags, new_tags])
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
        gv.active = torch.cat([gv.active, torch.zeros(n_new, dtype=torch.int32, device=gv.device)])
 
        return torch.arange(gv.position.shape[0] - n_new, gv.position.shape[0], dtype=torch.long, device=gv.device)
 
    # -- node state (active/frozen regions) --------------------------------
    @staticmethod
    def _update_active_region(nozzle_center: torch.Tensor, radius: torch.Tensor):
        """ Update nodes statebetween OUTSIDE -> INSIDE -> EXITED based on distance to the nozzle."""
        dists = torch.norm(gv.position - nozzle_center, dim=1)
        physically_inside = dists <= radius

        # Node state update 
        just_entered = (gv.active == 0) & physically_inside
        gv.active[just_entered] = 1

        just_exited = (gv.active == 1) & ~physically_inside
        gv.active[just_exited] = 0
 
    # -- local remeshing ---------------------------------------------------
    def _local_remesh(self, new_node_indices: torch.Tensor):
        """Remesh only the active region around the nozzle."""
        active = (gv.active == 1).detach().cpu().numpy()
        cells = gv.cells
        cell_active = active[cells]

        any_active = cell_active.any(axis=1)
        all_active = cell_active.all(axis=1)

        kept_cells = cells[~any_active]
        active_nodes = np.unique(cells[all_active])
        inactive_nodes = np.unique(cells[~all_active])

        kept_nodes = np.unique(kept_cells)
        discarded_nodes = np.unique(cells[any_active])
        shared_nodes = np.intersect1d(kept_nodes, discarded_nodes, assume_unique=True)
        active_boundary_nodes = inactive_nodes[active[inactive_nodes]]
        interface_nodes = np.unique(np.concatenate((shared_nodes, active_boundary_nodes)))

        nodes_to_remesh = np.concatenate((active_nodes, new_node_indices.detach().cpu().numpy()))
        if nodes_to_remesh.size == 0:
            return

        new_cells_local, _ = self.mesher.generate_mesh(gv.position[nodes_to_remesh], apply_node_rules=True)
        new_cells = nodes_to_remesh[new_cells_local.cpu().numpy()]

        if interface_nodes.size > 5:
            new_cells = new_cells[~np.isin(new_cells, active_boundary_nodes).all(axis=1)]

            interface_cells_local, _ = self.mesher.generate_mesh(gv.position[interface_nodes], apply_node_rules=False)
            interface_cells_local = interface_cells_local.cpu().numpy()
            is_old = np.isin(interface_nodes, shared_nodes)
            cell_old = is_old[interface_cells_local]
            mixed = cell_old.any(axis=1) & ~cell_old.all(axis=1)
            interface_cells = interface_nodes[interface_cells_local[mixed]]
        else:
            interface_cells = np.empty((0, 4), dtype=np.int64)

        gv.cells = np.concatenate((new_cells, interface_cells, kept_cells), axis=0)
        gv.free_surf = np.zeros(gv.position.shape[0], dtype=np.int32)



    # -- forward pass & state write-back ------------------------------------
    def _forward_predict(self, spatial_mask: torch.Tensor, nozzle_center: torch.Tensor, tp):
        """Run encode/process/decode on the active window and write the new state to gv."""
        scaling_pos = torch.tensor([3 * 0.0125, 3 * 0.0125, 0.02], device=gv.device)
        scaled_position = (gv.position - nozzle_center) / scaling_pos
 
        node_features, global_features = self.preprocessor(
            most_recent_position=gv.position[spatial_mask],
            velocity_sequence=gv.prev_velocities[spatial_mask],
            pressure_sequence=gv.prev_pressures[spatial_mask],
            z_floor=gv.z_floor,
        )
 
        node_latent = self._encode(node_features, global_features)
        node_latent = self._process(
            node_latent, torch.zeros_like(scaled_position[spatial_mask, 0]), scaled_position[spatial_mask]
        )
        norm_pred_vel, norm_pred_pos, norm_pred_press = self._decode(node_latent, global_features)
 
        velocity = torch.zeros_like(gv.position)
        pressure = torch.zeros(gv.position.shape[0], device=gv.device, dtype=torch.float16)
        next_pos = gv.position.clone()
 
        pred_vel_world = (norm_pred_vel * gv.vel_std + gv.vel_mean).to(torch.float32)
 
        velocity[spatial_mask] = pred_vel_world
        next_pos[spatial_mask] = gv.position[spatial_mask] + velocity[spatial_mask] * gv.cfg.dt
        pressure[spatial_mask] = norm_pred_press.squeeze(-1) * gv.press_std + gv.press_mean
 
        # Floor boundary condition.
        bd_mask = next_pos[:, -1] <= 0
        velocity[bd_mask] = 0.0
        next_pos[bd_mask, -1] = 0
 
        # Nozzle nodes follow the prescribed toolpath exactly.
        nozzle_vel_world = torch.tensor(
            [tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity],
            dtype=torch.float32, device=gv.device,
        )
        velocity[gv.nozzle_ids] = nozzle_vel_world
        next_pos[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
 
        gv.position = next_pos
        gv.velocity = velocity
        gv.pressure = pressure