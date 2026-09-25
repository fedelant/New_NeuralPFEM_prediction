"""
Deep learning surrogate for 3DCP simulation.
"""
import numpy as np
import torch
import torch.nn as nn
from npfem import encoder_processor_decoder
from npfem import global_variables as gv
from npfem import mesh

_ADD_NODES_CONST = 0.17


class Surrogate(nn.Module):
    """Encoder-processor-decoder surrogate model for the printing simulation."""

    def __init__(self, n_in_features: int, latent_dim: int, n_mlp_layers: int, n_attn_heads: int, n_attn_layers: int, attn_dropout: float):
        super().__init__()
        self.mesher = mesh.MeshGenerator()
        self._encode = encoder_processor_decoder.Encoder(n_in_features=n_in_features, n_out_features=latent_dim, nmlp_layers=n_mlp_layers, mlp_hidden_dim=latent_dim).to(gv.device)
        self._process = encoder_processor_decoder.Processor(n_in_features=latent_dim, mlp_hidden_dim=latent_dim, nhead=n_attn_heads, nlayers=n_attn_layers, dropout=attn_dropout).to(gv.device)
        self._decode = encoder_processor_decoder.Decoder(n_in_features=latent_dim, nmlp_layers=n_mlp_layers, mlp_hidden_dim=latent_dim, output_dim_vel=3).to(gv.device)

    def learned_update(self):
        if gv.phase1:
            self.free_fall_update()
        else:
            self.deep_learning_update()

    # ------------------------------------------------------------------
    # Layer bookkeeping
    # ------------------------------------------------------------------

    def _ensure_node_layers(self):
        n_nodes = gv.position.shape[0]
        n_nozzle = gv.n_nozzle_nodes
        if not hasattr(gv, "node_layer"):
            gv.node_layer = torch.ones(n_nodes, dtype=torch.int64, device=gv.device)
            gv.node_layer[:n_nozzle] = 0
            return
        current_size = gv.node_layer.shape[0]
        if current_size == n_nodes:
            return
        if current_size > n_nodes:
            gv.node_layer = gv.node_layer[:n_nodes]
            return
        n_new = n_nodes - current_size
        new_layers = torch.ones(n_new, dtype=gv.node_layer.dtype, device=gv.device)
        gv.node_layer = torch.cat((gv.node_layer, new_layers), dim=0)

    @staticmethod
    def _new_node_layer():
        if getattr(gv, "second_layer_init_done", False):
            return 2
        return 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _boundary_nodes(cells: torch.Tensor, n_nozzle: int):
        mask = (cells < n_nozzle).any(dim=1)
        nodes = torch.unique(cells[mask])
        nozzle_side = nodes[nodes < n_nozzle]
        deposition_side = nodes[nodes >= n_nozzle]
        return nozzle_side, deposition_side

    @staticmethod
    def _add_nodes_check(position: torch.Tensor, nozzle_side, other_side):
        if nozzle_side.numel() == 0 or other_side.numel() == 0:
            return False, 0.0, 0.0
        mean_z_nozzle = position[nozzle_side, 2].mean()
        mean_z_other = position[other_side, 2].mean()
        min_x = position[nozzle_side, 0].min()
        max_x = position[nozzle_side, 0].max()
        gap = (mean_z_nozzle - mean_z_other).abs()
        add_check = gap >= _ADD_NODES_CONST * (max_x - min_x)
        return add_check, mean_z_nozzle, mean_z_other

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def free_fall_update(self):
        tp = gv.toolpath
        tp.advance_if_needed()
        self._ensure_node_layers()
        non_nozzle_z = gv.position[gv.n_nozzle_nodes:, 2]
        if non_nozzle_z.numel() > 0 and (non_nozzle_z <= 0).any():
            gv.phase1 = False
            gv.active = torch.zeros(gv.position.shape[0], dtype=torch.long, device=gv.device)
            return
        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity], dtype=torch.float32, device=gv.device)
        self._add_nodes_free_fall(nozzle_vel_world)
        self._remesh_global()
        next_pos = gv.position + nozzle_vel_world * gv.cfg.dt
        next_pos[next_pos[:, 2] < 0, 2] = 0.0
        next_pos[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
        z_top = gv.position[0, 2].clamp(min=1e-6)
        alpha = (next_pos[:, 2].clamp(min=0.0) / z_top).clamp(0.0, 1.0)
        velocity = alpha[:, None] * nozzle_vel_world[None, :]
        velocity[gv.nozzle_ids] = nozzle_vel_world
        gv.prev_velocities = velocity[:, None, :].expand(-1, gv.prev_velocities.shape[1], -1).clone().to(gv.prev_velocities.dtype)
        gv.position = next_pos
        gv.velocity = velocity
        gv.pressure = torch.zeros(gv.position.shape[0], device=gv.device, dtype=torch.float16)
        gv.was_z_transition = tp.is_z_transition

    def _add_nodes_free_fall(self, nozzle_vel_world: torch.Tensor):
        if gv.cells is None:
            return
        cells = gv.cells
        nozzle_cells = cells[(cells < gv.n_nozzle_nodes).any(dim=1)]
        nozzle_side, deposition_side = self._boundary_nodes(nozzle_cells, gv.n_nozzle_nodes)
        if deposition_side.numel() == 0:
            return
        add_check, mean_z_nozzle, mean_z_other = self._add_nodes_check(gv.position, nozzle_side, deposition_side)
        if not add_check:
            return
        new_nodes = gv.position[nozzle_side].clone()
        new_nodes[:, 2] = 0.5 * (mean_z_nozzle + mean_z_other)
        n_new = new_nodes.shape[0]
        new_velocities = nozzle_vel_world[None, None, :].expand(n_new, gv.prev_velocities.shape[1], -1).clone()
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device, dtype=gv.prev_pressures.dtype)
        new_layers = torch.full((n_new,), self._new_node_layer(), dtype=gv.node_layer.dtype, device=gv.device)
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
        gv.node_layer = torch.cat([gv.node_layer, new_layers], dim=0)

    def _remesh_global(self):
        cells, fs_tags = self.mesher.generate_initial_mesh(gv.position)
        gv.cells = cells
        gv.free_surf = torch.zeros(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        gv.free_surf[fs_tags] = True
        self.filter_mesh()

    # ------------------------------------------------------------------
    # Preprocessor
    # ------------------------------------------------------------------

    def preprocessor(self, most_recent_position: torch.Tensor, velocity_sequence: torch.Tensor, pressure_sequence: torch.Tensor, z_floor: torch.Tensor = None):
        n_particles = most_recent_position.shape[0]
        scaled_vel = (velocity_sequence - gv.vel_mean) / gv.vel_std
        scaled_press = (pressure_sequence - gv.press_mean) / gv.press_std
        velpress_hist = torch.cat([scaled_vel, scaled_press.unsqueeze(-1)], dim=2)
        node_features = torch.cat([velpress_hist.view(n_particles, -1), ((most_recent_position[:, 2].unsqueeze(-1) - z_floor) / 0.005).clamp(-1, 1)], dim=-1)
        tau0_star = gv.yield_stress / (gv.density * 9.81 * 2 * gv.nozzle_radius)
        p1 = gv.viscosity / 100
        global_features = torch.tensor([tau0_star, p1], device=gv.device)
        return node_features, global_features

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def deep_learning_update(self):
        tp = gv.toolpath
        self._ensure_node_layers()
        self.check_and_insert_nodes()
        nozzle_center = gv.position[gv.nozzle_ids].mean(dim=0)
        tp.advance_if_needed()
        if self._handle_z_transition(tp):
            return
        self._update_active_region(nozzle_center, radius=6 * gv.nozzle_radius)
        self._cull_layer2_nodes()
        self._local_remesh()
        spatial_mask = gv.active == 1
        self._forward_predict(spatial_mask, nozzle_center, tp)
        fs_tags = torch.unique(self.mesher._boundary_faces_3d(gv.cells))
        gv.free_surf = torch.zeros(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        gv.free_surf[fs_tags] = True

    # ------------------------------------------------------------------
    # Z transition
    # ------------------------------------------------------------------

    def _handle_z_transition(self, tp):
        is_transition = bool(tp.is_z_transition)
        was_transition = bool(getattr(gv, "was_z_transition", False))
        if is_transition:
            self._handle_z_transition_state(tp)
            return True
        if was_transition and not getattr(gv, "second_layer_init_done", False):
            nozzle_pos = gv.position[gv.nozzle_ids].clone()
            self._init_second_layer(tp, gv.flow_velocity, nozzle_pos)
        gv.was_z_transition = False
        return False

    def _handle_z_transition_state(self, tp):
        n_nozzle = gv.n_nozzle_nodes
        nozzle_vel_world = torch.tensor([0.0, 0.0, float(tp.nozzle_velocity[2]) - gv.flow_velocity], dtype=torch.float32, device=gv.device)
        gv.position[:n_nozzle] = gv.position[:n_nozzle] + tp.nozzle_move * gv.cfg.dt
        gv.velocity = torch.zeros_like(gv.prev_velocities[:, -1, :])
        gv.pressure = torch.zeros_like(gv.prev_pressures[:, -1])
        gv.velocity[:n_nozzle] = nozzle_vel_world
        last_velocities = torch.zeros_like(gv.prev_velocities[:, -1:, :])
        last_pressures = torch.zeros_like(gv.prev_pressures[:, -1:])
        gv.prev_velocities = torch.cat([gv.prev_velocities[:, 1:, :], last_velocities], dim=1)
        gv.prev_pressures = torch.cat([gv.prev_pressures[:, 1:], last_pressures], dim=1)
        gv.was_z_transition = True

    # ------------------------------------------------------------------
    # Second layer initialization
    # ------------------------------------------------------------------

    def _init_second_layer(self, tp, flow_velocity, nozzle_pos):
        if getattr(gv, "second_layer_init_done", False):
            return
        n_nozzle = gv.n_nozzle_nodes
        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -flow_velocity], dtype=torch.float32, device=gv.device)
        radius = nozzle_pos[:, 0].max() - nozzle_pos[:, 0].min()
        if tp.is_straight:
            pos_local = gv.position
            nozzle_local = nozzle_pos
        else:
            nozzle_center = nozzle_pos.mean(dim=0)
            pos_local = tp.rotate_to_local(gv.position - nozzle_center) + nozzle_center
            nozzle_local = tp.rotate_to_local(nozzle_pos - nozzle_center) + nozzle_center
            if tp.is_rotating_right:
                pos_local[:, 1] *= -1
                nozzle_local[:, 1] *= -1
        x_nozzle = nozzle_local[:, 0].mean()
        x_dist = torch.abs(pos_local[:, 0] - x_nozzle)
        far = (x_dist >= 2 * radius) & (x_dist <= 3 * radius)
        if far.any():
            gv.z_floor = float(gv.position[far, 2].max())
        else:
            gv.z_floor = float(gv.position[n_nozzle:, 2].min())
        h0 = torch.tensor(gv.z_floor, dtype=torch.float32, device=gv.device)
        z_top = gv.position[0, 2].clamp(min=h0 + 1e-6)
        z_all = gv.position[:, 2]
        span = (z_top - h0).clamp(min=1e-6)
        alpha = ((z_all - h0) / span).clamp(0.0, 1.0)
        velocity_init = alpha.unsqueeze(-1) * nozzle_vel_world.unsqueeze(0)
        velocity_init[:n_nozzle] = nozzle_vel_world
        gv.prev_velocities = velocity_init.unsqueeze(1).expand(-1, gv.prev_velocities.shape[1], -1).clone().to(gv.prev_velocities.dtype)
        gv.prev_pressures.zero_()
        self._ensure_node_layers()
        gv.node_layer[n_nozzle:] = torch.where(gv.node_layer[n_nozzle:] == 0, torch.ones_like(gv.node_layer[n_nozzle:]), gv.node_layer[n_nozzle:])
        gv.second_layer_init_done = True

    # ------------------------------------------------------------------
    # Layer 2 culling
    # ------------------------------------------------------------------
    def _cull_layer2_nodes(self, cull_radius=0.0025):
        if not getattr(gv, "second_layer_init_done", False):
            return
        layer1_idx = torch.where(gv.node_layer == 1)[0]
        layer2_idx = torch.where(gv.node_layer == 2)[0]
        if layer1_idx.numel() == 0 or layer2_idx.numel() == 0:
            return
        active_layer2_idx = layer2_idx[gv.active[layer2_idx] == 1]
        if active_layer2_idx.numel() == 0:
            return
        distances = torch.cdist(gv.position[active_layer2_idx], gv.position[layer1_idx])
        min_distances = distances.min(dim=1).values
        drop_global = active_layer2_idx[min_distances < cull_radius]
        if drop_global.numel() == 0:
            return
        keep = torch.ones(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        keep[drop_global] = False
        keep[:gv.n_nozzle_nodes] = True
        old_cells = gv.cells
        if old_cells is not None and old_cells.numel() > 0:
            cell_keep = keep[old_cells].all(dim=1)
            surviving_cells = old_cells[cell_keep]
            old_to_new = torch.cumsum(keep.to(torch.long), dim=0) - 1
            gv.cells = old_to_new[surviving_cells]
        gv.position = gv.position[keep]
        gv.prev_velocities = gv.prev_velocities[keep]
        gv.prev_pressures = gv.prev_pressures[keep]
        gv.active = gv.active[keep]
        gv.node_layer = gv.node_layer[keep]
        gv.velocity = gv.velocity[keep]
        gv.pressure = gv.pressure[keep]
        if gv.free_surf is not None:
            gv.free_surf = gv.free_surf[keep]

    # ------------------------------------------------------------------
    # Node insertion
    # ------------------------------------------------------------------
    def check_and_insert_nodes(self):
        prev_cells = gv.cells
        if prev_cells is None:
            return
        mask_contains_eucl = (prev_cells < gv.n_nozzle_nodes).any(dim=1)
        relevant_cells = prev_cells[mask_contains_eucl]
        connected_eucl, connected_non = self._boundary_nodes(relevant_cells, gv.n_nozzle_nodes)
        if connected_eucl.numel() == 0 or connected_non.numel() == 0:
            return
        add_check, mean_z_eucl, mean_z_non = self._add_nodes_check(gv.position, connected_eucl, connected_non)
        if not add_check:
            return
        self.interpolate_on_new_nodes(relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non)

    def interpolate_on_new_nodes(self, relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non):
        new_z = 0.5 * (mean_z_eucl + mean_z_non)
        new_nodes = gv.position[connected_eucl].clone()
        new_nodes[:, 2] = new_z
        n_new = new_nodes.shape[0]
        cells_old = torch.as_tensor(relevant_cells, dtype=torch.long, device=gv.device)
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
        new_velocities = torch.zeros(n_new, n_steps, gv.prev_velocities.shape[2], device=gv.device) + torch.tensor([gv.printing_velocity, 0, -gv.flow_velocity], device=gv.device)
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device, dtype=gv.prev_pressures.dtype)
        if has_tet.any():
            rows = torch.arange(n_new, device=gv.device)[has_tet]
            bary_sel = bary[rows, tet_ids[has_tet]]
            tet_vels = gv.prev_velocities[cells_valid[tet_ids[has_tet]]]
            tet_press = gv.prev_pressures[cells_valid[tet_ids[has_tet]]]
            new_velocities[has_tet] = (bary_sel[:, :, None, None] * tet_vels).sum(dim=1)
            new_pressures[has_tet] = (bary_sel[:, :, None] * tet_press).sum(dim=1)
        if (~has_tet).any():
            connected_nodes_t = torch.unique(torch.cat([connected_eucl, connected_non]))
            nearest = torch.cdist(new_nodes[~has_tet], gv.position[connected_nodes_t]).argmin(dim=1)
            new_velocities[~has_tet] = gv.prev_velocities[connected_nodes_t[nearest]].to(new_velocities.dtype)
            new_pressures[~has_tet] = gv.prev_pressures[connected_nodes_t[nearest]].to(new_pressures.dtype)
        new_layers = torch.full((n_new,), self._new_node_layer(), dtype=gv.node_layer.dtype, device=gv.device)
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.velocity = torch.cat([gv.velocity, new_velocities[:, -1, :]], dim=0)
        gv.pressure = torch.cat([gv.pressure, new_pressures[:, -1]], dim=0)
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
        gv.active = torch.cat([gv.active, torch.ones(n_new, dtype=gv.active.dtype, device=gv.device)])
        gv.node_layer = torch.cat([gv.node_layer, new_layers], dim=0)
        gv.free_surf = torch.cat([gv.free_surf, torch.zeros(n_new, dtype=torch.bool, device=gv.device)], dim=0)

    def filter_mesh(self, alpha: float = 300.0):
        """Applies global node rules to filter the current mesh."""
        if gv.cells is None or gv.cells.numel() == 0:
            return

        cells = gv.cells
        num_nodes = gv.position.shape[0]

        # 1. Identify node types using global indices
        nozzle_nodes = torch.arange(num_nodes, device=gv.device) < gv.n_nozzle_nodes
        new_nodes = torch.arange(num_nodes, device=gv.device) >= num_nodes - gv.n_nozzle_nodes
        others = ~(nozzle_nodes | new_nodes)

        # 2. Evaluate cell compositions
        has_euclidean = nozzle_nodes[cells].any(dim=1)
        has_new = new_nodes[cells].any(dim=1)
        internal_cells = others[cells].any(dim=1)

        free_surf = gv.free_surf
        all_free_surf = free_surf[cells].all(dim=1)

        simplex_layers = gv.node_layer[cells]
        has_layer1 = (simplex_layers == 1).any(dim=1)
        has_layer2 = (simplex_layers == 2).any(dim=1)

        # Calculate circumradius for the alpha-shape condition
        radius = self.mesher._circumradius_3d(gv.position[cells])
        too_large = radius > (1.0 / alpha)

        # Start with all cells kept
        mask = torch.ones(cells.shape[0], dtype=torch.bool, device=gv.device)

        # 3. Apply the filtering rules
        # Remove cells connecting nozzle nodes to internal nodes
        mask &= ~(has_euclidean & internal_cells)
        
        # Remove bridging cells across layers that are entirely on the free surface
        mask &= ~(all_free_surf & has_layer1 & has_layer2)

        # Remove bridging cells across layers that fail the alpha-shape condition
        mask &= ~(has_layer1 & has_layer2 & too_large)

        # EXCEPTION RULE: A cell connecting nozzle node with new nodes cannot be eliminated
        mask |= (has_euclidean & has_new & ~internal_cells)

        # 4. Update the global cells
        filtered_cells = cells[mask]
        gv.cells = filtered_cells

        # 5. Recompute the boundary/free surface tags now that invalid cells are gone
        if filtered_cells.numel() > 0:
            fs_tags = torch.unique(self.mesher._boundary_faces_3d(filtered_cells))
            gv.free_surf = torch.zeros(num_nodes, dtype=torch.bool, device=gv.device)
            gv.free_surf[fs_tags] = True

    # ------------------------------------------------------------------
    # Active region
    # ------------------------------------------------------------------

    @staticmethod
    def _update_active_region(nozzle_center: torch.Tensor, radius: torch.Tensor):
        dists = torch.norm(gv.position - nozzle_center, dim=1)
        physically_inside = dists <= radius
        # Nodes that are currently inactive (either 0 or -1) but are inside the radius
        just_entered = (gv.active != 1) & physically_inside
        # Block re-entry ONLY for nodes that have explicitly exited (active == -1) 
        # while we are still on the first layer
        if not getattr(gv, "second_layer_init_done", False):
            exited_layer_1 = (gv.node_layer == 1) & (gv.active == -1)
            just_entered = just_entered & ~exited_layer_1
        gv.active[just_entered] = 1
        # When a node leaves the active region, mark it as exited (-1) instead of 0
        just_exited = (gv.active == 1) & ~physically_inside
        gv.active[just_exited] = -1

    # ------------------------------------------------------------------
    # Local remeshing
    # ------------------------------------------------------------------

    def _local_remesh(self):
        active = gv.active == 1
        cells = gv.cells
        if cells is None or cells.numel() == 0:
            return

        cell_active = active[cells]
        any_active = cell_active.any(dim=1)
        kept_cells = cells[~any_active]
        discarded_cells = cells[any_active]

        discarded_nodes = torch.unique(discarded_cells)
        meshed_nodes = torch.unique(cells)
        active_nodes = torch.where(active)[0]
        unmeshed_active_nodes = active_nodes[~torch.isin(active_nodes, meshed_nodes)]
        nodes_to_remesh = torch.unique(torch.cat([discarded_nodes, unmeshed_active_nodes]))
        if nodes_to_remesh.numel() < 4:
            return

        new_cells_local = self.mesher.generate_mesh(
            gv.position[nodes_to_remesh],
            return_boundary_nodes=False,
        )
        new_cells = nodes_to_remesh[new_cells_local]
        new_cell_active = active[new_cells]
        valid_new_cells = new_cells[new_cell_active.any(dim=1)]

        gv.cells = torch.cat((kept_cells, valid_new_cells), dim=0)
        self.filter_mesh()

    # ------------------------------------------------------------------
    # Forward prediction
    # ------------------------------------------------------------------

    def _forward_predict(self, spatial_mask: torch.Tensor, nozzle_center: torch.Tensor, tp):
        prev_vel_local = tp.rotate_to_local(gv.prev_velocities.view(-1, 3)).view(gv.prev_velocities.shape)
        pos_local = tp.rotate_to_local(gv.position - nozzle_center)
        if tp.is_rotating_right:
            prev_vel_local[:, :, 1] *= -1
            pos_local[:, 1] *= -1
        scaling_pos = torch.tensor([3 * 0.0125, 3 * 0.0125, 0.02], device=gv.device)
        scaled_position = (pos_local - nozzle_center) / scaling_pos
        node_features, global_features = self.preprocessor(most_recent_position=pos_local[spatial_mask], velocity_sequence=prev_vel_local[spatial_mask], pressure_sequence=gv.prev_pressures[spatial_mask], z_floor=gv.z_floor)
        node_latent = self._encode(node_features, global_features)
        node_latent = self._process(node_latent, torch.zeros_like(scaled_position[spatial_mask, 0]), scaled_position[spatial_mask])
        norm_pred_vel, norm_pred_pos, norm_pred_press = self._decode(node_latent, global_features)

        pred_vel_local = (norm_pred_vel * gv.vel_std + gv.vel_mean).to(torch.float32)
        pred_disp_local = (norm_pred_pos * gv.vel_std + gv.vel_mean).to(torch.float32) * gv.cfg.dt
        if tp.is_rotating_right:
            pred_vel_local[:, 1] *= -1
            pred_disp_local[:, 1] *= -1
        pred_vel_world = tp.rotate_to_world(pred_vel_local).to(torch.float32)
        pred_disp_world = tp.rotate_to_world(pred_disp_local).to(torch.float32)

        # Reuse the existing state tensors instead of allocating full-state copies.
        gv.velocity.zero_()
        gv.pressure.zero_()
        gv.position[spatial_mask] += pred_vel_world * gv.cfg.dt
        gv.velocity[spatial_mask] = pred_vel_world
        gv.pressure[spatial_mask] = (norm_pred_press.squeeze(-1) * gv.press_std + gv.press_mean).to(gv.pressure.dtype)

        bd_mask = gv.position[:, -1] <= 0
        gv.velocity[bd_mask] = 0.0
        gv.position[bd_mask, -1] = 0

        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity], dtype=torch.float32, device=gv.device)
        gv.velocity[gv.nozzle_ids] = nozzle_vel_world
        gv.position[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
"""
Deep learning surrogate for 3DCP simulation.
"""
import numpy as np
import torch
import torch.nn as nn
from npfem import encoder_processor_decoder
from npfem import global_variables as gv
from npfem import mesh

_ADD_NODES_CONST = 0.17


class Surrogate(nn.Module):
    """Encoder-processor-decoder surrogate model for the printing simulation."""

    def __init__(self, n_in_features: int, latent_dim: int, n_mlp_layers: int, n_attn_heads: int, n_attn_layers: int, attn_dropout: float):
        super().__init__()
        self.mesher = mesh.MeshGenerator()
        self._encode = encoder_processor_decoder.Encoder(n_in_features=n_in_features, n_out_features=latent_dim, nmlp_layers=n_mlp_layers, mlp_hidden_dim=latent_dim).to(gv.device)
        self._process = encoder_processor_decoder.Processor(n_in_features=latent_dim, mlp_hidden_dim=latent_dim, nhead=n_attn_heads, nlayers=n_attn_layers, dropout=attn_dropout).to(gv.device)
        self._decode = encoder_processor_decoder.Decoder(n_in_features=latent_dim, nmlp_layers=n_mlp_layers, mlp_hidden_dim=latent_dim, output_dim_vel=3).to(gv.device)

    def learned_update(self):
        if gv.phase1:
            self.free_fall_update()
        else:
            self.deep_learning_update()

    # ------------------------------------------------------------------
    # Layer bookkeeping
    # ------------------------------------------------------------------

    def _ensure_node_layers(self):
        n_nodes = gv.position.shape[0]
        n_nozzle = gv.n_nozzle_nodes
        if not hasattr(gv, "node_layer"):
            gv.node_layer = torch.ones(n_nodes, dtype=torch.int64, device=gv.device)
            gv.node_layer[:n_nozzle] = 0
            return
        current_size = gv.node_layer.shape[0]
        if current_size == n_nodes:
            return
        if current_size > n_nodes:
            gv.node_layer = gv.node_layer[:n_nodes]
            return
        n_new = n_nodes - current_size
        new_layers = torch.ones(n_new, dtype=gv.node_layer.dtype, device=gv.device)
        gv.node_layer = torch.cat((gv.node_layer, new_layers), dim=0)

    @staticmethod
    def _new_node_layer():
        if getattr(gv, "second_layer_init_done", False):
            return 2
        return 1

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _boundary_nodes(cells: torch.Tensor, n_nozzle: int):
        mask = (cells < n_nozzle).any(dim=1)
        nodes = torch.unique(cells[mask])
        nozzle_side = nodes[nodes < n_nozzle]
        deposition_side = nodes[nodes >= n_nozzle]
        return nozzle_side, deposition_side

    @staticmethod
    def _add_nodes_check(position: torch.Tensor, nozzle_side, other_side):
        if nozzle_side.numel() == 0 or other_side.numel() == 0:
            return False, 0.0, 0.0
        mean_z_nozzle = position[nozzle_side, 2].mean()
        mean_z_other = position[other_side, 2].mean()
        min_x = position[nozzle_side, 0].min()
        max_x = position[nozzle_side, 0].max()
        gap = (mean_z_nozzle - mean_z_other).abs()
        add_check = gap >= _ADD_NODES_CONST * (max_x - min_x)
        return add_check, mean_z_nozzle, mean_z_other

    # ------------------------------------------------------------------
    # Phase 1
    # ------------------------------------------------------------------

    def free_fall_update(self):
        tp = gv.toolpath
        tp.advance_if_needed()
        self._ensure_node_layers()
        non_nozzle_z = gv.position[gv.n_nozzle_nodes:, 2]
        if non_nozzle_z.numel() > 0 and (non_nozzle_z <= 0).any():
            gv.phase1 = False
            gv.active = torch.zeros(gv.position.shape[0], dtype=torch.long, device=gv.device)
            return
        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity], dtype=torch.float32, device=gv.device)
        self._add_nodes_free_fall(nozzle_vel_world)
        self._remesh_global()
        next_pos = gv.position + nozzle_vel_world * gv.cfg.dt
        next_pos[next_pos[:, 2] < 0, 2] = 0.0
        next_pos[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
        z_top = gv.position[0, 2].clamp(min=1e-6)
        alpha = (next_pos[:, 2].clamp(min=0.0) / z_top).clamp(0.0, 1.0)
        velocity = alpha[:, None] * nozzle_vel_world[None, :]
        velocity[gv.nozzle_ids] = nozzle_vel_world
        gv.prev_velocities = velocity[:, None, :].expand(-1, gv.prev_velocities.shape[1], -1).clone().to(gv.prev_velocities.dtype)
        gv.position = next_pos
        gv.velocity = velocity
        gv.pressure = torch.zeros(gv.position.shape[0], device=gv.device, dtype=torch.float16)
        gv.was_z_transition = tp.is_z_transition

    def _add_nodes_free_fall(self, nozzle_vel_world: torch.Tensor):
        if gv.cells is None:
            return
        cells = gv.cells
        nozzle_cells = cells[(cells < gv.n_nozzle_nodes).any(dim=1)]
        nozzle_side, deposition_side = self._boundary_nodes(nozzle_cells, gv.n_nozzle_nodes)
        if deposition_side.numel() == 0:
            return
        add_check, mean_z_nozzle, mean_z_other = self._add_nodes_check(gv.position, nozzle_side, deposition_side)
        if not add_check:
            return
        new_nodes = gv.position[nozzle_side].clone()
        new_nodes[:, 2] = 0.5 * (mean_z_nozzle + mean_z_other)
        n_new = new_nodes.shape[0]
        new_velocities = nozzle_vel_world[None, None, :].expand(n_new, gv.prev_velocities.shape[1], -1).clone()
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device, dtype=gv.prev_pressures.dtype)
        new_layers = torch.full((n_new,), self._new_node_layer(), dtype=gv.node_layer.dtype, device=gv.device)
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
        gv.node_layer = torch.cat([gv.node_layer, new_layers], dim=0)

    def _remesh_global(self):
        cells, fs_tags = self.mesher.generate_initial_mesh(gv.position)
        gv.cells = cells
        gv.free_surf = torch.zeros(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        gv.free_surf[fs_tags] = True
        self.filter_mesh()

    # ------------------------------------------------------------------
    # Preprocessor
    # ------------------------------------------------------------------

    def preprocessor(self, most_recent_position: torch.Tensor, velocity_sequence: torch.Tensor, pressure_sequence: torch.Tensor, z_floor: torch.Tensor = None):
        n_particles = most_recent_position.shape[0]
        scaled_vel = (velocity_sequence - gv.vel_mean) / gv.vel_std
        scaled_press = (pressure_sequence - gv.press_mean) / gv.press_std
        velpress_hist = torch.cat([scaled_vel, scaled_press.unsqueeze(-1)], dim=2)
        node_features = torch.cat([velpress_hist.view(n_particles, -1), ((most_recent_position[:, 2].unsqueeze(-1) - z_floor) / 0.005).clamp(-1, 1)], dim=-1)
        tau0_star = gv.yield_stress / (gv.density * 9.81 * 2 * gv.nozzle_radius)
        p1 = gv.viscosity / 100
        global_features = torch.tensor([tau0_star, p1], device=gv.device)
        return node_features, global_features

    # ------------------------------------------------------------------
    # Phase 2
    # ------------------------------------------------------------------

    def deep_learning_update(self):
        tp = gv.toolpath
        self._ensure_node_layers()
        self.check_and_insert_nodes()
        nozzle_center = gv.position[gv.nozzle_ids].mean(dim=0)
        tp.advance_if_needed()
        if self._handle_z_transition(tp):
            return
        self._update_active_region(nozzle_center, radius=6 * gv.nozzle_radius)
        self._cull_layer2_nodes()
        self._local_remesh()
        spatial_mask = gv.active == 1
        self._forward_predict(spatial_mask, nozzle_center, tp)
        fs_tags = torch.unique(self.mesher._boundary_faces_3d(gv.cells))
        gv.free_surf = torch.zeros(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        gv.free_surf[fs_tags] = True

    # ------------------------------------------------------------------
    # Z transition
    # ------------------------------------------------------------------

    def _handle_z_transition(self, tp):
        is_transition = bool(tp.is_z_transition)
        was_transition = bool(getattr(gv, "was_z_transition", False))
        if is_transition:
            self._handle_z_transition_state(tp)
            return True
        if was_transition and not getattr(gv, "second_layer_init_done", False):
            nozzle_pos = gv.position[gv.nozzle_ids].clone()
            self._init_second_layer(tp, gv.flow_velocity, nozzle_pos)
        gv.was_z_transition = False
        return False

    def _handle_z_transition_state(self, tp):
        n_nozzle = gv.n_nozzle_nodes
        nozzle_vel_world = torch.tensor([0.0, 0.0, float(tp.nozzle_velocity[2]) - gv.flow_velocity], dtype=torch.float32, device=gv.device)
        gv.position[:n_nozzle] = gv.position[:n_nozzle] + tp.nozzle_move * gv.cfg.dt
        gv.velocity = torch.zeros_like(gv.prev_velocities[:, -1, :])
        gv.pressure = torch.zeros_like(gv.prev_pressures[:, -1])
        gv.velocity[:n_nozzle] = nozzle_vel_world
        last_velocities = torch.zeros_like(gv.prev_velocities[:, -1:, :])
        last_pressures = torch.zeros_like(gv.prev_pressures[:, -1:])
        gv.prev_velocities = torch.cat([gv.prev_velocities[:, 1:, :], last_velocities], dim=1)
        gv.prev_pressures = torch.cat([gv.prev_pressures[:, 1:], last_pressures], dim=1)
        gv.was_z_transition = True

    # ------------------------------------------------------------------
    # Second layer initialization
    # ------------------------------------------------------------------

    def _init_second_layer(self, tp, flow_velocity, nozzle_pos):
        if getattr(gv, "second_layer_init_done", False):
            return
        n_nozzle = gv.n_nozzle_nodes
        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -flow_velocity], dtype=torch.float32, device=gv.device)
        radius = nozzle_pos[:, 0].max() - nozzle_pos[:, 0].min()
        if tp.is_straight:
            pos_local = gv.position
            nozzle_local = nozzle_pos
        else:
            nozzle_center = nozzle_pos.mean(dim=0)
            pos_local = tp.rotate_to_local(gv.position - nozzle_center) + nozzle_center
            nozzle_local = tp.rotate_to_local(nozzle_pos - nozzle_center) + nozzle_center
            if tp.is_rotating_right:
                pos_local[:, 1] *= -1
                nozzle_local[:, 1] *= -1
        x_nozzle = nozzle_local[:, 0].mean()
        x_dist = torch.abs(pos_local[:, 0] - x_nozzle)
        far = (x_dist >= 2 * radius) & (x_dist <= 3 * radius)
        if far.any():
            gv.z_floor = float(gv.position[far, 2].max())
        else:
            gv.z_floor = float(gv.position[n_nozzle:, 2].min())
        h0 = torch.tensor(gv.z_floor, dtype=torch.float32, device=gv.device)
        z_top = gv.position[0, 2].clamp(min=h0 + 1e-6)
        z_all = gv.position[:, 2]
        span = (z_top - h0).clamp(min=1e-6)
        alpha = ((z_all - h0) / span).clamp(0.0, 1.0)
        velocity_init = alpha.unsqueeze(-1) * nozzle_vel_world.unsqueeze(0)
        velocity_init[:n_nozzle] = nozzle_vel_world
        gv.prev_velocities = velocity_init.unsqueeze(1).expand(-1, gv.prev_velocities.shape[1], -1).clone().to(gv.prev_velocities.dtype)
        gv.prev_pressures.zero_()
        self._ensure_node_layers()
        gv.node_layer[n_nozzle:] = torch.where(gv.node_layer[n_nozzle:] == 0, torch.ones_like(gv.node_layer[n_nozzle:]), gv.node_layer[n_nozzle:])
        gv.second_layer_init_done = True

    # ------------------------------------------------------------------
    # Layer 2 culling
    # ------------------------------------------------------------------
    def _cull_layer2_nodes(self, cull_radius=0.0025):
        if not getattr(gv, "second_layer_init_done", False):
            return
        layer1_idx = torch.where(gv.node_layer == 1)[0]
        layer2_idx = torch.where(gv.node_layer == 2)[0]
        if layer1_idx.numel() == 0 or layer2_idx.numel() == 0:
            return
        active_layer2_idx = layer2_idx[gv.active[layer2_idx] == 1]
        if active_layer2_idx.numel() == 0:
            return
        distances = torch.cdist(gv.position[active_layer2_idx], gv.position[layer1_idx])
        min_distances = distances.min(dim=1).values
        drop_global = active_layer2_idx[min_distances < cull_radius]
        if drop_global.numel() == 0:
            return
        keep = torch.ones(gv.position.shape[0], dtype=torch.bool, device=gv.device)
        keep[drop_global] = False
        keep[:gv.n_nozzle_nodes] = True
        old_cells = gv.cells
        if old_cells is not None and old_cells.numel() > 0:
            cell_keep = keep[old_cells].all(dim=1)
            surviving_cells = old_cells[cell_keep]
            old_to_new = torch.cumsum(keep.to(torch.long), dim=0) - 1
            gv.cells = old_to_new[surviving_cells]
        gv.position = gv.position[keep]
        gv.prev_velocities = gv.prev_velocities[keep]
        gv.prev_pressures = gv.prev_pressures[keep]
        gv.active = gv.active[keep]
        gv.node_layer = gv.node_layer[keep]
        gv.velocity = gv.velocity[keep]
        gv.pressure = gv.pressure[keep]
        if gv.free_surf is not None:
            gv.free_surf = gv.free_surf[keep]

    # ------------------------------------------------------------------
    # Node insertion
    # ------------------------------------------------------------------
    def check_and_insert_nodes(self):
        prev_cells = gv.cells
        if prev_cells is None:
            return
        mask_contains_eucl = (prev_cells < gv.n_nozzle_nodes).any(dim=1)
        relevant_cells = prev_cells[mask_contains_eucl]
        connected_eucl, connected_non = self._boundary_nodes(relevant_cells, gv.n_nozzle_nodes)
        if connected_eucl.numel() == 0 or connected_non.numel() == 0:
            return
        add_check, mean_z_eucl, mean_z_non = self._add_nodes_check(gv.position, connected_eucl, connected_non)
        if not add_check:
            return
        self.interpolate_on_new_nodes(relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non)

    def interpolate_on_new_nodes(self, relevant_cells, connected_eucl, connected_non, mean_z_eucl, mean_z_non):
        new_z = 0.5 * (mean_z_eucl + mean_z_non)
        new_nodes = gv.position[connected_eucl].clone()
        new_nodes[:, 2] = new_z
        n_new = new_nodes.shape[0]
        cells_old = torch.as_tensor(relevant_cells, dtype=torch.long, device=gv.device)
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
        new_velocities = torch.zeros(n_new, n_steps, gv.prev_velocities.shape[2], device=gv.device) + torch.tensor([gv.printing_velocity, 0, -gv.flow_velocity], device=gv.device)
        new_pressures = torch.zeros(n_new, gv.prev_pressures.shape[1], device=gv.device, dtype=gv.prev_pressures.dtype)
        if has_tet.any():
            rows = torch.arange(n_new, device=gv.device)[has_tet]
            bary_sel = bary[rows, tet_ids[has_tet]]
            tet_vels = gv.prev_velocities[cells_valid[tet_ids[has_tet]]]
            tet_press = gv.prev_pressures[cells_valid[tet_ids[has_tet]]]
            new_velocities[has_tet] = (bary_sel[:, :, None, None] * tet_vels).sum(dim=1)
            new_pressures[has_tet] = (bary_sel[:, :, None] * tet_press).sum(dim=1)
        if (~has_tet).any():
            connected_nodes_t = torch.unique(torch.cat([connected_eucl, connected_non]))
            nearest = torch.cdist(new_nodes[~has_tet], gv.position[connected_nodes_t]).argmin(dim=1)
            new_velocities[~has_tet] = gv.prev_velocities[connected_nodes_t[nearest]].to(new_velocities.dtype)
            new_pressures[~has_tet] = gv.prev_pressures[connected_nodes_t[nearest]].to(new_pressures.dtype)
        new_layers = torch.full((n_new,), self._new_node_layer(), dtype=gv.node_layer.dtype, device=gv.device)
        gv.position = torch.cat([gv.position, new_nodes], dim=0)
        gv.velocity = torch.cat([gv.velocity, new_velocities[:, -1, :]], dim=0)
        gv.pressure = torch.cat([gv.pressure, new_pressures[:, -1]], dim=0)
        gv.prev_velocities = torch.cat([gv.prev_velocities, new_velocities], dim=0)
        gv.prev_pressures = torch.cat([gv.prev_pressures, new_pressures], dim=0)
        gv.active = torch.cat([gv.active, torch.ones(n_new, dtype=gv.active.dtype, device=gv.device)])
        gv.node_layer = torch.cat([gv.node_layer, new_layers], dim=0)
        gv.free_surf = torch.cat([gv.free_surf, torch.zeros(n_new, dtype=torch.bool, device=gv.device)], dim=0)

    def filter_mesh(self, alpha: float = 300.0):
        """Applies global node rules to filter the current mesh."""
        if gv.cells is None or gv.cells.numel() == 0:
            return

        cells = gv.cells
        num_nodes = gv.position.shape[0]

        # 1. Identify node types using global indices
        nozzle_nodes = torch.arange(num_nodes, device=gv.device) < gv.n_nozzle_nodes
        new_nodes = torch.arange(num_nodes, device=gv.device) >= num_nodes - gv.n_nozzle_nodes
        others = ~(nozzle_nodes | new_nodes)

        # 2. Evaluate cell compositions
        has_euclidean = nozzle_nodes[cells].any(dim=1)
        has_new = new_nodes[cells].any(dim=1)
        internal_cells = others[cells].any(dim=1)

        free_surf = gv.free_surf
        all_free_surf = free_surf[cells].all(dim=1)

        simplex_layers = gv.node_layer[cells]
        has_layer1 = (simplex_layers == 1).any(dim=1)
        has_layer2 = (simplex_layers == 2).any(dim=1)

        # Calculate circumradius for the alpha-shape condition
        radius = self.mesher._circumradius_3d(gv.position[cells])
        too_large = radius > (1.0 / alpha)

        # Start with all cells kept
        mask = torch.ones(cells.shape[0], dtype=torch.bool, device=gv.device)

        # 3. Apply the filtering rules
        # Remove cells connecting nozzle nodes to internal nodes
        mask &= ~(has_euclidean & internal_cells)
        
        # Remove bridging cells across layers that are entirely on the free surface
        mask &= ~(all_free_surf & has_layer1 & has_layer2)

        # Remove bridging cells across layers that fail the alpha-shape condition
        mask &= ~(has_layer1 & has_layer2 & too_large)

        # EXCEPTION RULE: A cell connecting nozzle node with new nodes cannot be eliminated
        mask |= (has_euclidean & has_new & ~internal_cells)

        # 4. Update the global cells
        filtered_cells = cells[mask]
        gv.cells = filtered_cells

        # 5. Recompute the boundary/free surface tags now that invalid cells are gone
        if filtered_cells.numel() > 0:
            fs_tags = torch.unique(self.mesher._boundary_faces_3d(filtered_cells))
            gv.free_surf = torch.zeros(num_nodes, dtype=torch.bool, device=gv.device)
            gv.free_surf[fs_tags] = True

    # ------------------------------------------------------------------
    # Active region
    # ------------------------------------------------------------------

    @staticmethod
    def _update_active_region(nozzle_center: torch.Tensor, radius: torch.Tensor):
        dists = torch.norm(gv.position - nozzle_center, dim=1)
        physically_inside = dists <= radius
        # Nodes that are currently inactive (either 0 or -1) but are inside the radius
        just_entered = (gv.active != 1) & physically_inside
        # Block re-entry ONLY for nodes that have explicitly exited (active == -1) 
        # while we are still on the first layer
        if not getattr(gv, "second_layer_init_done", False):
            exited_layer_1 = (gv.node_layer == 1) & (gv.active == -1)
            just_entered = just_entered & ~exited_layer_1
        gv.active[just_entered] = 1
        # When a node leaves the active region, mark it as exited (-1) instead of 0
        just_exited = (gv.active == 1) & ~physically_inside
        gv.active[just_exited] = -1

    # ------------------------------------------------------------------
    # Local remeshing
    # ------------------------------------------------------------------

    def _local_remesh(self):
        active = gv.active == 1
        cells = gv.cells
        if cells is None or cells.numel() == 0:
            return

        cell_active = active[cells]
        any_active = cell_active.any(dim=1)
        kept_cells = cells[~any_active]
        discarded_cells = cells[any_active]

        discarded_nodes = torch.unique(discarded_cells)
        meshed_nodes = torch.unique(cells)
        active_nodes = torch.where(active)[0]
        unmeshed_active_nodes = active_nodes[~torch.isin(active_nodes, meshed_nodes)]
        nodes_to_remesh = torch.unique(torch.cat([discarded_nodes, unmeshed_active_nodes]))
        if nodes_to_remesh.numel() < 4:
            return

        new_cells_local = self.mesher.generate_mesh(
            gv.position[nodes_to_remesh],
            return_boundary_nodes=False,
        )
        new_cells = nodes_to_remesh[new_cells_local]
        new_cell_active = active[new_cells]
        valid_new_cells = new_cells[new_cell_active.any(dim=1)]

        gv.cells = torch.cat((kept_cells, valid_new_cells), dim=0)
        self.filter_mesh()

    # ------------------------------------------------------------------
    # Forward prediction
    # ------------------------------------------------------------------

    def _forward_predict(self, spatial_mask: torch.Tensor, nozzle_center: torch.Tensor, tp):
        position = gv.position.clone()
        prev_vel_local = tp.rotate_to_local(gv.prev_velocities.view(-1, 3)).view(gv.prev_velocities.shape)
        pos_local = tp.rotate_to_local(position - nozzle_center)
        if tp.is_rotating_right:
            prev_vel_local[:, :, 1] *= -1
            pos_local[:, 1] *= -1
        pos_local = pos_local + nozzle_center
        scaling_pos = torch.tensor([3 * 0.0125, 3 * 0.0125, 0.02], device=gv.device)
        scaled_position = (pos_local - nozzle_center) / scaling_pos
        node_features, global_features = self.preprocessor(most_recent_position=pos_local[spatial_mask], velocity_sequence=prev_vel_local[spatial_mask], pressure_sequence=gv.prev_pressures[spatial_mask], z_floor=gv.z_floor)
        node_latent = self._encode(node_features, global_features)
        node_latent = self._process(node_latent, torch.zeros_like(scaled_position[spatial_mask, 0]), scaled_position[spatial_mask])
        norm_pred_vel, norm_pred_pos, norm_pred_press = self._decode(node_latent, global_features)
        velocity = torch.zeros_like(gv.position)
        pressure = torch.zeros(gv.position.shape[0], device=gv.device, dtype=torch.float16)
        next_pos = gv.position.clone()
        pred_vel_local = (norm_pred_vel * gv.vel_std + gv.vel_mean).to(torch.float32)
        pred_disp_local = (norm_pred_pos * gv.vel_std + gv.vel_mean).to(torch.float32) * gv.cfg.dt
        if tp.is_rotating_right:
            pred_vel_local[:, 1] *= -1
            pred_disp_local[:, 1] *= -1
        pred_vel_world = tp.rotate_to_world(pred_vel_local).to(torch.float32)
        pred_disp_world = tp.rotate_to_world(pred_disp_local).to(torch.float32)
        velocity[spatial_mask] = pred_vel_world
        next_pos[spatial_mask] = gv.position[spatial_mask] + velocity[spatial_mask] * gv.cfg.dt
        pressure[spatial_mask] = (norm_pred_press.squeeze(-1) * gv.press_std + gv.press_mean).to(pressure.dtype)
        bd_mask = next_pos[:, -1] <= 0
        velocity[bd_mask] = 0.0
        next_pos[bd_mask, -1] = 0
        nozzle_vel_world = torch.tensor([tp.nozzle_velocity[0], tp.nozzle_velocity[1], -gv.flow_velocity], dtype=torch.float32, device=gv.device)
        velocity[gv.nozzle_ids] = nozzle_vel_world
        next_pos[gv.nozzle_ids] = gv.position[gv.nozzle_ids] + tp.nozzle_move * gv.cfg.dt
        gv.position = next_pos
        gv.velocity = velocity
        gv.pressure = pressure