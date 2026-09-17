"""
VTK Writer Module
"""

import numpy as np
import os
import torch 
import meshio

def write_vtu(
    positions_list,    # list of [N, D] tensors or numpy arrays
    velocities_list,   # list of [N, D]
    pressures_list,    # list of [N]
    cells_list,        # list of [M, nodes_per_cell]
    output_dir: str,
    strain_rate_list=None,  # list of [M] (element-wise scalar)
    free_surf_list=None     # list of [N] (node-wise binary)
):
    """
    Write a sequence of simulation steps to binary VTU files for ParaView.

    Parameters
    ----------
    positions_list : list of [N, D]
        Node coordinates at each time step.
    velocities_list : list of [N, D]
        Nodal velocity vectors.
    pressures_list : list of [N]
        Nodal pressures.
    cells_list : list of [M, nodes_per_cell]
        Element connectivity per step.
    output_dir : str
        Directory to write VTU files.
    strain_rate_list : list of [M], optional
        Element-wise scalar effective strain-rate norm to write as cell data.
    free_surf_list : list of [N], optional
        Node-wise binary free surface indicator to write as point data.
    """
    os.makedirs(output_dir, exist_ok=True)
    num_steps = len(positions_list)
    save_idx = 0
    for step_idx in range(0, num_steps, 2):
        coords = positions_list[step_idx]
        if torch.is_tensor(coords):
            coords = coords.detach().cpu().numpy()
        velocity = velocities_list[step_idx]
        if torch.is_tensor(velocity):
            velocity = velocity.detach().cpu().numpy()
        pressure = pressures_list[step_idx]
        if torch.is_tensor(pressure):
            pressure = pressure.detach().cpu().numpy()
        cells = cells_list[step_idx]
        if torch.is_tensor(cells):
            cells = cells.detach().cpu().numpy()

        # Convert 2D to 3D for visualization
        if coords.shape[1] == 2:
            coords = np.hstack([coords, np.zeros((coords.shape[0], 1), dtype=coords.dtype)])
            velocity = np.hstack([velocity, np.zeros((velocity.shape[0], 1), dtype=velocity.dtype)])

        # Determine cell type
        nodes_per_cell = cells.shape[1]
        if nodes_per_cell == 3:
            cell_type = "triangle"
        elif nodes_per_cell == 4:
            cell_type = "tetra"
        else:
            raise ValueError(f"Unsupported cell type with {nodes_per_cell} nodes")

        cell_dict = {cell_type: cells.astype(np.int32)}

        # --- Cell data: strain-rate norm ---
        cell_data = {}
        if strain_rate_list is not None:
            str_rate = strain_rate_list[step_idx]
            if torch.is_tensor(str_rate):
                str_rate = str_rate.detach().cpu().numpy()
            cell_data = {"Stress": [str_rate.astype(np.float32)]}

        # --- Point data: velocity and pressure ---
        point_data = {
            "Velocity": velocity.astype(np.float32),
            "Pressure": pressure.astype(np.float32)
        }
        '''
        if free_surf_list is not None:
            free_surf = free_surf_list[step_idx]
            if torch.is_tensor(free_surf):
                free_surf = free_surf
            point_data["FreeSurface"] = free_surf.astype(np.float32)
        '''
        # --- Create mesh and write file ---
        mesh = meshio.Mesh(
            points=coords,
            cells=cell_dict,
            point_data=point_data,
            cell_data=cell_data
        )

        vtu_filename = os.path.join(output_dir, f"step_{save_idx:04d}.vtu")
        meshio.write(vtu_filename, mesh, file_format="vtu", binary=True, compression = None)
        save_idx += 1
    
def write_step(
    positions,
    velocities,
    pressures,
    cells,
    output_dir,
    step_idx,
    strain_rate=None,
    free_surf=None,
):
    """
    Write a single simulation timestep to a VTU file.

    Parameters
    ----------
    positions : (N,D) torch.Tensor or ndarray
    velocities : (N,D) torch.Tensor or ndarray
    pressures : (N,) torch.Tensor or ndarray
    cells : (M,nodes_per_cell) torch.Tensor or ndarray
    output_dir : str
    step_idx : int
        Output file index.
    strain_rate : (M,), optional
        Cell-wise scalar.
    free_surf : (N,), optional
        Point-wise scalar.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # Convert tensors to numpy
    # ----------------------------
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()

    if torch.is_tensor(velocities):
        velocities = velocities.detach().cpu().numpy()

    if torch.is_tensor(pressures):
        pressures = pressures.detach().cpu().numpy()

    if torch.is_tensor(cells):
        cells = cells.detach().cpu().numpy()

    # ----------------------------
    # Convert to 3D if needed
    # ----------------------------
    if positions.shape[1] == 2:
        coords = np.zeros((positions.shape[0], 3), dtype=np.float32)
        coords[:, :2] = positions

        velocity = np.zeros((velocities.shape[0], 3), dtype=np.float32)
        velocity[:, :2] = velocities
    else:
        coords = positions.astype(np.float32, copy=False)
        velocity = velocities.astype(np.float32, copy=False)

    # ----------------------------
    # Cell type
    # ----------------------------
    n_nodes = cells.shape[1]

    if n_nodes == 3:
        cell_type = "triangle"
    elif n_nodes == 4:
        cell_type = "tetra"
    else:
        raise ValueError(f"Unsupported element with {n_nodes} nodes.")

    # ----------------------------
    # Point data
    # ----------------------------
    point_data = {
        "Velocity": velocity.astype(np.float32, copy=False),
        "Pressure": pressures.astype(np.float32, copy=False),
    }

    if free_surf is not None:
        if torch.is_tensor(free_surf):
            free_surf = free_surf.detach().cpu().numpy()

        point_data["FreeSurface"] = free_surf.astype(np.float32, copy=False)

    # ----------------------------
    # Cell data
    # ----------------------------
    cell_data = {}

    if strain_rate is not None:
        if torch.is_tensor(strain_rate):
            strain_rate = strain_rate.detach().cpu().numpy()

        cell_data["Stress"] = [
            strain_rate.astype(np.float32, copy=False)
        ]

    # ----------------------------
    # Mesh
    # ----------------------------
    mesh = meshio.Mesh(
        points=coords,
        cells={cell_type: cells.astype(np.int32, copy=False)},
        point_data=point_data,
        cell_data=cell_data,
    )

    filename = os.path.join(output_dir, f"step_{step_idx:04d}.vtu")

    meshio.write(
        filename,
        mesh,
        file_format="vtu",
        binary=True,
        compression="zlib",
    )

    # Free memory immediately
    del mesh