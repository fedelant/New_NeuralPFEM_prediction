"""
VTK Writer Module
"""

import numpy as np
import os
import torch 
import meshio
import pyvista as pv
    
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
    Write a single simulation timestep to a VTU file using
    VTK/PyVista with LZ4 compression.

    Output is standard .vtu and can be opened directly in ParaView.
    """

    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------------------
    # Convert tensors to NumPy
    # ---------------------------------------------------------
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()

    if torch.is_tensor(velocities):
        velocities = velocities.detach().cpu().numpy()

    if torch.is_tensor(pressures):
        pressures = pressures.detach().cpu().numpy()

    if torch.is_tensor(cells):
        cells = cells.detach().cpu().numpy()

    # ---------------------------------------------------------
    # Points: convert 2D -> 3D
    # ---------------------------------------------------------
    if positions.shape[1] == 2:
        coords = np.empty(
            (positions.shape[0], 3),
            dtype=np.float32,
        )
        coords[:, :2] = positions
        coords[:, 2] = 0.0
    else:
        coords = np.asarray(
            positions,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Velocity: convert 2D -> 3D
    # ---------------------------------------------------------
    if velocities.shape[1] == 2:
        velocity = np.empty(
            (velocities.shape[0], 3),
            dtype=np.float32,
        )
        velocity[:, :2] = velocities
        velocity[:, 2] = 0.0
    else:
        velocity = np.asarray(
            velocities,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Cell type
    # ---------------------------------------------------------
    n_nodes = cells.shape[1]

    if n_nodes == 3:
        vtk_cell_type = pv.CellType.TRIANGLE

    elif n_nodes == 4:
        vtk_cell_type = pv.CellType.TETRA

    else:
        raise ValueError(
            f"Unsupported element with {n_nodes} nodes."
        )

    # ---------------------------------------------------------
    # VTK cell array
    #
    # VTK requires:
    #
    # [n_nodes, i0, i1, i2,
    #  n_nodes, i0, i1, i2, ...]
    # ---------------------------------------------------------
    cells = np.asarray(cells, dtype=np.int64)

    vtk_cells = np.empty(
        cells.shape[0] * (n_nodes + 1),
        dtype=np.int64,
    )

    vtk_cells[::n_nodes + 1] = n_nodes

    for j in range(n_nodes):
        vtk_cells[j + 1::n_nodes + 1] = cells[:, j]

    # ---------------------------------------------------------
    # Cell types
    # ---------------------------------------------------------
    cell_types = np.full(
        cells.shape[0],
        vtk_cell_type,
        dtype=np.uint8,
    )

    # ---------------------------------------------------------
    # Create UnstructuredGrid
    # ---------------------------------------------------------
    grid = pv.UnstructuredGrid(
        vtk_cells,
        cell_types,
        coords,
    )

    # ---------------------------------------------------------
    # Point data
    # ---------------------------------------------------------
    grid.point_data["Velocity"] = velocity

    grid.point_data["Pressure"] = np.asarray(
        pressures,
        dtype=np.float32,
    )

    if free_surf is not None:

        if torch.is_tensor(free_surf):
            free_surf = free_surf.detach().cpu().numpy()

        grid.point_data["FreeSurface"] = np.asarray(
            free_surf,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Cell data
    # ---------------------------------------------------------
    if strain_rate is not None:

        if torch.is_tensor(strain_rate):
            strain_rate = strain_rate.detach().cpu().numpy()

        grid.cell_data["Stress"] = np.asarray(
            strain_rate,
            dtype=np.float32,
        )

    # ---------------------------------------------------------
    # Write VTU with LZ4
    # ---------------------------------------------------------
    filename = os.path.join(
        output_dir,
        f"step_{step_idx:04d}.vtu",
    )

    grid.save(
        filename,
        binary=True,
        compression="zlib",
    )

    del grid