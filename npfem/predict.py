"""NeuralPFEM 3DCP predictions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import DictConfig
from tqdm import tqdm

from npfem import global_variables as gv
from npfem import surrogate, toolpath, write_vtk


def initialize_simulator() -> Any:
    # input features: History of velocity and pressure, and the coordinate z*
    n_in_features = gv.cfg.input_sequence_length * 3 + gv.cfg.input_sequence_length + 1

    return surrogate.Surrogate(
        n_in_features=n_in_features,
        latent_dim=gv.cfg.latent_dim,
        n_mlp_layers=gv.cfg.n_layers,
        n_attn_heads=gv.cfg.n_attn_heads,
        n_attn_layers=gv.cfg.n_attn_layers,
        attn_dropout=gv.cfg.attn_dropout,
    )


    '''
        """Instantiate the learned simulator with normalized stats."""
        normalization_stats = {
            "velocity": {
                "mean": torch.tensor(metadata["vel_mean"], dtype=torch.float32, device=device),
                "std": torch.tensor(metadata["vel_std"], dtype=torch.float32, device=device),
            },
            "pressure": {
                "mean": metadata["press_mean"],
                "std": metadata["press_std"],
            },
            "position": {
                "min": torch.tensor(metadata["pos_min"], dtype=torch.float32, device=device),
                "max": torch.tensor(metadata["pos_max"], dtype=torch.float32, device=device),
            },
        }
'''


def build_toolpath(printing_speed: float, flow_speed: float):
    # if cfg.toolpath_type == "straight":
    return toolpath.Toolpath.make_straight(printing_speed, flow_speed)
    """
    other to be added and to understand whats the best way to handle this.
    """


def build_initial_position() -> torch.Tensor:

    points = gv.nozzle_points.to(
        device=gv.device,
        dtype=torch.float32,
    )

    # Reference center of the nozzle
    center = points[:, :2].mean(dim=0)

    # Reference radius computed from the nodes
    radial_distances = torch.linalg.norm(
        points[:, :2] - center,
        dim=1,
    )
    r_ref = radial_distances.max()

    # Rescale reference geometry to requested nozzle radius
    scale = gv.nozzle_radius / r_ref

    xy = center + (points[:, :2] - center) * scale

    z = torch.full(
        (len(points), 1),
        gv.nozzle_height,
        dtype=points.dtype,
        device=gv.device,
    )

    gv.position = torch.cat([xy, z], dim=1)
    second_layer = gv.position.clone()
    second_layer[:, 2] -= 0.002
    gv.position = torch.vstack([gv.position, second_layer])


def initialize_prediction_state() -> None:

    gv.nozzle_radius = gv.parameters["nozzle_radius"]
    gv.nozzle_height = gv.parameters["nozzle_height"]
    gv.printing_velocity = gv.parameters["printing_velocity"]
    gv.flow_velocity = gv.parameters["flow_velocity"]
    gv.density = gv.parameters["density"]
    gv.yield_stress = gv.parameters["yield_stress"]
    gv.viscosity = gv.parameters["viscosity"]

    build_initial_position()

    gv.velocity = torch.zeros_like(gv.position) + torch.tensor(
        [0.0, gv.printing_velocity, -gv.flow_velocity], device=gv.device
    )
    gv.pressure = torch.zeros(gv.position.shape[0], device=gv.device)
    gv.prev_pressures = torch.zeros(
        gv.position.shape[0], gv.cfg.input_sequence_length, device=gv.device
    )
    gv.prev_velocities = torch.zeros(
        gv.position.shape[0], gv.cfg.input_sequence_length, 3, device=gv.device
    ) + torch.tensor([0.0, gv.printing_velocity, -gv.flow_velocity], device=gv.device)

    # Store IDs of the nozzle nodes
    gv.nozzle_ids = torch.arange(
        gv.nozzle_points.shape[0],
        device=gv.device,
    )
    gv.n_nozzle_nodes = gv.nozzle_ids.shape[0]

    gv.vel_mean = torch.tensor(gv.cfg.vel_mean, device=gv.device)
    gv.vel_std = torch.tensor(gv.cfg.vel_std, device=gv.device)
    gv.press_mean = torch.tensor(gv.cfg.press_mean, device=gv.device)
    gv.press_std = torch.tensor(gv.cfg.press_std, device=gv.device)
    gv.pos_min = torch.tensor(gv.cfg.pos_min, device=gv.device)
    gv.pos_max = torch.tensor(gv.cfg.pos_max, device=gv.device)


def _reset_prediction_outputs() -> None:

    gv.position_output = []
    gv.velocity_output = []
    gv.pressure_output = []
    gv.cells_output = []
    gv.free_surf_output = []
    gv.h0 = float(gv.position[0, 2])
    gv.phase1 = True
    gv.was_z_transition = False
    gv.second_layer_init_done = False
    gv.z_floor = 0.0


def run_prediction() -> None:

    gv.toolpath = build_toolpath(gv.printing_velocity, gv.flow_velocity)

    nozzle_center_0 = gv.position[gv.nozzle_ids].mean(dim=0)
    toolpath_start = torch.tensor(
        [
            gv.toolpath.waypoints[0][0],
            gv.toolpath.waypoints[0][1],
            nozzle_center_0[2].item(),
        ],
        dtype=torch.float32,
        device=gv.device,
    )
    gv.position = gv.position + (toolpath_start - nozzle_center_0)

    gv.tags = np.arange(0, gv.position.shape[0])
    gv.cells = np.empty((0, 4))

    _reset_prediction_outputs()
    write_step = 0
    for step in tqdm(
        range(100), desc=f"Predicting {gv.example_i}"
    ):  # fino alla fine del toolpath, o fino a un numero massimo di step
        gv.model.learned_update()
        # if step % 3 == 0 : # make cfg.
        write_vtk.write_step(
            gv.position,
            gv.velocity,
            gv.pressure,
            gv.cells,
            output_dir=os.path.join(
                ".", "output", gv.cfg.model_name, f"{gv.example_i}_vtk"
            ),
            step_idx=write_step,
            strain_rate=None,
            free_surf=None,
        )
        write_step += 1

        gv.prev_velocities = torch.cat(
            [gv.prev_velocities[:, 1:], gv.velocity.unsqueeze(1)], dim=1
        )
        gv.prev_pressures = torch.cat(
            [gv.prev_pressures[:, 1:], gv.pressure.unsqueeze(1)], dim=1
        )

        if gv.toolpath.is_finished:
            print(f"Toolpath finished at step {step}, stopping rollout.")
            break


def load_initial_points(path: str) -> torch.Tensor:

    points = np.loadtxt(path)
    return torch.from_numpy(points).float()


def load_parameter_samples(params_path: str) -> list[dict]:

    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    mode = config["mode"]

    if mode == "single":
        return [config["single"]]

    if mode == "dataset":
        dataset = config["dataset"]

        df = pd.read_csv(
            dataset["file"],
            sep=dataset.get("separator", ","),
        )

        return df.to_dict(orient="records")

    raise ValueError(f"Unsupported prediction mode: {mode!r}")


def run() -> None:

    input_dir = Path("./input")

    # ---------------------------------------------------------
    # Load initial points
    # ---------------------------------------------------------
    initial_points_path = input_dir / "initial_nodes.txt"
    print(f"Loading initial points from: {initial_points_path}")
    gv.nozzle_points = load_initial_points(str(initial_points_path))
    print(f"Loaded {len(gv.nozzle_points)} initial points.")

    # ---------------------------------------------------------
    # Load parameter samples
    # ---------------------------------------------------------
    params_path = input_dir / "params.yaml"
    print(f"Loading parameters from: {params_path}")
    parameter_samples = load_parameter_samples(str(params_path))
    print(f"Number of parameter samples: {len(parameter_samples)}")

    # ---------------------------------------------------------
    # Initialize and load pre-trained model
    # ---------------------------------------------------------
    gv.model = initialize_simulator()

    model_path = os.path.join(
        ".",
        "models",
        gv.cfg.model_name,
        gv.cfg.model_file,
    )
    print(f"Loading model from: {model_path}")
    gv.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    gv.model.to(gv.device)
    gv.model.eval()

    # ---------------------------------------------------------
    # Run predictions
    # ---------------------------------------------------------
    with torch.no_grad():
        for gv.example_i, gv.parameters in enumerate(parameter_samples):
            print(f"\n--- Predicting sample {gv.example_i} ---")
            print(
                "Parameters: " + ", ".join(f"{k}={v}" for k, v in gv.parameters.items())
            )

            initialize_prediction_state()

            with torch.autocast(
                device_type=gv.device.type,
                enabled=True,  # cfg.use_amp,
            ):
                run_prediction()


@hydra.main(
    version_base=None,
    config_path="../input",
    config_name="config",
)
def main(cfg: DictConfig) -> None:

    if torch.cuda.is_available():
        gv.device = torch.device("cuda")
    else:
        gv.device = torch.device("cpu")
    print(f"Using device: {gv.device}")

    os.makedirs("./output", exist_ok=True)

    gv.cfg = cfg

    run()


if __name__ == "__main__":
    main()
