import numpy as np
import torch
from npfem import global_variables as gv


class Toolpath:
    """
    Manages a piecewise-linear toolpath defined by 3D waypoints (x, y, z).

    Model coordinate system: x = print direction, y = out-of-plane, z = up.
    Flow is applied along -z by the caller (unchanged from 2D convention).

    Rotation strategy: 2D rotations in the x-y plane, same as before.
    Segments whose tangent is along ±z are flagged as z-transitions and
    handled separately: R = I, nozzle_velocity = [0, 0, ±speed].

    Parameters
    ----------
    waypoints : np.ndarray [M, 3]
        Sequence of (x, y, z) points.
    printing_speed : float
        Nozzle travel speed magnitude.
    flow_speed : float
        Material flow speed (applied along -z by the caller).
    device : torch.device or str
    start_tangent : np.ndarray [3] or None
        Unit tangent of the direction the training history was recorded in.
        Defaults to [1, 0, 0]. Only the x-y components are used for R_approach.
    """

    def __init__(self, waypoints, printing_speed, flow_speed,
                 start_tangent=None):
        assert len(waypoints) >= 2, "Toolpath needs at least 2 waypoints"
        assert np.asarray(waypoints).shape[1] == 3, "Waypoints must be 3D (x, y, z)"

        self.waypoints      = np.asarray(waypoints, dtype=np.float32)
        self.printing_speed = printing_speed
        self.flow_speed     = flow_speed
        self.segment_idx    = 0

        self.start_tangent = (
            np.asarray(start_tangent, dtype=np.float32)
            if start_tangent is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        self._precompute_segments()
        self._build_approach_rotation()
        self._update_current_segment()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _precompute_segments(self):
        pts = self.waypoints
        self.segment_dirs      = []
        self.segment_lengths   = []
        self.segment_R         = []   # 3x3 tensor, identity for z-transitions
        self.segment_is_z_move = []   # bool flag per segment
        self.segment_is_right  = []   # 1 if turning right, 0 otherwise

        # Initialize the previous direction to calculate the first turn
        prev_d_xy = self.start_tangent[:2]
        prev_d_xy = prev_d_xy / (np.linalg.norm(prev_d_xy) + 1e-9)

        for i in range(len(pts) - 1):
            delta  = pts[i + 1] - pts[i]
            length = float(np.linalg.norm(delta))
            d      = delta / length  # unit tangent [3]

            # Classify: is this segment purely along ±z?
            is_z = abs(d[2]) > 0.99

            if is_z:
                R = np.eye(3, dtype=np.float32)
                self.segment_is_right.append(0) # Z-moves don't turn right
            else:
                # 2D rotation in the x-y plane
                d_xy = d[:2] / np.linalg.norm(d[:2])   # re-normalise just in case
                R = np.array([[d_xy[0], -d_xy[1], 0.0],
                              [d_xy[1],  d_xy[0], 0.0],
                              [0.0,      0.0,     1.0]], dtype=np.float32)

                # Calculate 2D cross product: prev_x * curr_y - prev_y * curr_x
                # If < 0, it's a clockwise rotation (turn to the right)
                cross_prod = prev_d_xy[0] * d_xy[1] - prev_d_xy[1] * d_xy[0]
                is_right_turn = 1 if cross_prod < -1e-4 else 0
                self.segment_is_right.append(is_right_turn)

                # Update prev_d_xy for the next segment
                prev_d_xy = d_xy

            self.segment_dirs.append(d.astype(np.float32))
            self.segment_lengths.append(length)
            self.segment_R.append(R)
            self.segment_is_z_move.append(is_z)

    def _build_approach_rotation(self):
        """
        R_approach rotates the straight training history into the first
        segment's local frame. Uses 2D logic (x-y only), same as before.
        Skipped if the first segment is a z-transition (identity is fine).
        """
        if self.segment_is_z_move[0]:
            self.R_approach = torch.eye(3, dtype=torch.float32, device=gv.device)
            return

        d0   = self.segment_dirs[0][:2]
        st   = self.start_tangent[:2]
        st   = st / np.linalg.norm(st)
        cos_ = float(np.dot(st, d0))
        sin_ = float(np.cross(st, d0))
        R2   = np.array([[cos_, -sin_],
                         [sin_,  cos_]], dtype=np.float32)
        self.R_approach = self._R2_to_3x3(R2)

    def _update_current_segment(self):
        i      = self.segment_idx
        d      = self.segment_dirs[i]
        is_z   = self.segment_is_z_move[i]

        if is_z:
            # Pure z-move: identity frame, velocity straight up/down
            self.nozzle_velocity = torch.tensor(
                [0.0, 0.0, float(d[2]) * self.printing_speed],
                dtype=torch.float32, device=gv.device
            )
            self.nozzle_move = self.nozzle_velocity.clone()
            self.R  = torch.eye(3, dtype=torch.float32, device=gv.device)
            self.Rt = torch.eye(3, dtype=torch.float32, device=gv.device)
        else:
            # Normal x-y segment: original 2D rotation logic
            self.nozzle_velocity = torch.tensor(
                [d[0] * self.printing_speed,
                 d[1] * self.printing_speed,
                 0.0],
                dtype=torch.float32, device=gv.device
            )
            self.nozzle_move = self.nozzle_velocity.clone()
            self.R  = self._R2_to_3x3(self.segment_R[i])
            self.Rt = self.R.T

    def _R2_to_3x3(self, R2: np.ndarray) -> torch.Tensor:
        """Embed 2D rotation into 3D (acts on x-y plane, z unchanged)."""
        R3 = torch.eye(3, dtype=torch.float32, device=gv.device)
        R3[0, 0] = float(R2[0, 0])
        R3[0, 1] = float(R2[0, 1])
        R3[1, 0] = float(R2[1, 0])
        R3[1, 1] = float(R2[1, 1])
        return R3

    # ------------------------------------------------------------------
    # Runtime methods  (unchanged API)
    # ------------------------------------------------------------------

    def advance_if_needed(self) -> bool:
        i       = self.segment_idx
        p_start = self.waypoints[i]
        d       = self.segment_dirs[i]
        length  = self.segment_lengths[i]

        nozzle_center = gv.position[gv.nozzle_nodes].mean(dim=0)
        nozzle_xyz = nozzle_center[:3].cpu().numpy()
        proj       = float(np.dot(nozzle_xyz - p_start, d))

        if proj >= length:
            if self.segment_idx >= len(self.segment_dirs) - 1:
                # Already on the last segment and nozzle passed its end
                self._finished = True
                return False
            self.segment_idx += 1
            self._update_current_segment()
            return True

        return False

    def rotate_to_local(self, vectors: torch.Tensor) -> torch.Tensor:
        """Rotate [N, 3] vectors from world frame into current segment local frame."""
        return (self.Rt @ vectors.T).T

    def rotate_to_world(self, vectors: torch.Tensor) -> torch.Tensor:
        """Rotate [N, 3] vectors from current segment local frame to world frame."""
        return (self.R @ vectors.T).T

    def rotate_history_to_local(self, prev_velocities: torch.Tensor) -> torch.Tensor:
        """Rotate [N, C, 3] velocity history into current segment local frame."""
        N, C, D = prev_velocities.shape
        return self.rotate_to_local(prev_velocities.view(-1, 3)).view(N, C, D)

    def rotate_approach_history(self, prev_velocities: torch.Tensor) -> torch.Tensor:
        """
        Rotate [N, C, 3] initial straight history using R_approach,
        mapping the training direction into the first segment's local frame.
        """
        N, C, D = prev_velocities.shape
        return (self.R_approach @ prev_velocities.view(-1, 3).T).T.view(N, C, D)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def is_rotating_right(self) -> int:
        """Returns 1 if the current segment is rotating right (clockwise), 0 otherwise."""
        return self.segment_is_right[self.segment_idx]

    @property
    def is_straight(self) -> bool:
        return len(self.segment_dirs) == 1

    @property
    def is_z_transition(self) -> bool:
        """True when the current active segment is a z-move."""
        return self.segment_is_z_move[self.segment_idx]
    
    @property
    def is_finished(self) -> bool:
        return getattr(self, '_finished', False)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @staticmethod
    def make_straight(printing_speed, flow_speed, length=10.0):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float32
        )
        return Toolpath(waypoints, printing_speed, flow_speed)

    @staticmethod
    def make_arc(n_points, radius, scale, printing_speed, flow_speed, x_max=None):
        waypoints = generate_arc_toolpath(n_points, radius, scale, x_max)
        return Toolpath(waypoints, printing_speed, flow_speed)

    @staticmethod
    def make_sinusoid(n_points, amplitude, wavelength, scale, printing_speed,
                      flow_speed, x_max=None):
        waypoints = generate_sinusoidal_toolpath(
            n_points, amplitude, wavelength, scale, x_max
        )
        return Toolpath(waypoints, printing_speed, flow_speed)

    @staticmethod
    def make_2layer(printing_speed, flow_speed, length=10.0, layer_gap=0.2,
                    n_points_per_layer=2):
        """
        Layer 1   : (0, 0, 0)              -> (length, 0, 0)         +x
        Transition: (length, 0, 0)         -> (length, 0, layer_gap) +z
        Layer 2   : (length, 0, layer_gap) -> (0, 0, layer_gap)      -x

        Parameters
        ----------
        length : float
            Length of each straight layer pass.
        layer_gap : float
            z-offset between the two layers (build-up height).
        n_points_per_layer : int
            Waypoint density on each straight segment (min 2).
            The z-transition is always a single extra waypoint.
        """
        waypoints = generate_two_layer_toolpath(length, layer_gap,
                                                n_points_per_layer)
        return Toolpath(waypoints, printing_speed, flow_speed,
                        start_tangent=np.array([1.0, 0.0, 0.0], dtype=np.float32))

    @staticmethod
    def make_from_dat(filepath, scale, printing_speed, flow_speed):
        """
        Creates a Toolpath from a custom .dat file containing X, Y, Z coordinates.
        """
        waypoints = generate_from_dat(filepath, scale)
        return Toolpath(waypoints, printing_speed, flow_speed)


# ----------------------------------------------------------------------
# Waypoint generators
# ----------------------------------------------------------------------

def generate_arc_toolpath(n_points, radius, scale, x_max=None):
    """Quarter-circle arc in the x-y plane at z=0, starting at (0, 0, 0)."""
    theta = np.linspace(0, np.pi / 2, n_points)
    x = scale * radius * np.sin(theta)
    y = scale * radius * (1 - np.cos(theta))
    z = np.zeros_like(x)

    if x_max is not None:
        mask = x <= x_max
        x, y, z = x[mask], y[mask], z[mask]

    return np.stack((x, y, z), axis=1).astype(np.float32)


def generate_sinusoidal_toolpath(n_points, amplitude, wavelength, scale,
                                 x_max=None):
    """
    Sinusoidal path in the x-y plane at z=0, starting at (0, 0, 0).
    y = A * (cos(2*pi*x/lambda) - 1)  ->  y=0, dy/dx=0 at x=0.
    """
    x_end = x_max if x_max is not None else wavelength
    x     = np.linspace(0, x_end, n_points)
    y     = amplitude * (np.cos(2 * np.pi * x / wavelength) - 1)
    x     = scale * x
    y     = scale * y
    z     = np.zeros(len(x), dtype=np.float32)

    return np.stack((x, y, z), axis=1).astype(np.float32)


def generate_two_layer_toolpath(length, layer_gap, n_points_per_layer=2):
    """
    3D two-layer toolpath. See Toolpath.make_2layer for layout details.
    """
    n     = max(2, n_points_per_layer)
    zeros = np.zeros(n, dtype=np.float32)
    
    layer1 = np.stack([
        np.linspace(0.0, length, n, dtype=np.float32),
        zeros,
        zeros,
    ], axis=1)

    transition = np.array([[length, 0.0, layer_gap]], dtype=np.float32)
    
    layer2 = np.stack([
        np.linspace(length, 0.0, n, dtype=np.float32),
        zeros,
        np.full(n, layer_gap, dtype=np.float32),
    ], axis=1)

    # drop duplicate corners: layer1[-1] == transition[0], transition[-1] == layer2[0]
    return np.concatenate([layer1, transition, layer2[1:]], axis=0)

def generate_from_dat(filepath: str, scale: float = 1.0) -> np.ndarray:
    """
    Parses a .dat file to extract 3D toolpath waypoints.
    Expects format:
    % Toolpath
    ... headers ...
        X      Y      Z
    0    1.0    2.0    3.0
    %%
    """
    waypoints = []
    in_data_section = False
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
                
            # Stop parsing when hitting the end marker
            if line.startswith('%%') or line.startswith('% End'):
                break
                
            # Detect the start of the coordinate data
            if 'X' in line and 'Y' in line and 'Z' in line:
                in_data_section = True
                continue
                
            # Parse the coordinates if we are in the data section
            if in_data_section:
                parts = line.split()
                
                # Ensure we have at least 3 columns to parse (X, Y, Z)
                # We use parts[-3:] to grab the last three values, ignoring the leading '0'
                if len(parts) >= 3:
                    try:
                        x = float(parts[-3]) * scale
                        y = float(parts[-2]) * scale
                        z = float(parts[-1]) * scale
                        waypoints.append([x, y, z])
                    except ValueError:
                        # Skip lines that cannot be converted to floats
                        continue

    if not waypoints:
        raise ValueError(f"No valid coordinate data found in {filepath}")

    return np.array(waypoints, dtype=np.float32)



def get_sinusoidal_start_tangent(amplitude, wavelength):
    """Unit tangent at x=0 of y = A*sin(2pi*x/lam), z=0."""
    slope = 2 * np.pi * amplitude / wavelength
    d     = np.array([1.0, slope, 0.0], dtype=np.float32)
    return d / np.linalg.norm(d)