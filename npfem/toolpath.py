import numpy as np
import torch
import npfem.global_variables as gv


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

    def __init__(self, waypoints, printing_speed, flow_speed, device,
                 start_tangent=None, min_segment_length=1e-6):
        assert len(waypoints) >= 2, "Toolpath needs at least 2 waypoints"
        assert np.asarray(waypoints).shape[1] == 3, "Waypoints must be 3D (x, y, z)"

        self.waypoints      = np.asarray(waypoints, dtype=np.float32)
        self.printing_speed = printing_speed
        self.flow_speed     = flow_speed
        self.device         = device
        self.segment_idx    = 0
        self.min_segment_length = min_segment_length
        self._finished      = False

        self.start_tangent = (
            np.asarray(start_tangent, dtype=np.float32)
            if start_tangent is not None
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )

        self._precompute_segments()      # nominal geometry (unchanged)
        self._build_approach_rotation()

        # Active (dynamic) segment state: starts nominal, re-anchored on advance
        self._seg_start = self.waypoints[0].copy()
        self._seg_dir   = self.segment_dirs[0].copy()
        self._seg_len   = self.segment_lengths[0]
        self._seg_R     = self.segment_R[0]
        self._update_current_segment()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _rot_from_dir_xy(d_xy):
        return np.array([[d_xy[0], -d_xy[1], 0.0],
                         [d_xy[1],  d_xy[0], 0.0],
                         [0.0,      0.0,     1.0]], dtype=np.float32)

    def _retarget(self, i, nozzle_xyz) -> bool:
        """
        Make segment i the active one, starting at the nozzle's *actual*
        position and ending at waypoints[i+1].
        Returns False if the nozzle is already at/past that waypoint
        (segment should be skipped).
        """
        eps    = self.min_segment_length
        target = self.waypoints[i + 1]
        d_nom  = self.segment_dirs[i]

        if self.segment_is_z_move[i]:
            sgn = 1.0 if d_nom[2] > 0 else -1.0
            rem = sgn * float(target[2] - nozzle_xyz[2])   # remaining height
            if rem <= eps:
                return False
            d      = np.array([0.0, 0.0, sgn], dtype=np.float32)
            length = rem
            R      = np.eye(3, dtype=np.float32)
        else:
            d_nom_xy = d_nom[:2] / np.linalg.norm(d_nom[:2])
            delta_xy = (target[:2] - nozzle_xyz[:2]).astype(np.float32)
            # already beyond this waypoint along its nominal direction?
            if float(np.dot(delta_xy, d_nom_xy)) <= eps:
                return False
            length = float(np.linalg.norm(delta_xy))
            d_xy   = delta_xy / length
            d      = np.array([d_xy[0], d_xy[1], 0.0], dtype=np.float32)
            R      = self._rot_from_dir_xy(d_xy)

        self._seg_start = nozzle_xyz.astype(np.float32).copy()
        self._seg_dir   = d
        self._seg_len   = length
        self._seg_R     = R
        self.segment_idx = i
        return True

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
            self.R_approach = torch.eye(3, dtype=torch.float32, device=self.device)
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
        i    = self.segment_idx
        d    = self._seg_dir
        is_z = self.segment_is_z_move[i]

        if is_z:
            self.nozzle_velocity = torch.tensor(
                [0.0, 0.0, float(d[2]) * self.printing_speed],
                dtype=torch.float32, device=self.device
            )
            self.nozzle_move = self.nozzle_velocity.clone()
            self.R  = torch.eye(3, dtype=torch.float32, device=self.device)
            self.Rt = torch.eye(3, dtype=torch.float32, device=self.device)
        else:
            self.nozzle_velocity = torch.tensor(
                [d[0] * self.printing_speed,
                 d[1] * self.printing_speed,
                 0.0],
                dtype=torch.float32, device=self.device
            )
            self.nozzle_move = self.nozzle_velocity.clone()
            self.R  = self._R2_to_3x3(self._seg_R)
            self.Rt = self.R.T

    def _R2_to_3x3(self, R2: np.ndarray) -> torch.Tensor:
        """Embed 2D rotation into 3D (acts on x-y plane, z unchanged)."""
        R3 = torch.eye(3, dtype=torch.float32, device=self.device)
        R3[0, 0] = float(R2[0, 0])
        R3[0, 1] = float(R2[0, 1])
        R3[1, 0] = float(R2[1, 0])
        R3[1, 1] = float(R2[1, 1])
        return R3

    # ------------------------------------------------------------------
    # Runtime methods  (unchanged API)
    # ------------------------------------------------------------------
    def advance_if_needed(self) -> bool:
        nozzle_center = gv.position[gv.nozzle_ids].mean(dim=0)
        nozzle_xyz    = nozzle_center[:3].detach().cpu().numpy().astype(np.float32)

        proj = float(np.dot(nozzle_xyz - self._seg_start, self._seg_dir))
        if proj < self._seg_len:
            return False

        last = len(self.segment_dirs) - 1
        i = self.segment_idx
        while True:
            if i >= last:
                self._finished = True
                return False
            i += 1
            # start next segment from where the nozzle actually is
            if self._retarget(i, nozzle_xyz):
                break
            # else: nozzle already past that waypoint -> skip to the next one

        self._update_current_segment()
        return True

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
    def make_straight(printing_speed, flow_speed, length=10.0, device="cuda"):
        waypoints = np.array(
            [[0.0, 0.0, 0.0], [length, 0.0, 0.0]], dtype=np.float32
        )
        return Toolpath(waypoints, printing_speed, flow_speed, device)

    @staticmethod
    def make_arc(n_points, radius, scale, printing_speed, flow_speed,
                 device="cuda", x_max=None):
        waypoints = generate_arc_toolpath(n_points, radius, scale, x_max)
        return Toolpath(waypoints, printing_speed, flow_speed, device)

    @staticmethod
    def make_sinusoid(n_points, amplitude, wavelength, scale, printing_speed,
                      flow_speed, device="cuda", x_max=None):
        waypoints = generate_sinusoidal_toolpath(
            n_points, amplitude, wavelength, scale, x_max
        )
        return Toolpath(waypoints, printing_speed, flow_speed, device)

    @staticmethod
    def make_2layer(printing_speed, flow_speed, length=10.0, layer_gap=0.2,
                    n_points_per_layer=2, device="cuda"):
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
        #waypoints[:, 0] -= 0.03
        #waypoints[:, 0] *= 1.3
        return Toolpath(waypoints, printing_speed, flow_speed, device,
                        start_tangent=np.array([1.0, 0.0, 0.0], dtype=np.float32))


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


def get_sinusoidal_start_tangent(amplitude, wavelength):
    """Unit tangent at x=0 of y = A*sin(2pi*x/lam), z=0."""
    slope = 2 * np.pi * amplitude / wavelength
    d     = np.array([1.0, slope, 0.0], dtype=np.float32)
    return d / np.linalg.norm(d)

@staticmethod
def make_circle(printing_speed, flow_speed, radius=5.0, n_points=100, device="cuda"):
    """
    Full circle in the x-y plane at z=0.
    Starts and ends at (radius, 0, 0), traveling counter-clockwise.
    """
    waypoints = generate_two_layer_circle_toolpath(radius, 2*0.01,n_points)
    start_tangent = np.array([0.0, 1.0, 0.0], dtype=np.float32)  # tangent at angle=0 is +y
    return Toolpath(waypoints, printing_speed, flow_speed, device, start_tangent=start_tangent)

@staticmethod
def make_spiral(printing_speed, flow_speed, outer_radius=10.0, inner_radius=0.0,
                n_turns=3.0, n_points=300, clockwise=True, device="cuda"):
    """
    Outside-in spiral toolpath in the x-y plane at z=0.
    Starts at (outer_radius, 0, 0) and spirals inward to inner_radius.
    """
    waypoints = generate_spiral_toolpath(
        outer_radius, inner_radius, n_turns, n_points, clockwise
    )
    
    # Approximate the initial tangent from the first segment
    d = waypoints[1] - waypoints[0]
    start_tangent = d / np.linalg.norm(d)
    
    return Toolpath(waypoints, printing_speed, flow_speed, device, start_tangent=start_tangent)

@staticmethod
def make_square(printing_speed, flow_speed, side=5.0, corner_radius=0.5,
                n_points=200, device="cuda"):
    """
    Square with rounded corners in the x-y plane at z=0.
    Centered at origin. corner_radius must be < side / 2.
    """
    waypoints = generate_two_layer_rounded_square_toolpath(side, corner_radius, 2*0.01, n_points)
    start_tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return Toolpath(waypoints, printing_speed, flow_speed, device, start_tangent=start_tangent)

@staticmethod
def make_triangle(printing_speed, flow_speed, side=5.0, corner_radius=0.5,
                  n_points=200, device="cuda"):
    """
    Equilateral triangle with rounded corners in the x-y plane at z=0.
    Centered at origin. corner_radius must be < side / (2 * sqrt(3)).
    """
    waypoints = generate_two_layer_rounded_triangle_toolpath(side, corner_radius, 2*0.01, n_points)
    start_tangent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return Toolpath(waypoints, printing_speed, flow_speed, device, start_tangent=start_tangent)

@staticmethod
def make_from_file(filepath, scale, printing_speed, flow_speed, device="cuda"):
    """
    Creates a Toolpath from a custom .dat file containing X, Y, Z coordinates.
    """
    waypoints = generate_from_file(filepath, scale)
    return Toolpath(waypoints, printing_speed, flow_speed, device)


# ----------------------------------------------------------------------
# Waypoint generators
# ----------------------------------------------------------------------

def generate_circle_toolpath(radius: float, n_points: int) -> np.ndarray:
    """
    Full counter-clockwise circle of given radius in the x-y plane.
    Closes back to the start point so the toolpath is a complete loop.
    """
    theta = np.linspace(0.0, 2 * np.pi, n_points, endpoint=True, dtype=np.float32)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1)

def generate_two_layer_circle_toolpath(radius: float, layer_gap: float, n_points_per_layer: int) -> np.ndarray:
    """
    3D two-layer circle toolpath.
    
    Instead of tracing backwards, this completes the first layer counter-clockwise, 
    moves straight up by `layer_gap`, and traces the second layer in the exact 
    same counter-clockwise direction.
    """
    # 1. Generate the base layer (Z=0)
    layer1 = generate_circle_toolpath(radius, n_points_per_layer)
    
    # layer1 finishes exactly where it started. We grab those final X, Y coordinates.
    x_end = layer1[-1, 0]
    y_end = layer1[-1, 1]
    
    # 2. Create a transition point that moves purely up in Z
    transition = np.array([[x_end, y_end, layer_gap]], dtype=np.float32)
    
    # 3. Generate the second layer by copying layer 1 and updating the Z coordinates
    layer2 = layer1.copy()
    layer2[:, 2] = layer_gap
    
    # 4. Concatenate the segments. 
    # We skip layer2[0] because it is mathematically identical to the transition point.
    return np.concatenate([layer1, transition, layer2[1:]], axis=0)


def generate_spiral_toolpath(outer_radius: float, inner_radius: float, 
                             n_turns: float, n_points: int, 
                             clockwise: bool = True) -> np.ndarray:
    """
    Archimedean spiral starting at outer_radius and moving inward to inner_radius.
    Starting position is always on the positive x-axis (y=0).
    """
    # Define total rotational travel
    theta_start = 0.0
    theta_travel = n_turns * 2 * np.pi
    
    if clockwise:
        theta_end = -theta_travel
    else:
        theta_end = theta_travel

    # Linearly interpolate angle and radius
    theta = np.linspace(theta_start, theta_end, n_points, dtype=np.float32)
    r = np.linspace(outer_radius, inner_radius, n_points, dtype=np.float32)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.zeros_like(x)

    return np.stack([x, y, z], axis=1)


def _arc_segment(cx: float, cy: float, r: float,
                 angle_start: float, angle_end: float, n: int) -> np.ndarray:
    """
    Helper: counter-clockwise arc centered at (cx, cy) from angle_start to angle_end.
    Returns (n, 3) array. Endpoint is included.
    """
    theta = np.linspace(angle_start, angle_end, n, dtype=np.float32)
    x = cx + r * np.cos(theta)
    y = cy + r * np.sin(theta)
    z = np.zeros(n, dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def generate_rounded_square_toolpath(side: float, corner_radius: float,
                                     n_points: int) -> np.ndarray:
    """
    Counter-clockwise square with rounded corners, centered at origin.

    Layout (corners of the inner rectangle, i.e. arc centers):
        top-left     (-h, +h)    arc: π   → 3π/2
        top-right    (+h, +h)    arc: π/2 → π
        bottom-right (+h, -h)    arc: 0   → π/2
        bottom-left  (-h, -h)    arc: 3π/2 → 2π

    where h = side/2 - corner_radius.

    Starting point: bottom-left corner of the bottom edge (after the
    bottom-left arc), traveling right (+x direction).
    """
    assert corner_radius < side / 2, "corner_radius must be < side / 2"

    r = corner_radius
    h = side / 2 - r  # distance from center to each arc center

    # Distribute n_points: 4 straight edges + 4 quarter-circle arcs.
    # Straight-edge length vs arc length to decide point budget.
    straight_len = side - 2 * r          # per edge
    arc_len      = 0.5 * np.pi * r       # per corner (quarter circle)
    total_len    = 4 * straight_len + 4 * arc_len

    n_straight = max(2, round(n_points * straight_len / total_len))
    n_arc      = max(2, round(n_points * arc_len      / total_len))

    def straight(x0, y0, x1, y1):
        xs = np.linspace(x0, x1, n_straight, dtype=np.float32)
        ys = np.linspace(y0, y1, n_straight, dtype=np.float32)
        return np.stack([xs, ys, np.zeros(n_straight, dtype=np.float32)], axis=1)

    segments = [
        # Bottom edge: left → right
        straight(-h, -(h + r), +h, -(h + r)),
        # Bottom-right corner arc: center (+h, -h), 270° → 360°
        _arc_segment(+h, -h, r, -0.5 * np.pi,  0.0,           n_arc),
        # Right edge: bottom → top
        straight(+(h + r), -h,  +(h + r), +h),
        # Top-right corner arc: center (+h, +h), 0° → 90°
        _arc_segment(+h, +h,  r,  0.0,           0.5 * np.pi,  n_arc),
        # Top edge: right → left
        straight(+h, +(h + r),  -h, +(h + r)),
        # Top-left corner arc: center (-h, +h), 90° → 180°
        _arc_segment(-h, +h,  r,  0.5 * np.pi,  np.pi,         n_arc),
        # Left edge: top → bottom
        straight(-(h + r), +h,  -(h + r), -h),
        # Bottom-left corner arc: center (-h, -h), 180° → 270°
        _arc_segment(-h, -h, r,  np.pi,          1.5 * np.pi,  n_arc),
    ]

    # Concatenate, dropping the duplicate junction points between segments
    pts = np.concatenate([seg[:-1] for seg in segments] + [segments[0][:1]], axis=0)
    return pts

def generate_two_layer_rounded_square_toolpath(side: float, corner_radius: float, 
                                               layer_gap: float, n_points_per_layer: int) -> np.ndarray:
    """
    3D two-layer rounded square toolpath.
    
    Instead of tracing backwards, this completes the first layer counter-clockwise, 
    moves straight up by `layer_gap`, and traces the second layer in the exact 
    same counter-clockwise direction.
    """
    # 1. Generate the base layer (Z=0)
    layer1 = generate_rounded_square_toolpath(side, corner_radius, n_points_per_layer)
    
    # layer1 finishes exactly where it started. We grab those final X, Y coordinates.
    x_end = layer1[-1, 0]
    y_end = layer1[-1, 1]
    
    # 2. Create a transition point that moves purely up in Z
    transition = np.array([[x_end, y_end, layer_gap]], dtype=np.float32)
    
    # 3. Generate the second layer by copying layer 1 and updating the Z coordinates
    layer2 = layer1.copy()
    layer2[:, 2] = layer_gap
    
    # 4. Concatenate the segments. 
    # We skip layer2[0] because it is mathematically identical to the transition point.
    return np.concatenate([layer1, transition, layer2[1:]], axis=0)

def generate_two_layer_rounded_triangle_toolpath(side: float, corner_radius: float, 
                                                 layer_gap: float, n_points_per_layer: int) -> np.ndarray:
    """
    3D two-layer rounded triangle toolpath.
    
    This completes the first layer counter-clockwise, moves straight up by `layer_gap`, 
    and traces the second layer backwards (clockwise) to return to the start.
    """
    # 1. Generate the base layer (Z=0)
    layer1 = generate_rounded_triangle_toolpath(side, corner_radius, n_points_per_layer)
    
    # layer1 finishes exactly where it started. We grab those final X, Y coordinates.
    x_end = layer1[-1, 0]
    y_end = layer1[-1, 1]
    
    # 2. Create a transition point that moves purely up in Z
    transition = np.array([[x_end, y_end, layer_gap]], dtype=np.float32)
    
    # 3. Generate the second layer by copying and REVERSING layer 1, then updating Z
    layer2 = layer1.copy()
    layer2[:, 2] = layer_gap
    
    # 4. Concatenate the segments. 
    # We skip layer2[0] because it is mathematically identical to the transition point.
    return np.concatenate([layer1, transition, layer2[1:]], axis=0)

def generate_rounded_triangle_toolpath(side: float, corner_radius: float,
                                       n_points: int) -> np.ndarray:
    """
    Counter-clockwise equilateral triangle with rounded corners, centered at origin.

    The three vertices of the *sharp* triangle point at:
        bottom-left,  bottom-right,  top (apex)

    Starting point: 1/4 of the way along the bottom edge (measured from left to right),
    traveling right (+x).
    """
    R_circ = side / np.sqrt(3)   # circumradius
    raw_verts = np.array([
        [ R_circ * np.cos(np.radians(210)),  R_circ * np.sin(np.radians(210))],  # bottom-left
        [ R_circ * np.cos(np.radians(330)),  R_circ * np.sin(np.radians(330))],  # bottom-right
        [ R_circ * np.cos(np.radians( 90)),  R_circ * np.sin(np.radians( 90))],  # apex
    ], dtype=np.float32)

    inradius = side / (2 * np.sqrt(3))
    assert corner_radius < inradius, \
        f"corner_radius must be < side / (2*sqrt(3)) ≈ {inradius:.4f}"

    # Arc centers: move each vertex toward the centroid (origin)
    # The interior angle is 60°, so the half-angle is 30°.
    inset = corner_radius / np.sin(np.radians(30))
    centroid = raw_verts.mean(axis=0)
    arc_centers = np.array([
        v + inset * (centroid - v) / np.linalg.norm(centroid - v)
        for v in raw_verts
    ], dtype=np.float32)

    arc_span = np.radians(120)

    # Outgoing tangent angle at each vertex
    edge_angles = [0.0, np.radians(120), np.radians(240)]

    # Straight-edge length vs arc length budget
    straight_len = side - 2 * corner_radius / np.tan(np.radians(30))
    arc_len      = corner_radius * arc_span   # per corner
    total_len    = 3 * straight_len + 3 * arc_len

    # Calculate point distribution
    n_straight = max(2, round(n_points * straight_len / total_len))
    n_arc      = max(2, round(n_points * arc_len      / total_len))
    
    # Split point counts for the 1/4 and 3/4 sections of the bottom edge
    n_quarter_1 = max(2, round(n_straight * 0.25))
    n_quarter_3 = max(2, round(n_straight * 0.75))

    def make_edge(v_from, v_to, num_pts):
        """Straight segment between two points."""
        xs = np.linspace(v_from[0], v_to[0], num_pts, dtype=np.float32)
        ys = np.linspace(v_from[1], v_to[1], num_pts, dtype=np.float32)
        return np.stack([xs, ys, np.zeros(num_pts, dtype=np.float32)], axis=1)

    # Pre-calculate arcs and edge start/end points
    arcs = []
    edges_from = []
    edges_to = []

    for i in range(3):
        incoming_angle = edge_angles[(i - 1) % 3]
        arc_start_angle = incoming_angle - np.radians(90)
        arc_end_angle   = arc_start_angle + arc_span
        arcs.append(_arc_segment(
            arc_centers[i, 0], arc_centers[i, 1],
            corner_radius, arc_start_angle, arc_end_angle, n_arc
        ))

        next_i = (i + 1) % 3
        next_incoming  = edge_angles[(next_i - 1) % 3]
        arc_start_next = next_incoming - np.radians(90)

        p_from = arc_centers[i]    + corner_radius * np.array([np.cos(arc_end_angle),   np.sin(arc_end_angle)])
        p_to   = arc_centers[next_i] + corner_radius * np.array([np.cos(arc_start_next), np.sin(arc_start_next)])
        
        edges_from.append(p_from)
        edges_to.append(p_to)

    # The bottom edge is index 0. Let's find the point 1/4 of the way across.
    # edges_from[0] is the left side, edges_to[0] is the right side.
    start_point = edges_from[0] + 0.25 * (edges_to[0] - edges_from[0])

    # Build the sequence starting from 1/4 of the bottom edge
    segments = [
        # 1. 1/4 mark → bottom-right arc (this covers 3/4 of the edge length)
        make_edge(start_point, edges_to[0], n_quarter_3),
        # 2. Bottom-right arc
        arcs[1],
        # 3. Right edge (bottom-right → apex)
        make_edge(edges_from[1], edges_to[1], n_straight),
        # 4. Apex arc
        arcs[2],
        # 5. Left edge (apex → bottom-left)
        make_edge(edges_from[2], edges_to[2], n_straight),
        # 6. Bottom-left arc
        arcs[0],
        # 7. Bottom-left arc → 1/4 mark (this covers 1/4 of the edge length)
        make_edge(edges_from[0], start_point, n_quarter_1)
    ]

    # Concatenate, dropping the duplicate junction points between segments
    pts = np.concatenate([seg[:-1] for seg in segments] + [segments[0][:1]], axis=0)
    return pts.astype(np.float32)

def generate_from_file(filepath: str, scale: float = 1.5) -> np.ndarray:
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
            '''
            if 'X' in line and 'Y' in line and 'Z' in line:
                in_data_section = True
                continue
            '''
            parts = line.split()
            # Ensure we have at least 3 columns to parse (X, Y, Z)
            # We use parts[-3:] to grab the last three values, ignoring the leading '0'
            if len(parts) >= 2:
                try:
                    x = float(parts[-2]) * scale
                    y = float(parts[-1]) * scale
                    z = 0.0#float(parts[-1]) * scale
                    waypoints.append([x, y, z])
                except ValueError:
                    # Skip lines that cannot be converted to floats
                    continue

    if not waypoints:
        raise ValueError(f"No valid coordinate data found in {filepath}")

    return np.array(waypoints, dtype=np.float32)