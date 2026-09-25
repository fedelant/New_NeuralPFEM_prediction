import torch

# Example index
example_i = None

# Current state
nozzle_points = None
nozzle_ids = None
n_nozzle_nodes = None
position = None
tags = None
velocity = None
pressure = None
cells = None
free_surf = None
node_layer = None
active = None

# History buffers
prev_velocities = None
prev_pressures = None

new_node_indices = None
last_new_vel = None
last_new_press = None

# Input parameters
parameters = None
nozzle_radius = None
nozzle_height = None
printing_velocity = None
flow_velocity = None
density = None
yield_stress = None
viscosity = None

# Toolpath
toolpath = None
phase1 = None
h0 = None
was_z_transition = None
second_layer_init_done = None
z_floor = None

# Cuda device
device = None

# config variables
cfg = None

# Surrogate model
model = None

# Dataset statistics
vel_mean = None
vel_std = None
press_mean = None
press_std = None
pos_min = None
pos_max = None