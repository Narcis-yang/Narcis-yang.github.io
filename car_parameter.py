# Parameters for two-car following RHG simulation
import numpy as np

# System dimensions
nx = 2      # state dimension: [position, velocity]
nu = 1      # input dimension: acceleration
M = 2       # number of agents (cars)
N = 10      # prediction horizon
control_hor = 20

# Toggle stability check (False for basic RHG simulation)
stability_check = False

# Discrete-time model parameters
delta_t = 0.1
A = np.array([[1.0, delta_t],
              [0.0, 1.0]])
B = np.array([[0.5 * delta_t**2],
              [delta_t]])

# Flatten and duplicate for each agent
alpha = np.tile(A.flatten(), (M, 1))   # shape: (M, nx*nx)
beta  = np.tile(B.flatten(), (M, 1))   # shape: (M, nx*nu)

# Cost weights: each agent has (position, velocity) state weight and input weight
W = [np.diag([1.0, 0.1]) for _ in range(M)]
R = [np.diag([0.01]) for _ in range(M)]

# Safety distance & 物理约束
d_safe = 50  # meters

v_min   = 0.0        # 最小速度 (m/s)
v_max   = 100/3.6       # 最大速度
a_min   = -3.0       # 最小加速度 (m/s²)  —— 制动
a_max   =  2.0       # 最大加速度       —— 踩油门

