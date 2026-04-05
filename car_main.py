# ============================ ====================================
# Toy_main_two_car.py  ——  Two‑car following RHG demo
# ================================================================
"""Run a receding‑horizon game (RHG) for two vehicles in a same‑lane
car‑following scenario.  It uses:
    • Toy_parameters.py         – stores A, B, cost weights, limits
    • Transition Matrices Rhg   – prediction‑matrix builder
    • constraint_matrices_rhg   – G, h construction (local + coupling)
    • optimizer_fbrs.py         – fast GNE solver from original repo
"""
import numpy as np
from scipy.linalg import block_diag
import car_parameter as pm
from car_matrices import transitionMatrices
from car_matrices import constraintMatrices
from optimizer_fbrs import fbrs      # original solver

# ------------------------- helper ---------------------------------

def stack_state(xA, xB):
    """[pA,vA,pB,vB] col‑vector."""
    return np.vstack([xA.reshape(-1, 1), xB.reshape(-1, 1)])

# ----------------------- simulation setup -------------------------
# ---- 数值上下限（可以随场景调整）----
pm.v_min   = 0.0        # 最小速度 (m/s)
pm.v_max   = 30.0       # 最大速度
pm.a_min   = -3.0       # 最小加速度 (m/s²)  —— 制动
pm.a_max   =  2.0       # 最大加速度       —— 踩油门
pm.d_safe  = 1.5        # 安全距离 (m)


np.set_printoptions(precision=3, suppress=True)

# initial states  (you can change)
xA0 = np.array([0.0,  10.0])   # car A starts at 0 m, 10 m/s
xB0 = np.array([-15.0,  8.0])  # car B 15 m behind, slower

x0_concat = stack_state(xA0, xB0)    # (4,1)

# total simulation steps
SIM_STEPS = 50

# storage for plotting
hist_xA, hist_xB = [xA0], [xB0]

# ----------------------- main RHG loop ----------------------------
for t in range(SIM_STEPS):
    # 1) Build transition matrices for current horizon
    Av, Av_tilde, Bv, Bv_tilde, Bv_hat = transitionMatrices(pm)

    # 2) Build cost matrices (simple quadratic)
    #    F_u = B^T W B + R ,  F_x = B^T W A
    W_big = block_diag(*([pm.W[0]] * pm.N * pm.M))   # identical W for all agents
    R_big = block_diag(*([pm.R[0]] * pm.N * pm.M))

    Btilde_blk = block_diag(*Bv_tilde)                # (2MN × N M)
    Atilde_blk = block_diag(*Av_tilde)

    Fu = Btilde_blk.T @ W_big @ Btilde_blk + R_big
    Fx = Btilde_blk.T @ W_big @ Atilde_blk

    H = Fu
    # F_mat =  Fx
    # f = F_mat @ x0_concat.flatten()  # (20,)
    f = Fx @ x0_concat.flatten()

    # 3) Build inequality GU ≤ h  (depends on x0)
    G, h = constraintMatrices(pm, Av_tilde, Bv_tilde)

    # ---- fill RHS with current x0 --------------------------------
    # velocity constraints
    S_v_block = np.array([0, 1])
    S_v = block_diag(*([S_v_block]*pm.N))
    row = 0
    for v in range(pm.M):
        vel_from_x0 = S_v @ (Av_tilde[v] @ x0_concat[v*pm.nx:(v+1)*pm.nx])
        h[row           : row+pm.N] =  pm.v_max - vel_from_x0.flatten()
        h[row+pm.N      : row+2*pm.N] =  -pm.v_min + vel_from_x0.flatten()
        row += 2*pm.N

    # acceleration RHS (constant)
    h[row           : row+pm.M*pm.N]       =  pm.a_max
    h[row+pm.M*pm.N : row+2*pm.M*pm.N]     = -pm.a_min
    row += 2*pm.M*pm.N

    # coupling RHS  pA - pB ≥ d_safe
    S_p_block = np.array([1, 0]);  S_p = block_diag(*([S_p_block]*pm.N))
    pA_x0 = S_p @ (Av_tilde[0] @ x0_concat[0:2])
    pB_x0 = S_p @ (Av_tilde[1] @ x0_concat[2:4])
    h[row:row+pm.N] = -pm.d_safe + pA_x0.flatten() - pB_x0.flatten()

    # 4) Solve RHG (variational GNE)   — original fast solver
    dim_u = pm.M * pm.N * pm.nu  # 总决策变量长度
    dim_g = G.shape[0]  # 约束行数

    # u0 = np.zeros(pm.M * pm.N * pm.nu)
    # v0 = np.zeros(G.shape[0])

    u0 = np.zeros(H.shape[0])  # (20,)
    v0 = np.zeros(G.shape[0])  # 行数≈90
    # ---- 4) 调用 fbrs ------------------------------------------
    U_opt, info = fbrs(H, f, G, h, u0, v0)  # 6 实参

    # u0 = np.zeros(dim_u)  # 全 0 初猜
    # v0 = np.zeros(dim_g)  # 乘子初猜

    # U_opt, info = fbrs(Fu, f, G, h, u0, v0)

    # U_opt, _ = fbrs(Fu, f, G, h)

    # 5) Extract first inputs
    uA0 = U_opt[0]                 # first control of car A
    uB0 = U_opt[pm.N]              # first control of car B (because nu=1)

    # 6) Propagate one step
    xA0 = Av[0] @ xA0 + Bv[0].flatten() * uA0
    xB0 = Av[1] @ xB0 + Bv[1].flatten() * uB0
    x0_concat = stack_state(xA0, xB0)

    hist_xA.append(xA0)
    hist_xB.append(xB0)

# ------------------ simple text output ----------------------------
print("Final positions / speeds:\nA:", hist_xA[-1], "\nB:", hist_xB[-1])
