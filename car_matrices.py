import numpy as np
from scipy.linalg import block_diag

#%%
def build_prediction_matrices(A: np.ndarray, B: np.ndarray, T: int):
    """
    Construct prediction matrices A_bar and B_bar for discrete-time LTI system:
        x[k+1] = A x[k] + B u[k]

    Returns:
        A_bar: (n*T) x n matrix
        B_bar: (n*T) x (T) matrix
    """
    n = A.shape[0]  # state dimension
    A_bar = np.zeros((n * T, n))
    B_bar = np.zeros((n * T, T))

    for i in range(T):
        A_power = np.linalg.matrix_power(A, i + 1)
        A_bar[i * n:(i + 1) * n, :] = A_power

        for j in range(i + 1):
            B_bar[i * n:(i + 1) * n, j] = (np.linalg.matrix_power(A, i - j) @ B).flatten()

    return A_bar, B_bar

def transitionMatrices(pm):
    """Construct system dynamics and prediction matrices for each agent."""
    Av, Bv = [], []
    Av_tilde, Bv_tilde, Bv_hat = [], [], []

    Psi = np.append(np.eye(pm.nu), np.tile(np.zeros([pm.nu, pm.nu]), pm.N - 1), axis=1)

    for v in range(pm.M):
        A = pm.alpha[v, :].reshape(pm.nx, pm.nx)
        B = pm.beta[v, :].reshape(pm.nx, pm.nu)

        Av.append(A)
        Bv.append(B)

        A_bar, B_bar = build_prediction_matrices(A, B, pm.N)
        Av_tilde.append(A_bar)
        Bv_tilde.append(B_bar)

        B_hat = B_bar @ Psi.T
        Bv_hat.append(B_hat)

    return Av, Av_tilde, Bv, Bv_tilde, Bv_hat

#%%
import numpy as np
from scipy.linalg import block_diag

"""
Constraint matrix builder for two‑car following RHG.

Returned objects follow the usual cvxpy/quadprog convention
    G u <= h   (inequality)
    Eu  = e    (equality — not used here)

All numerical limits (speed / acceleration / safety distance) are **not**
hard‑coded — they are read from the pm (parameter) object so that the main
script can modify them on the fly.
"""

def constraintMatrices(pm, Av_tilde, Bv_tilde):
    """Generate block‑diagonal local constraints and a single coupling constraint.

    Parameters
    ----------
    pm : parameter module (already imported as pm in Toy_main)
        Must expose
            .M           – number of agents (2 here)
            .N           – prediction horizon
            .nx, .nu     – state / input dimensions (2,1)
            .v_min, .v_max
            .a_min, .a_max
            .d_safe      – safety distance
    Av_tilde, Bv_tilde : list
        Prediction matrices for every agent (output of transitionMatrices)

    Returns
    -------
    G : 2‑D ndarray
        Inequality matrix so that  G U <= h
    h : 1‑D ndarray
    """
    M, N, nx, nu = pm.M, pm.N, pm.nx, pm.nu

    # ---------- 1. Local constraints -------------------------------------------------
    # Velocity bounds  v_min <= v[k] <= v_max  for every prediction step
    # Extract the velocity row (index 1) from the predicted state  x = A_bar x0 + B_bar U
    #   v_seq = Sv A_tilde x0  +  Sv B_tilde U
    # where  Sv  selects the velocity component from each 2‑state block.

    # Selection matrix that picks velocity from each (p,v) block
    S_v_block = np.array([0, 1])        # 1×2
    S_v = block_diag(*([S_v_block]*(N)))   # (N×2N)

    G_vel_upper, G_vel_lower, h_vel_upper, h_vel_lower = [], [], [], []
    for v in range(M):
        B_bar = Bv_tilde[v]             # (2N × N)
        # Only the part multiplying U matters (x0 part goes to RHS)
        SvB = S_v @ B_bar               # (N × N)
        G_vel_upper.append(  SvB)       #  + v_max >= SvB U + SvA x0
        G_vel_lower.append( -SvB)       #  -(v_min) >= -SvB U - SvA x0

    G_vel_upper = block_diag(*G_vel_upper)
    G_vel_lower = block_diag(*G_vel_lower)

    # Prepare h later in main: here just placeholders of correct length (fill with np.inf)
    h_vel_upper = np.full(G_vel_upper.shape[0],  np.inf)
    h_vel_lower = np.full(G_vel_lower.shape[0],  np.inf)

    # ---------- 2. Acceleration bounds ----------------------------------------------
    # a_min <= u[k] <= a_max  for each agent & step
    I_u = np.eye(N*nu)
    G_acc_upper =  block_diag(*([ I_u ]*M))      #  +a_max >=  U

    G_acc_lower = block_diag(*([-I_u] * M))  # ← 把前面的 - 换成普通 -

    #  -a_min >= -U
    h_acc_upper = np.full(G_acc_upper.shape[0], np.inf)
    h_acc_lower = np.full(G_acc_lower.shape[0], np.inf)

    # ---------- 3. Coupling constraint (safety distance) -----------------------------
    #  p_A[k] - p_B[k] >= d_safe   for every k in prediction horizon
    # Position selector  S_p = [1 0]
    S_p_block = np.array([1,0])
    S_p = block_diag(*([S_p_block]*N))             # (N × 2N)

    # Predicted positions:  p_seq = S_p (A_bar x0 + B_bar U)
    A_bar_A, B_bar_A = Av_tilde[0], Bv_tilde[0]    # agent 0 = A
    A_bar_B, B_bar_B = Av_tilde[1], Bv_tilde[1]    # agent 1 = B

    # Only keep control‑input term to build GU <= h  (x0 term to RHS)
    diff_B =  S_p @ B_bar_B * (-1) +  S_p @ B_bar_A   # N×N for each agent block

    # Coupling constraint involves controls of BOTH agents → need full-column matrix
    # Arrange as [B_A , B_B] order (same as stacking U=[U_A;U_B])
    G_cpl = np.hstack([S_p @ B_bar_A, -S_p @ B_bar_B])  # ← 同上
    h_cpl = np.full(G_cpl.shape[0], -np.inf)   # placeholder; will be RHS‑computed

    # ---------- 4. Stack all inequalities -------------------------------------------
    G = np.vstack([
        G_vel_upper,
        G_vel_lower,
        G_acc_upper,
        G_acc_lower,
        G_cpl
    ])
    h = np.hstack([
        h_vel_upper,
        h_vel_lower,
        h_acc_upper,
        h_acc_lower,
        h_cpl
    ])

    return G, h  # Eu = e not used
