import matplotlib.pyplot as plt
import numpy as np

def plotResults(x_traj, u_traj, pm):
    """
    可视化两车跟驰的 RHG 控制结果。
    - x_traj: 状态轨迹 (nx*M, T+1)
    - u_traj: 控制轨迹 (nu*M, T)
    - pm: 参数结构体，含 nx, nu, M, delta_t, control_hor
    """

    T = pm.control_hor
    time = np.arange(T+1) * pm.delta_t
    time_u = np.arange(T) * pm.delta_t

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # --- 1. 位置轨迹 ---
    for v in range(pm.M):
        p = x_traj[v * pm.nx, :]
        axs[0].plot(time, p, label=f"Agent {v+1}")
    axs[0].set_ylabel("Position [m]")
    axs[0].legend()
    axs[0].grid(True)

    # --- 2. 速度轨迹 ---
    for v in range(pm.M):
        v_val = x_traj[v * pm.nx + 1, :]
        axs[1].plot(time, v_val, label=f"Agent {v+1}")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].grid(True)

    # --- 3. 控制输入轨迹 ---
    for v in range(pm.M):
        u_val = u_traj[v * pm.nu, :]
        axs[2].step(time_u, u_val, where='post', label=f"Agent {v+1}")
    axs[2].set_ylabel("Acceleration [m/s²]")
    axs[2].set_xlabel("Time [s]")
    axs[2].grid(True)

    plt.suptitle("RHG-Controlled Following Vehicles Trajectory")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
