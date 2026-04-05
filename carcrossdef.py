# rhg_crossing_simulation_LQR_cost.py
# A complete, runnable simulation with a more professional LQR-style cost function.

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import numba

# ==============================================================================
# 1. 模型参数定义 (Parameters)
# ==============================================================================
PARAMS = {
    # 仿真参数
    'dt': 0.2, 'total_sim_time': 12, 'N': 8, 'max_ibr_iters': 5,
    'ibr_tolerance': 1e-2,

    # 车辆物理约束
    'v_max': 15.0, 'a_max': 4.0,

    # 安全与环境约束
    'd_safe': 5.0,
    'x_min': -5.0, 'x_max': 55.0, 'y_min': -5.0, 'y_max': 55.0,

    # [!!! MODIFIED !!!] 目标函数权重矩阵 Q 和 R
    # Q: 状态权重 [x, y, vx, vy]
    'Q_A': np.diag([10.0, 10.0, 0.1, 0.1]),
    'Q_B': np.diag([1.0, 1.0, 0.1, 0.1]),
    # R: 控制权重 [ax, ay]
    'R_A': np.diag([0.5, 0.5]),
    'R_B': np.diag([0.1, 0.1]),
}

# 初始状态 [x_A, y_A, vx_A, vy_A, x_B, y_B, vx_B, vy_B]
# x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0])  # 从静止开始
x0 = np.array([
    0.0, 0.0, 10.0, 10.0,   # A: 初始速度大小 sqrt(5^2+5^2) ≈ 7.1 m/s
    0.0, 50.0, 10.0, -10.0, # B: 初始速度大小 sqrt(8^2+8^2) ≈ 11.3 m/s (B明显更快)
])

# 目标点
target_A = np.array([50.0, 50.0])
target_B = np.array([50.0, 0.0])


# ==============================================================================
# [!!! NEW !!!] 参考轨迹生成函数
# ==============================================================================
def generate_reference_trajectory(start_state, target_pos, total_time, dt):
    """
    生成一条从起点到终点的、匀速的参考状态轨迹。
    """
    num_steps = int(total_time / dt)
    # 状态 [x, y, vx, vy]
    ref_traj = np.zeros((num_steps + 1, 4))

    start_pos = start_state[:2]

    # 计算匀速的速度分量
    total_dist = np.linalg.norm(target_pos - start_pos)
    avg_speed = total_dist / total_time
    direction_vec = (target_pos - start_pos) / total_dist
    ref_vx, ref_vy = avg_speed * direction_vec

    for i in range(num_steps + 1):
        t = i * dt
        ref_traj[i, 0] = start_pos[0] + ref_vx * t  # x_ref
        ref_traj[i, 1] = start_pos[1] + ref_vy * t  # y_ref
        ref_traj[i, 2] = ref_vx  # vx_ref
        ref_traj[i, 3] = ref_vy  # vy_ref

    return ref_traj


# --- 预先计算动力学矩阵和参考轨迹 ---
dt = PARAMS['dt']
A_matrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
B_matrix = np.array([[0.5 * dt ** 2, 0], [0, 0.5 * dt ** 2], [dt, 0], [0, dt]])
# 生成参考轨迹
x_ref_A = generate_reference_trajectory(x0[0:4], target_A, PARAMS['total_sim_time'], dt)
x_ref_B = generate_reference_trajectory(x0[4:8], target_B, PARAMS['total_sim_time'], dt)


# ==============================================================================
# 2. 加速的辅助函数 (Numba JIT Accelerated Helpers)
# ==============================================================================
@numba.jit(nopython=True)
def predict_trajectory_numba(x_current, u_joint_seq, N, A, B):
    # ... (这部分代码无需修改) ...
    x_prediction = np.zeros((N + 1, 8))
    x_prediction[0, :] = x_current
    for k in range(N):
        x_A_k, x_B_k = x_prediction[k, 0:4], x_prediction[k, 4:8]
        u_A_k, u_B_k = u_joint_seq[k, 0:2], u_joint_seq[k, 2:4]
        x_A_next = A @ x_A_k + B @ u_A_k
        x_B_next = A @ x_B_k + B @ u_B_k
        x_prediction[k + 1, 0:4] = x_A_next
        x_prediction[k + 1, 4:8] = x_B_next
    return x_prediction


# [!!! MODIFIED !!!] 使用新的LQR代价函数
@numba.jit(nopython=True)
def calculate_cost_numba(player_id_is_A, x_trajectory, my_actions, x_ref_slice, Q_diag, R_diag):
    total_cost = 0.0
    N = my_actions.shape[1]

    for k in range(N):
        # 状态代价
        if player_id_is_A:
            state_k = x_trajectory[k + 1, 0:4]
        else:
            state_k = x_trajectory[k + 1, 4:8]

        state_error = state_k - x_ref_slice[k, :]
        state_cost = np.sum((state_error * Q_diag) * state_error)  # e.T @ Q @ e
        total_cost += state_cost

        # 控制代价
        action_k = my_actions[:, k]
        control_cost = np.sum((action_k * R_diag) * action_k)  # u.T @ R @ u
        total_cost += control_cost

    return total_cost


# ==============================================================================
# 3. 最佳响应求解器 (Best Response Solver)
# ==============================================================================
# [!!! MODIFIED !!!] 接收参考轨迹作为参数
def solve_best_response(player_id, current_x, opponent_actions_sequence, x_ref_player, params, A, B):
    N = params['N']
    initial_guess = np.zeros(2 * N)

    my_idx_start, opp_idx_start = (0, 4) if player_id == 'A' else (4, 0)
    my_u_idx_start, opp_u_idx_start = (0, 2) if player_id == 'A' else (2, 0)

    if player_id == 'A':
        player_id_is_A = True
        Q_diag, R_diag = np.diag(params['Q_A']), np.diag(params['R_A'])
    else:
        player_id_is_A = False
        Q_diag, R_diag = np.diag(params['Q_B']), np.diag(params['R_B'])

    # 找到当前时间步在参考轨迹中的索引
    current_t_idx = int(round(params.get('current_time', 0) / params['dt']))
    x_ref_slice = x_ref_player[current_t_idx + 1: current_t_idx + 1 + N, :]
    if x_ref_slice.shape[0] < N:  # 如果参考轨迹不够长，则用最后一个点填充
        padding = np.tile(x_ref_slice[-1, :], (N - x_ref_slice.shape[0], 1))
        x_ref_slice = np.vstack([x_ref_slice, padding])

    def objective(my_actions_flat):
        my_actions = my_actions_flat.reshape(2, N)
        u_joint_seq = np.zeros((N, 4))
        u_joint_seq[:, my_u_idx_start:my_u_idx_start + 2] = my_actions.T
        u_joint_seq[:, opp_u_idx_start:opp_u_idx_start + 2] = opponent_actions_sequence.T

        x_traj = predict_trajectory_numba(current_x, u_joint_seq, N, A, B)
        return calculate_cost_numba(player_id_is_A, x_traj, my_actions, x_ref_slice, Q_diag, R_diag)

    # ... [约束定义部分与之前代码相同，此处为简洁省略] ...
    constraints = []

    def predict_for_constr(my_actions_flat, A_mat, B_mat):
        my_actions = my_actions_flat.reshape(2, N)
        u_joint_seq = np.zeros((N, 4))
        u_joint_seq[:, my_u_idx_start:my_u_idx_start + 2] = my_actions.T
        u_joint_seq[:, opp_u_idx_start:opp_u_idx_start + 2] = opponent_actions_sequence.T
        return predict_trajectory_numba(current_x, u_joint_seq, N, A_mat, B_mat)

    for k in range(N):
        def vel_constr(my_actions_flat, k, A_mat, B_mat):
            x_traj = predict_for_constr(my_actions_flat, A_mat, B_mat)
            v_x, v_y = x_traj[k + 1, my_idx_start + 2], x_traj[k + 1, my_idx_start + 3]
            return params['v_max'] ** 2 - (v_x ** 2 + v_y ** 2)

        constraints.append({'type': 'ineq', 'fun': vel_constr, 'args': (k, A, B)})

        def acc_constr(my_actions_flat, k):
            my_actions = my_actions_flat.reshape(2, N)
            a_x, a_y = my_actions[:, k]
            return params['a_max'] ** 2 - (a_x ** 2 + a_y ** 2)

        constraints.append({'type': 'ineq', 'fun': acc_constr, 'args': (k,)})

        def coll_avoid_constr(my_actions_flat, k, A_mat, B_mat):
            x_traj = predict_for_constr(my_actions_flat, A_mat, B_mat)
            pos_my, pos_opp = x_traj[k + 1, my_idx_start:my_idx_start + 2], x_traj[k + 1,
                                                                            opp_idx_start:opp_idx_start + 2]
            return np.sum((pos_my - pos_opp) ** 2) - params['d_safe'] ** 2

        # constraints.append({'type': 'ineq', 'fun': coll_avoid_constr, 'args': (k, A, B)})

    result = optimize.minimize(objective, initial_guess, method='SLSQP', constraints=constraints)
    return result.x.reshape(2, N)


# ==============================================================================
# 4. RHG 求解器 (Game Solver)
# ==============================================================================
# [!!! MODIFIED !!!] 接收参考轨迹
def solve_rhg_step(current_x, current_time, params, A, B, x_ref_A, x_ref_B):
    N = params['N']
    u_seq_A = np.zeros((2, N))
    u_seq_B = np.zeros((2, N))

    # 传递当前时间给 `solve_best_response`
    params['current_time'] = current_time

    for i in range(params['max_ibr_iters']):
        u_seq_A_old = u_seq_A.copy()
        u_seq_A = solve_best_response("A", current_x, u_seq_B, x_ref_A, params, A, B)
        u_seq_B = solve_best_response("B", current_x, u_seq_A, x_ref_B, params, A, B)
        diff = np.linalg.norm(u_seq_A - u_seq_A_old)
        if diff < params['ibr_tolerance']:
            break

    return u_seq_A, u_seq_B


# ==============================================================================
# 5. 主仿真循环 (Main Simulation Loop)
# ==============================================================================
def run_simulation(x_ref_A, x_ref_B):
    num_steps = int(PARAMS['total_sim_time'] / PARAMS['dt'])
    x_history = np.zeros((num_steps + 1, 8))
    u_history = np.zeros((num_steps, 4))
    x_history[0, :] = x0

    print("Starting simulation with LQR-style cost... This will take a long time.")
    start_time = time.time()

    for t in range(num_steps):
        current_x = x_history[t, :]
        current_time = t * PARAMS['dt']

        u_seq_A, u_seq_B = solve_rhg_step(current_x, current_time, PARAMS, A_matrix, B_matrix, x_ref_A, x_ref_B)

        u_current_A, u_current_B = u_seq_A[:, 0], u_seq_B[:, 0]
        u_current = np.hstack([u_current_A, u_current_B])

        x_next = predict_trajectory_numba(current_x, u_current.reshape(1, 4), 1, A_matrix, B_matrix)[1, :]

        x_history[t + 1, :] = x_next
        u_history[t, :] = u_current

        elapsed = time.time() - start_time
        print(f"Step {t + 1}/{num_steps} | Time elapsed: {elapsed:.2f}s")

    total_time = time.time() - start_time
    print(f"Simulation finished in {total_time:.2f} seconds.")
    return x_history, u_history


# ==============================================================================
# 6. 结果可视化 (Visualization)
# ==============================================================================
# [!!! MODIFIED !!!] 绘制参考轨迹
# def plot_results(x_history, u_history, params, x_ref_A, x_ref_B):
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))
#
#     # ... [Panel 1, 2, 3 的代码与之前版本几乎一样, 增加了参考轨迹的绘制] ...
#     # Panel 1: Trajectories
#     ax1.set_title("Vehicle Trajectories")
#     ax1.plot(x_ref_A[:, 0], x_ref_A[:, 1], 'r:', label='Reference A', alpha=0.7)  # 绘制A的参考线
#     ax1.plot(x_ref_B[:, 0], x_ref_B[:, 1], 'b:', label='Reference B', alpha=0.7)  # 绘制B的参考线
#     # ... [此处省略与上一版相同的绘图代码] ...
#     ax1.set_xlabel("X position (m)"), ax1.set_ylabel("Y position (m)")
#     ax1.set_xlim(params['x_min'], params['x_max']), ax1.set_ylim(params['y_min'], params['y_max'])
#     ax1.set_aspect('equal', adjustable='box'), ax1.grid(True)
#     pos_A_x, pos_A_y = x_history[:, 0], x_history[:, 1]
#     pos_B_x, pos_B_y = x_history[:, 4], x_history[:, 5]
#     ax1.plot(pos_A_x, pos_A_y, '-', label='Trajectory A (Ego)', color='red')
#     ax1.plot(pos_B_x, pos_B_y, '-', label='Trajectory B (Surrounding)', color='blue')
#     ax1.plot(target_A[0], target_A[1], 'r*', markersize=20, label='Target A')
#     ax1.plot(target_B[0], target_B[1], 'b*', markersize=20, label='Target B')
#     ax1.legend()
#
#     # Panel 2 & 3
#     t_array = np.arange(x_history.shape[0]) * params['dt']
#     vel_A = np.linalg.norm(x_history[:, 2:4], axis=1)
#     vel_B = np.linalg.norm(x_history[:, 6:8], axis=1)
#     ax2.set_title("Vehicle Speeds"), ax2.set_xlabel("Time (s)"), ax2.set_ylabel("Speed (m/s)")
#     ax2.plot(t_array, vel_A, label='Speed A', color='red'), ax2.plot(t_array, vel_B, label='Speed B', color='blue')
#     ax2.grid(True), ax2.legend(), ax2.set_ylim(bottom=0)
#
#     distances = np.linalg.norm(x_history[:, 0:2] - x_history[:, 4:6], axis=1)
#     ax3.set_title("Distance Between Vehicles"), ax3.set_xlabel("Time (s)"), ax3.set_ylabel("Distance (m)")
#     ax3.plot(t_array, distances, label='Distance A-B', color='green')
#     ax3.axhline(y=params['d_safe'], color='orange', linestyle='--', label=f'Safe Distance ({params["d_safe"]:.1f}m)')
#     ax3.grid(True), ax3.legend(), ax3.set_ylim(bottom=0)
#
#     plt.tight_layout(), plt.show()

#%%
# def plot_results(x_history, u_history, params, x_ref_A, x_ref_B):
#     """
#     Generates and saves a two-panel plot suitable for posters.
#     - Panel 1: Vehicle trajectories.
#     - Panel 2: Distance between vehicles.
#     The speed plot has been removed.
#     """
#
#     # --- Poster Quality Enhancements ---
#     plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
#
#     # Create a 1x2 figure layout, removing the middle speed plot.
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
#
#     # --- Panel 1: Trajectories ---
#     ax1.set_title("Vehicle Trajectories", fontsize=20, weight='bold')
#
#     # Plot reference lines (dotted, thicker)
#     ax1.plot(x_ref_A[:, 0], x_ref_A[:, 1], 'r:', label='Reference A', alpha=0.7, linewidth=2.0)
#     ax1.plot(x_ref_B[:, 0], x_ref_B[:, 1], 'b:', label='Reference B', alpha=0.7, linewidth=2.0)
#
#     # Plot actual trajectories (solid, thicker)
#     pos_A_x, pos_A_y = x_history[:, 0], x_history[:, 1]
#     pos_B_x, pos_B_y = x_history[:, 4], x_history[:, 5]
#     ax1.plot(pos_A_x, pos_A_y, '-', label='Trajectory A (Ego)', color='red', linewidth=2.5)
#     ax1.plot(pos_B_x, pos_B_y, '-', label='Trajectory B (Other)', color='blue', linewidth=2.5)
#
#     # Plot targets (large markers)
#     ax1.plot(target_A[0], target_A[1], 'r*', markersize=20, label='Target A', markeredgecolor='black')
#     ax1.plot(target_B[0], target_B[1], 'b*', markersize=20, label='Target B', markeredgecolor='black')
#
#     # Set labels, limits, and grid (larger fonts, subtle grid)
#     ax1.set_xlabel("X position (m)", fontsize=16, weight='bold')
#     ax1.set_ylabel("Y position (m)", fontsize=16, weight='bold')
#     ax1.set_xlim(params['x_min'], params['x_max'])
#     ax1.set_ylim(params['y_min'], params['y_max'])
#     ax1.set_aspect('equal', adjustable='box')
#     ax1.grid(True, linestyle=':', alpha=0.6)
#     ax1.legend(fontsize=14)
#     # ax1.legend(fontsize=14, loc='lower right')
#     ax1.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc='upper left')
#
#     # --- Panel 2: Distance Between Vehicles ---
#     t_array = np.arange(x_history.shape[0]) * params['dt']
#     distances = np.linalg.norm(x_history[:, 0:2] - x_history[:, 4:6], axis=1)
#
#     ax2.set_title("Safety Distance Monitoring", fontsize=20, weight='bold')
#
#     # Plot distance line (thicker)
#     ax2.plot(t_array, distances, label='Distance A-B', color='green', linewidth=3.0)
#
#     # Plot safe distance threshold (dashed, thicker)
#     safe_dist_label = f'Safe Distance ({params["d_safe"]:.1f}m)'
#     ax2.axhline(y=params['d_safe'], color='orange', linestyle='--', label=safe_dist_label, linewidth=2.5)
#
#     # Set labels, limits, and grid (larger fonts, subtle grid)
#     ax2.set_xlabel("Time (s)", fontsize=16, weight='bold')
#     ax2.set_ylabel("Distance (m)", fontsize=16, weight='bold')
#     ax2.grid(True, linestyle=':', alpha=0.6)
#     ax2.legend(fontsize=14)
#     ax2.set_ylim(bottom=0)
#     ax2.set_xlim(left=0, right=params['total_sim_time'])
#
#     # --- Finalize and Save ---
#     plt.tight_layout(pad=3.0)
#     plt.savefig("rhg_crossing_poster.png", dpi=300, bbox_inches='tight')
#     plt.show()



#%%

# ==============================================================================
# 6. 结果可视化 (Visualization) - 最终简洁版
# ==============================================================================
def plot_results(x_history, u_history, params, x_ref_A, x_ref_B):
    """
    生成一个简洁的、左右布局的双面板海报图。
    - 左侧: 安全距离监控。
    - 右侧: 车辆轨迹，图例在图框外部。
    """

    # --- 全局海报质量增强设置 ---
    plt.rcParams.update({'font.size': 30, 'font.weight': 'normal'})

    # --- 创建一个 1x2 的图组布局 ---
    # 稍微加宽尺寸以便为外部图例留出空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 8))

    # --- Panel 1 (Left): Safety Distance Monitoring ---
    ax1.set_title("Safety Distance Monitoring", fontsize=30)

    # 准备时间数据
    t_array = np.arange(x_history.shape[0]) * params['dt']
    distances = np.linalg.norm(x_history[:, 0:2] - x_history[:, 4:6], axis=1)

    # 绘制距离曲线 (加粗)
    ax1.plot(t_array, distances, label='Distance A-B', color='green', linewidth=4)

    # 绘制安全距离阈值线 (加粗虚线)
    safe_dist_label = f'Safe Distance ({params["d_safe"]:.1f}m)'
    ax1.axhline(y=params['d_safe'], color='orange', linestyle='--', label=safe_dist_label, linewidth=3)

    # 设置标签、限制和网格
    ax1.set_xlabel("Time (s)", fontsize=30)
    ax1.set_ylabel("Distance (m)", fontsize=30)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(fontsize=30)
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(left=0, right=params['total_sim_time'])

    # --- Panel 2 (Right): Vehicle Trajectories ---
    ax2.set_title("Vehicle Trajectories", fontsize=28)

    # 绘制参考线 (虚线)
    ax2.plot(x_ref_A[:, 0], x_ref_A[:, 1], 'r:', label='Reference A', alpha=0.7, linewidth=3)
    ax2.plot(x_ref_B[:, 0], x_ref_B[:, 1], 'b:', label='Reference B', alpha=0.7, linewidth=3)

    # 绘制实际轨迹 (实线)
    ax2.plot(x_history[:, 0], x_history[:, 1], '-', label='Trajectory A (Ego)', color='red', linewidth=4)
    ax2.plot(x_history[:, 4], x_history[:, 5], '-', label='Trajectory B (Other)', color='blue', linewidth=4)

    # 绘制目标点 (大号星星)
    # 假设 target_A, target_B 已在全局定义
    ax2.plot(target_A[0], target_A[1], 'r*', markersize=30, label='Target A', markeredgecolor='black')
    ax2.plot(target_B[0], target_B[1], 'b*', markersize=30, label='Target B', markeredgecolor='black')

    # 设置标签、限制和网格
    ax2.set_xlabel("X position (m)", fontsize=30)
    ax2.set_ylabel("Y position (m)", fontsize=30)
    ax2.set_xlim(params['x_min'], params['x_max'])
    ax2.set_ylim(params['y_min'], params['y_max'])
    ax2.set_aspect('equal', adjustable='box')
    ax2.grid(True, linestyle=':', alpha=0.6)

    # [!!! MODIFIED !!!] 将图例放置在图框右侧外部
    ax2.legend(fontsize=30, bbox_to_anchor=(1.03, 1), loc='upper left')

    # --- 完成并保存图像 ---
    # tight_layout 可能会与外部图例冲突，使用 bbox_inches='tight' 来确保所有元素都被保存
    fig.tight_layout()
    plt.savefig("rhg_crossing_final_layout.png", dpi=300, bbox_inches='tight')
    plt.show()
# ==============================================================================
# 7. 主程序入口
# ==============================================================================
if __name__ == '__main__':
    x_hist, u_hist = run_simulation(x_ref_A, x_ref_B)
    plot_results(x_hist, u_hist, PARAMS, x_ref_A, x_ref_B)