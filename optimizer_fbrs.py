"""
Copyright 2024 ETH Zurich, Sophie Hall

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
"""

  @Author: Sophie Hall
  @Last modified: May 8 2024

  fbrs: A regularized and smoothed Fischer-Burmeister Method for 
  solving convex quadratic programs (QPs)
  [x,v,out] = fbrs(qp,x0,v0,opts) attempts to solve
  following problem

  min.  H x + f
  s.t.  Ax <= b
  
  FBRS algorithm for non symmetric H
  Implemented using numpy
  
  
  This method assumes the linear independance constraint qualification
  and strong second order sufficient conditions hold at the solution
  but often converges anyways even if they do not
  
  Inputs: 
  qp: A structure with the following fields
 	H: n x n Symmetric positive semidefinite Hessian matrix
 	f: n x 1 Gradient vector
 	A: q x n Constraint Jacobian
 	b: q x 1 Constraint vector
  x0: n x 1 Primal initial guess, use 0 if unsure
  v0: q x 1 Dual initial guess, use 0 if unsure
  opts: A structure containing any of the following fields
 	tol: Absolute tolerance
 	rtol: Relative tolerance
 	beta: backtracking linesearch parameter 
 	eta: sufficient decrease parameter
 	delta: Regularization parameter
 	lsmax: maximum number of linesearch iterations
 	max_newton_iters: maximum number of newton iterations
 	veps: smoothing parameter
 	use_nr: true to use the natural residual as a stopping criteria
  
  A default value will be used for any missing fields

  Outputs:
  x: n x 1 Primal solution
  v: q x 1 Dual solution
  out: Structure with the following fields
  prox_iters: number of proximal iterations taken
  newton_iters: total number of newton iterations
  res:  KKT residual
  eflag: Exit flag
 		 0: success
  		-1: Maximum number of iterations exceeded

"""

import numpy as np
from scipy import linalg


# evaluate the FB function
# 平滑Fischer-Burmeister函数（用于近似互补条件）
def fb(u,v,veps):
    """
        计算平滑Fischer-Burmeister函数值 (式8的数值处理)

        参数:
            u (np.array): 原始变量（控制输入序列）
            v (np.array): 对偶变量（约束乘子）
            veps (float): 平滑参数（避免不可导点）

        返回:
            phi (np.array): FB函数值向量
        数学形式:
            φ(u,v) = u + v - √(u² + v² + ε²)
        物理意义:
            当u≥0, v≥0且u*v=0时，φ(u,v)=0 对应KKT条件
    """
    phi = (u+v) - np.sqrt(u**2 + v**2 + veps**2)
    
    return phi

# derivative of the smoothed fb function
# taken w.r.t its first argument
# the Jacobian is diagonal so its returned as a vector
# FB函数对u的偏导数（雅可比矩阵对角元素）
def dfb(u,v,veps):
    """
        计算FB函数对u的偏导数（用于牛顿法雅可比矩阵）

        参数:
            u, v, veps: 同fb函数
        返回:
            gamma (np.array): ∂φ/∂u 的向量
            mu (np.array): ∂φ/∂v 的向量
        注意:
            雅可比矩阵为对角矩阵，故以向量形式返回对角元素
    """
    q = np.max(u.shape)  # 约束维度
    gamma = np.zeros([q, 1])  # 初始化∂φ/∂u
    mu = np.zeros([q, 1])  # 初始化∂φ/∂v

    # 计算分母项 r = √(u²+v²+ε²)
    r = (u ** 2 + v ** 2 + veps ** 2) ** (0.5)
    idx = r > 100 * 2.2204e-16  # 数值稳定性阈值

    # 在稳定区域计算解析导数
    gamma[idx] = (1 - u / r[idx]).reshape(-1, 1)
    mu[idx] = (1 - v / r[idx]).reshape(-1, 1)

    # 在接近奇点区域使用广义雅可比
    gamma[~idx] = 1 - 1 / np.sqrt(2)
    mu[~idx] = 1 - 1 / np.sqrt(2)

    return gamma, mu


#%% 主求解器：Fischer-Burmeier正则化方案
def fbrs(H,f,Xi_mat,xi_vec,u0,v0):
    """
        求解广义纳什均衡（GNEP）的FB正则化算法（对应论文式8）

        参数:
            H (np.array): 伪梯度Hessian近似 Fu (式30)
            f (np.array): 线性项 Fx*x (式12)
            Xi_mat (np.array): 耦合约束矩阵 Ξ
            xi_vec (np.array): 约束右端项 ξ
            u0 (np.array): 控制输入初值
            v0 (np.array): 对偶变量初值

        返回:
            u (np.array): 最优控制序列
            v (np.array): 最优对偶变量
            E (float): 最终残差
            out (dict): 求解器状态信息
        """
    # ========== 参数配置 ========== #
    # sizes
    n = np.shape(f)[0]
    q = xi_vec.size
    
	# default parameters
    tol = 1e-8 # absolute error tolerance
    rtol = 1e-10 #  relative error tolerance
    eta = 1e-10 # linesearch sufficient decrease parameter
    beta = 0.7 # backtracking parameter
    delta = np.sqrt(2.2204e-16) # regularization strength
    veps =  tol/(2*np.sqrt(q)) * np.ones(v0.shape)  # smoothing strength
    lsmax = 10; # max number of linesearch iterations
    max_newton_iters = 100 # max number of newton iterations
    use_nr = False # use the natural residual as a stopping criteria

    
    # initialize
    u = u0.copy()
    y = xi_vec - Xi_mat @ u.copy()
    v = v0.copy()
    
    E = 1e10
    E0 = E
    newton_iters = 0
    out = {}
    out['eflag'] = -1
    
    
    d = np.zeros([q,1])
    L = np.zeros(H.shape)
    E = 1e10
    j = 0

    # ========== 牛顿法主循环 ========== #
    for j in range(max_newton_iters):
        #print(j)

        # update reg strength
        # 动态调整正则化强度 (与当前残差关联)
        delta = min(delta,E)

        # 计算FB函数导数 (构建雅可比矩阵)
        # compute fb derivatives
        [gamma,mu] = dfb(y,v,veps)
        d = mu + delta  # 正则化分母项

            
        # eval rhs  # 计算当前残差 (KKT条件)
        rs = -(H @ u + f + Xi_mat.T @ v) # stationarity 平稳性残差: ∇ₓL = 0
        rc = -fb(y,v,veps);  # complimentarity  互补残差: φ(y,v) = 0
        
        # convergence check
        F = np.concatenate([rs , rc ], axis=0)  # 总残差向量
        
        if use_nr:
            E = 1 #linalg.norm(np.concatenate([rs, min(y,v)]))
           
        else:
            E = linalg.norm(F);
		
        if j == 0:
            E0 = E
        
        if E <= tol + E0*rtol:
            out['eflag'] = 0
			
            break

        B = np.zeros([q,n])

        for i in range(q):
            
            B[i,:] = 1/d[i] * gamma[i] * (Xi_mat[i,:])     
       
        
        # K = H + A'*diag(c./d)*A;
        K = H + Xi_mat.T @ B

        # 构造右端项 r = rs - Ξᵀ(rc/d)
        r = rs - Xi_mat.T @ (rc/d)
                    
        # newton step, need to do LU decomposition 
        # instead of cholesky as H not symmetric
        # 求解牛顿方程 K·du = r
        P, L, U = linalg.lu(K)
        du = linalg.solve(L,P.T @ r)
        du = linalg.solve(U, du)

        # 更新对偶变量 dv = (rc + γΞdu)/d
        dv = (rc + gamma*(Xi_mat @ du))/d
        dy = -Xi_mat @ du
        
        # Use this code block if H is symmetric
        # L = linalg.cholesky(K,lower = True)
        # du = linalg.solve(L,r)
        # du = linalg.solve(L.T, du)
        # dv = (rc + gamma*(Xi_mat @ du))/d
        # dy = -Xi_mat @ du
        
		# linesearch
        # ====== 回溯线搜索 ====== #
        alpha = 1;
        if lsmax > 0 :
            keep_going = True
            m0 = 1/2* linalg.norm(F)
            l = 0
            
            while keep_going and l < lsmax:
                # 计算候选点
                up = u + alpha*du
                vp = v + alpha*dv
                yp = y + alpha*dy

                # 计算候选残差
                rs = -(H @ up+ f + Xi_mat.T @ vp)
                rc = -fb(yp,vp,veps)
                Fp = np.concatenate([rs,rc], axis=0)
        
                malpha = 1/2*linalg.norm(Fp)

                # Armijo条件: m(α) ≤ (1-2ηα)m₀
                if malpha <= (1-2*eta*alpha)*m0:
                    keep_going = False
                else:
                    alpha = beta*alpha
                    l = l + 1
       
        u += alpha*du
        v += alpha*dv
        y += alpha*dy
        newton_iters = newton_iters + 1
        
    out['newton_iters'] = newton_iters
    out['res'] = E
    out['solver'] = 'fbrs'
    
    return u,v,E,out           

