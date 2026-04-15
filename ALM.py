# -*- coding: utf-8 -*-
"""
弱形式 Stokes 形状优化 - 增广拉格朗日方法 (ALM) + Uzawa 算法
修改说明：
1. 引入粘度系数 mu_visc = 0.1
2. 同步等比例调整 ALM 惩罚系数 (缩小10倍)
3. 绘图调整：去掉面积误差图，仅保留流场图与能量耗散演化图
4. 结果图片直接保存到本地目录
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子，保证结果可复现
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. 验证参数设定
# ==========================================
R_INIT = 0.3
AREA_TARGET = np.pi * (R_INIT ** 2)
MU_VISC = 0.1  # 【新增】动力粘度系数

print(f"【Uzawa ALM 形状优化】目标面积约束: {AREA_TARGET:.4f}")
print(f"【流体属性】动力粘度系数: {MU_VISC}")

# ==========================================
# 1. 映射层 (StarMapping - 强制物理对称性)
# ==========================================
class StarMapping(nn.Module):
    def __init__(self, num_fourier=15):
        super().__init__()
        self.num_fourier = num_fourier
        self.register_buffer('k_freq', torch.arange(1, num_fourier + 1).float())
        
        self.a0 = nn.Parameter(torch.tensor([R_INIT])) 
        self.ak = nn.Parameter(torch.zeros(num_fourier))

    def get_obstacle_radius(self, theta):
        theta = theta.view(-1, 1)
        k = self.k_freq.view(1, -1)
        perturb = torch.sum(self.ak * torch.cos(k*theta), dim=1, keepdim=True)
        R_obs = torch.clamp(self.a0 + perturb, min=0.05) 
        return R_obs

    def forward(self, xi):
        xi_r = xi[:, 0:1]
        theta = xi[:, 1:2]
        
        R_obs = self.get_obstacle_radius(theta)
        R_sq = (torch.cos(theta)**8 + torch.sin(theta)**8)**(-1/8)
        
        dr = R_sq - R_obs 
        r_phys = R_obs + xi_r * dr
        x = r_phys * torch.cos(theta)
        y = r_phys * torch.sin(theta)
        return x, y

# ==========================================
# 2. 弱形式 PINN (仅输出速度场 u, v)
# ==========================================
class StokesWeak_PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 80), nn.Tanh(),
            nn.Linear(80, 2) 
        )

    def forward(self, xi, x_phys, y_phys):
        xi_r = xi[:, 0:1]
        theta = xi[:, 1:2]
        
        periodic_features = torch.cat([xi_r, torch.cos(theta), torch.sin(theta)], dim=1)
        out = self.net(periodic_features)
        nn_u = out[:, 0:1]
        nn_v = out[:, 1:2]
        
        u_base = xi_r * (1.0 - y_phys**2)
        outlet_mask = torch.sigmoid(100* (x_phys - 0.95))
        dist_func = xi_r * ((1.0 - xi_r) + xi_r * outlet_mask)
        
        u = u_base + dist_func * nn_u
        v = 0.0 + dist_func * nn_v
        
        return u, v

# ==========================================
# 3. 工具函数：自动求物理域一阶偏导
# ==========================================
def get_phys_grad(f, xi, J11, J12, J21, J22, detJ):
    df_dxi = torch.autograd.grad(f, xi, torch.ones_like(f), create_graph=True)[0]
    df_dr = df_dxi[:, 0:1]
    df_dt = df_dxi[:, 1:2]
    
    f_x = (J22 * df_dr - J21 * df_dt) / detJ
    f_y = (-J12 * df_dr + J11 * df_dt) / detJ
    return f_x, f_y

# ==========================================
# 4. 增广拉格朗日目标函数 (ALM)
# ==========================================
def compute_alm_loss(model, mapper, xi_in, p_mult, lambda_area, mu_div, mu_area, w_reg, mu_visc):
    x, y = mapper(xi_in)
    
    dx_dxi = torch.autograd.grad(x, xi_in, torch.ones_like(x), create_graph=True)[0]
    dy_dxi = torch.autograd.grad(y, xi_in, torch.ones_like(y), create_graph=True)[0]
    J11, J12 = dx_dxi[:, 0:1], dx_dxi[:, 1:2]
    J21, J22 = dy_dxi[:, 0:1], dy_dxi[:, 1:2]
    detJ = J11 * J22 - J12 * J21 + 1e-8 
    
    u, v = model(xi_in, x, y)
    
    u_x, u_y = get_phys_grad(u, xi_in, J11, J12, J21, J22, detJ)
    v_x, v_y = get_phys_grad(v, xi_in, J11, J12, J21, J22, detJ)
    
    # ----------------------------------------------------
    # (A) 物理积分项: Drag + 散度的增广拉格朗日约束
    # ----------------------------------------------------
    # 【修改点】：加入粘度系数 mu_visc 计算物理能量耗散
    energy_dissipation = 0.5 * mu_visc * (u_x**2 + u_y**2 + v_x**2 + v_y**2)
    divergence = u_x + v_y
    
    alm_div = - p_mult * divergence + 0.5 * mu_div * (divergence**2)
    
    integrand = energy_dissipation + alm_div
    
    loss_physics = torch.mean(integrand * detJ) * 2 * np.pi
    obj_drag = torch.mean(energy_dissipation * detJ) * 2 * np.pi
    
    # ----------------------------------------------------
    # (B) 全局几何约束: 面积的增广拉格朗日约束
    # ----------------------------------------------------
    theta_eval = torch.linspace(0, 2*np.pi, 500, device=xi_in.device)[:-1] 
    R_obs_eval = mapper.get_obstacle_radius(theta_eval)
    area_curr = 0.5 * torch.mean(R_obs_eval**2) * 2 * np.pi
    area_diff = area_curr - AREA_TARGET
    
    loss_area = - lambda_area * area_diff + 0.5 * mu_area * (area_diff**2)
    
    # ----------------------------------------------------
    # (C) 表面平滑正则化
    # ----------------------------------------------------
    loss_reg = w_reg * torch.sum((mapper.k_freq ** 2) * (mapper.ak ** 2))
    
    total_alm_loss = loss_physics + loss_area + loss_reg
    
    return total_alm_loss, obj_drag, area_curr, divergence

# ==========================================
# 5. 绘图函数 (仅保留流场图和能量耗散图)
# ==========================================
def plot_stokes_results(mapper, pinn, history_obj, history_area, total_epochs):
    plt.style.use('default')
    # 调整画布大小以适应三个子图
    fig = plt.figure(figsize=(18, 5)) 
    
    # --- 第一个图：物理流场与边界 ---
    ax1 = fig.add_subplot(1, 3, 1)
    with torch.no_grad():
        t = np.linspace(0, 2*np.pi, 200)
        theta_tensor = torch.tensor(t, dtype=torch.float32)
        R_final = mapper.get_obstacle_radius(theta_tensor).numpy().flatten()
        X_obs = R_final * np.cos(t)
        Y_obs = R_final * np.sin(t)
        
        r_grid = np.linspace(0, 1, 80)
        t_grid = np.linspace(0, 2*np.pi, 120)
        R_mesh, T_mesh = np.meshgrid(r_grid, t_grid)
        xi_vis = torch.tensor(np.stack([R_mesh.flatten(), T_mesh.flatten()], axis=1), dtype=torch.float32)
        
        x_phys, y_phys = mapper(xi_vis)
        u_pred, v_pred = pinn(xi_vis, x_phys, y_phys)
        
        X_field = x_phys.reshape(T_mesh.shape).numpy()
        Y_field = y_phys.reshape(T_mesh.shape).numpy()
        Vel_mag = torch.sqrt(u_pred**2 + v_pred**2).reshape(T_mesh.shape).numpy()

    contour = ax1.contourf(X_field, Y_field, Vel_mag, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(contour, ax=ax1, fraction=0.046, pad=0.04, label='Velocity Magnitude |V|')
    
    circle = plt.Circle((0,0), R_INIT, color='g', fill=False, linestyle='--', linewidth=2, label='Initial Circle')
    ax1.add_patch(circle)
    ax1.plot(X_obs, Y_obs, 'k-', linewidth=3, label='Optimized Shape')
    
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.legend(loc='upper right', fontsize=8)
    ax1.set_title(f"Weak Form Stokes Velocity (Iter: {total_epochs})")

    # --- 第二个图：面积迭代变化 ---
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.plot(history_area, label='Area', color='blue')
    ax2.axhline(y=AREA_TARGET, color='r', linestyle='--', linewidth=1.5, label=f'Target ({AREA_TARGET:.4f})')
    ax2.set_title("Area (Volume) History")
    ax2.set_xlabel("Epochs")
    ax2.grid(True)
    ax2.legend()

    # --- 第三个图：物理目标历史 ---
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(history_obj, label='J(u) - Energy Dissipation', color='red')
    ax3.set_title("Physical Objective History (Drag)")
    ax3.set_xlabel("Epochs")
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    save_filename = "stokes_optimization_result.png"
    save_path = os.path.join(os.getcwd(), save_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() 
    
    print(f"\n=========================================")
    print(f"绘图完成！可视化结果已保存至: \n{save_path}")
    print(f"=========================================")

# ==========================================
# 6. 主函数  (Uzawa 交替乘子算法)
# ==========================================
def main():
    mapper = StarMapping(num_fourier=15)
    pinn = StokesWeak_PINN()
    
    n_collo = 4000
    xi_in_fixed = torch.rand(n_collo, 2)
    xi_in_fixed[:, 1] = xi_in_fixed[:, 1] * 2 * np.pi 
    
    opt_nn = torch.optim.Adam(pinn.parameters(), lr=1e-3)
    opt_sh = torch.optim.Adam(mapper.parameters(), lr=3e-4)
    
    history_ju = []
    history_area = []
    
    p_mult = torch.zeros((n_collo, 1), requires_grad=False)
    lambda_area = torch.tensor(0.0, requires_grad=False)   
    
    # 【修改点】由于粘度变为0.1 (物理量缩水10倍)，此处各项惩罚系数全部同步缩小10倍
    MU_DIV = 10.0     # 原为 100.0
    MU_AREA = 1000.0  # 原为 10000.0
    W_REG = 0.01      # 原为 0.1

    def train_nn(n_steps, label=""):
        for p in mapper.parameters(): p.requires_grad_(False)
        for ep in range(n_steps):
            opt_nn.zero_grad()
            xi = xi_in_fixed.clone().detach().requires_grad_(True)
            loss_alm, obj_drag, _, _ = compute_alm_loss(
                pinn, mapper, xi, p_mult, lambda_area, MU_DIV, MU_AREA, W_REG, MU_VISC)
            loss_alm.backward()
            torch.nn.utils.clip_grad_norm_(pinn.parameters(), 1.0)
            opt_nn.step()
            
            if n_steps <= 10 or ep % max(1, n_steps // 4) == 0:
                print(f"  {label}NN [{ep:3d}/{n_steps}] ALM_Loss={loss_alm.item():.4f} J(u)={obj_drag.item():.4f}")
        for p in mapper.parameters(): p.requires_grad_(True)
        return obj_drag.item()

    def train_shape(n_steps, label=""):
        for p in pinn.parameters(): p.requires_grad_(False)
        for ep in range(n_steps):
            opt_sh.zero_grad()
            xi = xi_in_fixed.clone().detach().requires_grad_(True)
            loss_alm, obj_drag, area, _ = compute_alm_loss(
                pinn, mapper, xi, p_mult, lambda_area, MU_DIV, MU_AREA, W_REG, MU_VISC)
            loss_alm.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 0.01)
            opt_sh.step()
            
            history_ju.append(obj_drag.item())
            history_area.append(area.item())
            
            if ep % max(1, n_steps // 5) == 0:
                print(f"  {label}Shape [{ep:2d}/{n_steps}] J(u)={obj_drag.item():.4f} "
                      f"Area={area.item():.4f}({AREA_TARGET:.4f}) a0={mapper.a0.item():.4f}")
        for p in pinn.parameters(): p.requires_grad_(True)

    print("=== 阶段一：预热流场 (500步) ===")
    train_nn(500, "预热 ")

    N_CYCLES = 80
    for c in range(N_CYCLES):
        print(f"\n--- 周期 {c+1}/{N_CYCLES} ---")
        
        train_shape(30, f"[{c+1}] ")
        drag = train_nn(100, f"[{c+1}] ")
        
        xi = xi_in_fixed.clone().detach().requires_grad_(True)
        _, _, area_curr, divergence = compute_alm_loss(
            pinn, mapper, xi, p_mult, lambda_area, MU_DIV, MU_AREA, W_REG, MU_VISC)
        
        with torch.no_grad():
            p_mult -= MU_DIV * divergence.detach()
            
            area_diff = (area_curr - AREA_TARGET).detach()
            lambda_area -= MU_AREA * area_diff
            
            print(f"  >> [Uzawa更新] J(u) = {drag:.4f} | Area Diff = {area_diff.item():.5f}")
            print(f"                 Area Multiplier (λ) = {lambda_area.item():.2f} | Max Pressure |p| = {p_mult.abs().max().item():.4f}")

    print(f"\n训练结束! 最终 J(u) (Drag) = {history_ju[-1]:.4f}")
    with torch.no_grad():
        print(f"a0={mapper.a0.item():.4f}, ak={mapper.ak.data.numpy().round(4)}")
    
    plot_stokes_results(mapper, pinn, history_ju, history_area, len(history_ju))

if __name__ == "__main__":
    main()
