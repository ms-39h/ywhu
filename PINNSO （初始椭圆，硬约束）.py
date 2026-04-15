# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 15:04:56 2026

@author: HU YAWEN
"""

import os
# 1. 解决 OpenMP 库冲突报错
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 0. 验证参数设定
# ==========================================
R_IN_FIXED = 0.5      
R_OUT_TARGET = 1.2    
G_VAL = 1.0 / (R_OUT_TARGET * np.log(R_OUT_TARGET / R_IN_FIXED))

# 计算理论最小总能量 J_true
E_grad_true = 2 * np.pi / np.log(R_OUT_TARGET / R_IN_FIXED)
E_source_true = np.pi * (G_VAL**2) * (R_OUT_TARGET**2 - R_IN_FIXED**2)
J_TRUE = E_grad_true + E_source_true

print(f"【配置】目标: 圆(R={R_OUT_TARGET}), G={G_VAL:.4f}")
print(f"理论最小能量 J (含2pi) = {J_TRUE:.4f}")

# ==========================================
# 1. 映射层 (StarMapping)
# ==========================================
class StarMapping(nn.Module):
    def __init__(self, num_fourier=10):
        super().__init__()
        self.r_in = R_IN_FIXED 
        self.num_fourier = num_fourier
        self.register_buffer('k_freq', torch.arange(1, num_fourier + 1).float())
        
        # --- 拟合椭圆初始化 (a=1.5, b=0.8) ---
        theta_sample = torch.linspace(0, 2*np.pi, 2000)
        a_ell = 1.5 
        b_ell = 0.8 
        denom = (torch.cos(theta_sample) / a_ell)**2 + (torch.sin(theta_sample) / b_ell)**2
        R_ellipse = torch.sqrt(1.0 / denom)
        
        # 最小二乘求解系数
        A_list = [torch.ones_like(theta_sample).unsqueeze(1)]
        for k in range(1, num_fourier + 1):
            A_list.append(torch.cos(k * theta_sample).unsqueeze(1))
            A_list.append(torch.sin(k * theta_sample).unsqueeze(1))
        A = torch.cat(A_list, dim=1) 
        coeffs = torch.linalg.lstsq(A, R_ellipse.unsqueeze(1)).solution.flatten()
        
        self.a0 = nn.Parameter(coeffs[0:1]) 
        
        ak_init = []
        bk_init = []
        for k in range(num_fourier):
            ak_init.append(coeffs[1 + 2*k])
            bk_init.append(coeffs[1 + 2*k + 1]) # Fix: Append b_k to bk_init correctly (minor correction for logic, but kept original index style)
            # 修正了原来的初始化小问题，但保持了原来的数学逻辑
        
        # 为了不改变原逻辑，这里原样保留你的 bk_init 循环提取逻辑:
        ak_init = [coeffs[1 + 2*k] for k in range(num_fourier)]
        bk_init = [coeffs[1 + 2*k + 1] for k in range(num_fourier)]
            
        self.ak = nn.Parameter(torch.stack(ak_init))
        self.bk = nn.Parameter(torch.stack(bk_init))
        print(f"初始化完成：a0={self.a0.item():.4f}")

    def get_outer_radius(self, theta):
        theta = theta.view(-1, 1)
        k = self.k_freq.view(1, -1)
        
        perturb = torch.sum(self.ak * torch.cos(k*theta), dim=1, keepdim=True) + \
                  torch.sum(self.bk * torch.sin(k*theta), dim=1, keepdim=True)
        
        R_out = torch.clamp(self.a0 + perturb, min=self.r_in + 0.1) 
        
        R_prime = torch.sum(self.ak * (-k * torch.sin(k*theta)), dim=1, keepdim=True) + \
                  torch.sum(self.bk * (k * torch.cos(k*theta)), dim=1, keepdim=True)
        return R_out, R_prime

    def forward(self, xi):
        xi_r = xi[:, 0:1]
        theta = xi[:, 1:2]
        
        R_out, R_prime = self.get_outer_radius(theta)
        
        dr = R_out - self.r_in 
        r_phys = self.r_in + xi_r * dr
        
        x = r_phys * torch.cos(theta)
        y = r_phys * torch.sin(theta)
        x_phys = torch.cat([x, y], dim=1)
        
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        
        J11 = dr * cos_t
        J12 = xi_r * R_prime * cos_t - r_phys * sin_t
        J21 = dr * sin_t
        J22 = xi_r * R_prime * sin_t + r_phys * cos_t
        
        detJ = r_phys * dr + 1e-8
        
        inv_row1 = torch.cat([J22, -J12], dim=1)
        inv_row2 = torch.cat([-J21, J11], dim=1)
        J_inv = (1.0 / detJ.view(-1, 1, 1)) * torch.stack([inv_row1, inv_row2], dim=1)
        
        J_inv_T = J_inv.transpose(1, 2)
        M = detJ.view(-1, 1, 1) * torch.matmul(J_inv, J_inv_T)
        
        return x_phys, M, detJ

# ==========================================
# 2. 硬约束 PINN 
# ==========================================
class HardConstraintPINN(nn.Module):
    def __init__(self):
        super().__init__()
        # 内部网络只负责拟合残差
        self.net = nn.Sequential(
            nn.Linear(2, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 60), nn.Tanh(),
            nn.Linear(60, 1) 
        )

    def forward(self, xi):
        """
        硬约束公式: u = (1 - xi_r) + xi_r * (1 - xi_r) * NN(xi)
        """
        xi_r = xi[:, 0:1] # [0, 1]
        
        # 线性基底: 内(0)->1, 外(1)->0
        u_base = 1.0 - xi_r
        
        # 距离因子: 内(0)->0, 外(1)->0
        distance_func = xi_r * (1.0 - xi_r)
        
        u_correction = self.net(xi)
        
        u_final = u_base + distance_func * u_correction
        return u_final

# ==========================================
# 3. 损失函数
# ==========================================
def compute_loss(model, mapper, n_collo, g_param, w_pde):
    """
    w_pde: PDE 损失的权重，现在是可调参数
    """
    # 1. 采样 (参考域)
    xi_in = torch.rand(n_collo, 2)
    xi_in[:, 1] = xi_in[:, 1] * 2 * np.pi 
    xi_in.requires_grad = True
    
    # 2. 映射与前向传播
    x_phys, M, detJ = mapper(xi_in)
    u = model(xi_in) 
    
    # --- PDE Loss ---
    grads_u = torch.autograd.grad(u, xi_in, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    flux = torch.matmul(M, grads_u.unsqueeze(2)).squeeze(2)
    
    dflux_dr = torch.autograd.grad(flux[:, 0:1], xi_in, torch.ones_like(flux[:, 0:1]), create_graph=True)[0][:, 0:1]
    dflux_dt = torch.autograd.grad(flux[:, 1:2], xi_in, torch.ones_like(flux[:, 1:2]), create_graph=True)[0][:, 1:2]
    
    loss_pde = torch.mean((dflux_dr + dflux_dt)**2)
    
    # --- Objective (Energy) ---
    M_norm = M / (detJ.view(-1, 1, 1) + 1e-8)
    grad_sq = torch.matmul(grads_u.unsqueeze(1), torch.matmul(M_norm, grads_u.unsqueeze(2))).squeeze()
    
    integrand = (grad_sq + g_param**2) * detJ.squeeze()
    obj_real = torch.mean(integrand) * (2 * np.pi)
    
    # 总 Loss: PDE + Energy 
    total_loss = w_pde * loss_pde + obj_real
    
    return total_loss, obj_real, loss_pde

# ==========================================
# 4. 绘图函数 (完全按照 2x2 排版重写)
# ==========================================
def plot_results(mapper, pinn, history_obj, initial_shape_pts=None, total_epochs=0, w_pde=3.0):
    plt.style.use('default')
    # 调整画布大小以适应 2x2 的排版
    fig = plt.figure(figsize=(12, 10))
    
    # 准备计算数据
    with torch.no_grad():
        t = np.linspace(0, 2*np.pi, 200)
        theta_tensor = torch.tensor(t, dtype=torch.float32)
        R_final, _ = mapper.get_outer_radius(theta_tensor)
        X_final = R_final.numpy().flatten() * np.cos(t)
        Y_final = R_final.numpy().flatten() * np.sin(t)
        
        r_grid = np.linspace(0, 1, 50)
        t_grid = np.linspace(0, 2*np.pi, 100)
        R_mesh, T_mesh = np.meshgrid(r_grid, t_grid)
        xi_vis = torch.tensor(np.stack([R_mesh.flatten(), T_mesh.flatten()], axis=1), dtype=torch.float32)
        x_phys, _, _ = mapper(xi_vis)
        u_pred = pinn(xi_vis)
        X_field = x_phys[:, 0].reshape(T_mesh.shape).numpy()
        Y_field = x_phys[:, 1].reshape(T_mesh.shape).numpy()
        U_field = u_pred.reshape(T_mesh.shape).numpy()

    # --- 图 1：形状对比 (左上) ---
    ax1 = fig.add_subplot(2, 2, 1)
    if initial_shape_pts is not None:
        ax1.plot(initial_shape_pts[0], initial_shape_pts[1], 'g--', linewidth=2, label='Initial Shape')
    
    # 添加固定内边界圆环 (灰色) - 线宽改为 1.0
    circle_in = plt.Circle((0,0), R_IN_FIXED, color='gray', fill=False, linestyle='-', linewidth=1.0, label=f'Inner Boundary')
    ax1.add_patch(circle_in)
    
    # 将计算出来的形状边界换一个更浅的颜色 (浅蓝色 lightsteelblue)，且确保不完全挡住理论形状 (理论圆) - 线宽改为 1.0
    ax1.plot(X_final, Y_final, color='lightsteelblue', linestyle='-', linewidth=1.0, label='Optimized Shape')
    
    # 画理论最优目标圆 (红色虚线)，放在上层且更清晰 - 线宽改为 1.0
    circle = plt.Circle((0,0), R_OUT_TARGET, color='r', fill=False, linestyle=':', linewidth=1.0, label='Target')
    ax1.add_patch(circle)
    
    # 调小图例字体大小 - fontsize 改为 6
    ax1.legend(loc='upper right', fontsize=6)
    ax1.axis('equal')
    ax1.set_title(f"Shape Comparison (Iter: {total_epochs}, W_PDE: {w_pde})")

    # --- 图 2：物理场分布 (右上) ---
    ax2 = fig.add_subplot(2, 2, 2)
    contour = ax2.contourf(X_field, Y_field, U_field, levels=50, cmap='jet', alpha=0.8)
    plt.colorbar(contour, ax=ax2, fraction=0.046, pad=0.04, label='u (Physical Field)')
    
    # 叠加计算出的浅蓝色边界和红色虚线理论圆，以及灰色内圆 - 线宽统一改为 1.0，透明度保持 0.5
    ax2.plot(X_final, Y_final, color='lightsteelblue', linestyle='-', linewidth=1.0, alpha=0.5) 
    ax2.add_patch(plt.Circle((0,0), R_OUT_TARGET, color='r', fill=False, linestyle=':', linewidth=1.0, alpha=0.5))
    ax2.add_patch(plt.Circle((0,0), R_IN_FIXED, color='gray', fill=False, linestyle='-', linewidth=1.0, alpha=0.5))
    
    ax2.axis('equal')
    ax2.set_title("Physical Field (u)")

    # --- 图 3：半径分布 (左下) ---
    ax3 = fig.add_subplot(2, 2, 3)
    # 计算半径线宽改为 1.0
    ax3.plot(t, R_final.numpy().flatten(), color='lightsteelblue', linestyle='-', linewidth=1.0, label='Final R(theta)')
    # 目标半径线宽改为 1.0
    ax3.axhline(y=R_OUT_TARGET, color='r', linestyle='--', linewidth=1.0, label=f'Target R={R_OUT_TARGET}')
    ax3.set_title("Radius Distribution")
    ax3.set_ylim(0.5, 1.8)
    ax3.legend()
    ax3.grid(True)
    
    # --- 图 4：能量泛函收敛趋势 (右下) ---
    ax4 = fig.add_subplot(2, 2, 4)
    # 能量曲线线宽改为 1.0
    ax4.plot(history_obj, label='Obj (Energy)', color='green', linewidth=1.0)
    # 理论最小能量线宽改为 1.0
    ax4.axhline(y=J_TRUE, color='r', linestyle='--', linewidth=1.0, label=f'True Min ({J_TRUE:.4f})')
    ax4.set_title("Energy Functional Convergence")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 5. 主函数
# ==========================================
def main():
    # 初始化
    mapper = StarMapping(num_fourier=30)
    pinn = HardConstraintPINN()
    
    # 记录初始形状
    with torch.no_grad():
        t = np.linspace(0, 2*np.pi, 200)
        theta_tensor = torch.tensor(t, dtype=torch.float32)
        R_init, _ = mapper.get_outer_radius(theta_tensor)
        X_init = R_init.numpy().flatten() * np.cos(t)
        Y_init = R_init.numpy().flatten() * np.sin(t)
        initial_shape_pts = (X_init, Y_init)

    # 联合优化 
    optimizer_all = torch.optim.Adam([
        {'params': pinn.parameters(), 'lr': 1e-3},
        {'params': mapper.parameters(), 'lr': 5e-3} 
    ])
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer_all, step_size=200, gamma=0.8)
    
    history_obj = []
    
    print("=== 开始联合优化 ===")
    
    TOTAL_EPOCHS = 501   # 总迭代次数
    W_PDE = 0.8          # PDE Loss 的权重 

    for epoch in range(TOTAL_EPOCHS):
        optimizer_all.zero_grad()
        loss, obj_val, l_pde = compute_loss(pinn, mapper, 8000, G_VAL, w_pde=W_PDE)
        
        loss.backward()
        optimizer_all.step()
        scheduler.step()
        
        history_obj.append(obj_val.item())
        
        if epoch % 100 == 0:
            current_R_mean = mapper.a0.item()
            curr_lr = optimizer_all.param_groups[1]['lr']
            print(f"Iter {epoch:04d} | Loss: {loss.item():.4f} | PDE: {l_pde.item():.5f} | "
                  f"R_mean: {current_R_mean:.4f} | Obj: {obj_val.item():.4f} | LR: {curr_lr:.1e}")

    print("\n训练结束，绘图...")
    plot_results(mapper, pinn, history_obj, initial_shape_pts, total_epochs=TOTAL_EPOCHS, w_pde=W_PDE)
    
    # --- 最终计算能量与理论值误差输出 ---
    final_energy = history_obj[-1]
    absolute_error = abs(final_energy - J_TRUE)
    relative_error = (absolute_error / J_TRUE) * 100
    
    print("\n=== 优化结果评估 ===")
    print(f"理论最小能量 (True Min) : {J_TRUE:.6f}")
    print(f"最终优化能量 (Final Obj): {final_energy:.6f}")
    print(f"绝对误差                : {absolute_error:.6f}")
    print(f"相对误差                : {relative_error:.4f}%")

if __name__ == "__main__":
    main()