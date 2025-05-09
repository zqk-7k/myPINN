import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定 y = 10，定义超几何函数
def hyp1f1_only(omega, y=10):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)
    a = 1j * omega_mp / 2
    b = 1
    z = 1j * omega_mp * y_mp ** 2 / 2
    return complex(mpmath.hyp1f1(a, b, z))

def rebuild_F_from_1F1(pred_1f1, omega, y=10):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)
    x_m = (y_mp + mpmath.sqrt(y_mp ** 2 + 4)) / 2
    phi_m = (x_m - y_mp) ** 2 / 2 - mpmath.log(x_m)
    phase = mpmath.pi * omega_mp / 4 + 1j * (omega_mp / 2) * (mpmath.log(omega_mp / 2) - 2 * phi_m)
    gamma_term = mpmath.gamma(1 - 1j * omega_mp / 2)
    return mpmath.exp(phase) * gamma_term * pred_1f1

# 网络模型
class PINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 2)
        )
    def forward(self, x):
        return self.net(x)

# 准备训练数据
omega_values = np.linspace(0.01, 0.1, 2000)
y_fixed = 10
X_train = np.array([[omega, y_fixed] for omega in omega_values])
F_exact_1f1 = np.array([hyp1f1_only(omega, y_fixed) for omega in omega_values])
F_exact = np.array([rebuild_F_from_1F1(F_exact_1f1[i], omega, y_fixed) for i, omega in enumerate(omega_values)])
X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)

# 权重组合
lambda_range = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
num_epochs = 5000

# 创建图像保存目录
os.makedirs("results", exist_ok=True)

# 超参数网格搜索
for lambda_bc1 in lambda_range:
    for lambda_bc2 in lambda_range:
        for lambda_phys in lambda_range:
            print(f"\nTraining with bc1={lambda_bc1}, bc2={lambda_bc2}, phys={lambda_phys}")
            model = PINN().to(device)
            optimizer = optim.Adam(model.parameters(), lr=5e-4)

            for epoch in range(num_epochs):
                omega = X_tensor[:, 0].clone().detach().requires_grad_(True)
                y = X_tensor[:, 1].clone().detach()
                input_xy = torch.stack([omega, y], dim=1)
                output = model(input_xy)
                F_r, F_i = output[:, 0], output[:, 1]

                grad_F_r = torch.autograd.grad(F_r, omega, grad_outputs=torch.ones_like(F_r), create_graph=True)[0]
                grad_F_i = torch.autograd.grad(F_i, omega, grad_outputs=torch.ones_like(F_i), create_graph=True)[0]
                grad2_F_r = torch.autograd.grad(grad_F_r, omega, grad_outputs=torch.ones_like(grad_F_r), create_graph=True)[0]
                grad2_F_i = torch.autograd.grad(grad_F_i, omega, grad_outputs=torch.ones_like(grad_F_i), create_graph=True)[0]

                term1_r = 2 * omega / (y ** 2) * grad2_F_i
                term1_i = -2 * omega / (y ** 2) * grad2_F_r
                term2_r = 2 * grad_F_i / (y ** 2) - omega * grad_F_r
                term2_i = -2 * grad_F_r / (y ** 2) - omega * grad_F_i
                term3_r = omega * F_i / 2.0
                term3_i = -omega * F_r / 2.0
                R_r = term1_r + term2_r + term3_r
                R_i = term1_i + term2_i + term3_i
                loss_phys = torch.mean(R_r**2 + R_i**2)

                # 边界点
                omega_bc = torch.tensor([[0.01]], dtype=torch.float32, requires_grad=True).to(device)
                y_bc = torch.tensor([[10.0]], dtype=torch.float32).to(device)
                input_bc = torch.cat([omega_bc, y_bc], dim=1)
                output_bc = model(input_bc)
                F0_r = output_bc[:, 0]
                F0_i = output_bc[:, 1]

                loss_bc1 = (F0_r - 1.0) ** 2 + F0_i ** 2

                gradF_r = torch.autograd.grad(F0_r, omega_bc, torch.ones_like(F0_r), create_graph=True)[0]
                gradF_i = torch.autograd.grad(F0_i, omega_bc, torch.ones_like(F0_i), create_graph=True)[0]

                target_grad_r = -0.48565947
                target_grad_i = -0.09200674
                loss_bc2 = (gradF_r - target_grad_r) ** 2 + (gradF_i - target_grad_i) ** 2

                loss = lambda_phys * loss_phys + lambda_bc1 * loss_bc1 + lambda_bc2 * loss_bc2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (epoch + 1) % 1000 == 0:
                    print(f"Epoch {epoch+1}, Loss: {loss.item():.4e}, Phys: {loss_phys.item():.2e}, BC1: {loss_bc1.item():.2e}, BC2: {loss_bc2.item():.2e}")

            # 可视化
            model.eval()
            with torch.no_grad():
                pred_tensor = model(X_tensor).cpu().numpy()
            F_pred_real = pred_tensor[:, 0]
            F_pred_imag = pred_tensor[:, 1]
            F_pred = [rebuild_F_from_1F1(F_pred_real[i] + 1j * F_pred_imag[i], omega, y_fixed)
                      for i, (omega, _) in enumerate(X_train)]
            F_pred = np.array(F_pred)

            plt.figure(figsize=(10, 6))
            plt.plot(omega_values, np.abs(F_pred), label='Predicted')
            plt.plot(omega_values, np.abs(F_exact), '--', label='True')
            plt.xlabel("omega")
            plt.ylabel("|F(omega)|")
            plt.title(f"λ_bc1={lambda_bc1}, λ_bc2={lambda_bc2}, λ_phys={lambda_phys}")
            plt.grid(True)
            plt.legend()
            fname = f"results/pinn_bc1={lambda_bc1}_bc2={lambda_bc2}_phys={lambda_phys}.png"
            plt.savefig(fname)
            plt.close()
