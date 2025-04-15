import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

# 训练数据
omega_values = np.linspace(0.01, 0.1, 2000)
y_fixed = 10
X_train = np.array([[omega, y_fixed] for omega in omega_values])
# F_exact = np.array([hyp1f1_only(omega, y_fixed) for omega in omega_values])
F_exact_1f1 = np.array([hyp1f1_only(omega, y_fixed) for omega in omega_values])
F_exact = np.array([rebuild_F_from_1F1(F_exact_1f1[i], omega, y_fixed) for i, omega in enumerate(omega_values)])

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

model = PINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
X_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
lambda_bc = 0.5
num_epochs = 1000

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
    loss_phys = torch.mean(R_r**2 + 10 * R_i**2)

    z_approx = 0.5 * omega * (y ** 2)
    mask = omega < 0.14
    if mask.sum() > 0:
        grad_F_r = torch.autograd.grad(F_r, omega, grad_outputs=torch.ones_like(F_r), create_graph=True)[0]
        grad_F_i = torch.autograd.grad(F_i, omega, grad_outputs=torch.ones_like(F_i), create_graph=True)[0]

        # 理论导数值
        target_grad_r = -omega[mask] * y[mask] ** 2 / 4
        target_grad_i = torch.zeros_like(grad_F_i[mask])

        loss_bc1 = torch.mean((F_r[mask] - 1.0) ** 2 + (F_i[mask]) ** 2)
        loss_bc2 = torch.mean((grad_F_r[mask] - target_grad_r) ** 2 + 10 * (grad_F_i[mask] - target_grad_i) ** 2)

        loss_bc = loss_bc1 + loss_bc2

    else:
        loss_bc = torch.tensor(0.0, device=device)

    # === 添加尾部边界点约束 ===
    omega_tail = torch.tensor([0.1], device=device)
    y_tail = torch.tensor([10.0], device=device)
    input_tail = torch.stack([omega_tail, y_tail], dim=1)
    F_tail_pred = model(input_tail)

    _1F1_tail = hyp1f1_only(0.1, 10)
    target_tail_r = torch.tensor([_1F1_tail.real], device=device)
    target_tail_i = torch.tensor([_1F1_tail.imag], device=device)

    loss_tail = (F_tail_pred[0, 0] - target_tail_r[0]) ** 2 + 10 * (F_tail_pred[0, 1] - target_tail_i[0]) ** 2

    loss = loss_phys + lambda_bc * (loss_bc ) #+ loss_tail
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4e}, Phys: {loss_phys.item():.2e}, BC: {loss_bc.item():.2e}")

# 可视化
model.eval()
with torch.no_grad():
    pred_tensor = model(X_tensor).cpu().numpy()
F_pred_real = pred_tensor[:, 0]
F_pred_imag = pred_tensor[:, 1]
F_pred = [rebuild_F_from_1F1(F_pred_real[i] + 1j * F_pred_imag[i], omega, y_fixed) for i, (omega, _) in enumerate(X_train)]
F_pred = np.array(F_pred)



plt.figure(figsize=(10, 6))
plt.plot(omega_values, np.abs(F_pred), label='Predicted')
plt.plot(omega_values, np.abs(F_exact), '--', label='True')
plt.xlabel("omega")
plt.ylabel("|F(omega)|")
plt.title("PINN (Pure Physics + BC) with y=10")
plt.grid(True)
plt.legend()
plt.show()