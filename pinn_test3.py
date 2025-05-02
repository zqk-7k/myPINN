
import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 物理损失，边界损失和数据损失
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def hyp1f1_only(omega, y):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)
    a = 1j * omega_mp / 2
    b = 1
    z = 1j * omega_mp * y_mp ** 2 / 2
    return complex(mpmath.hyp1f1(a, b, z))

def rebuild_F_from_1F1(pred_1f1, omega, y):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)
    x_m = (y_mp + mpmath.sqrt(y_mp ** 2 + 4)) / 2
    phi_m = (x_m - y_mp) ** 2 / 2 - mpmath.log(x_m)
    phase = mpmath.pi * omega_mp / 4 + 1j * (omega_mp / 2) * (mpmath.log(omega_mp / 2) - 2 * phi_m)
    gamma_term = mpmath.gamma(1 - 1j * omega_mp / 2)
    return mpmath.exp(phase) * gamma_term * pred_1f1

omega_values = np.linspace(0.01, 0.1, 1000)
y_values = [1, 5, 10, 40]
X_train = np.array([[omega, y] for omega in omega_values for y in y_values])
F_train = np.array([hyp1f1_only(omega, y) for omega, y in X_train])
F_train_real = np.real(F_train)
F_train_imag = np.imag(F_train)

class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)
        )
    def forward(self, x):
        return self.net(x)

model = PINNModel().to(device)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(np.column_stack((F_train_real, F_train_imag)), dtype=torch.float32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 1000
batch_size = 32
lambda_data = 1.0
lambda_phy = 0.3
lambda_bc = 2.0

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train_tensor.shape[0])
    epoch_loss = 0.0
    for i in range(0, X_train_tensor.shape[0], batch_size):
        indices = permutation[i:i + batch_size]
        batch = X_train_tensor[indices]
        target = Y_train_tensor[indices]
        omega = batch[:, 0].clone().detach().requires_grad_(True)
        y = batch[:, 1].clone().detach()
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
        loss_phys = torch.mean(R_r ** 2 + R_i ** 2)

        # 添加边界条件损失：当 omega ≈ 0 时，F ≈ 1 + 0i
        z_approx = 0.5 * omega * (y ** 2)
        mask = z_approx < 0.02  # 可以视情况调整阈值
        # mask = omega < 0.02
        if mask.sum() > 0:
            loss_bc = torch.mean((F_r[mask] - 1.0) ** 2 + (F_i[mask]) ** 2)
        else:
            loss_bc = torch.tensor(0.0, device=device)

        loss_data = torch.mean((F_r - target[:, 0]) ** 2 + (F_i - target[:, 1]) ** 2)
        loss = lambda_phy * loss_phys + lambda_data * loss_data + lambda_bc * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.6e}, Phys: {loss_phys.item():.2e}, Data: {loss_data.item():.2e}"
              f", Bc: {loss_bc.item():.2e}")

model.eval()
with torch.no_grad():
    pred_tensor = model(X_train_tensor).cpu().numpy()
F_pred_real = pred_tensor[:, 0]
F_pred_imag = pred_tensor[:, 1]

F_pred = []
for i, (omega, y) in enumerate(X_train):
    f1f1_pred = F_pred_real[i] + 1j * F_pred_imag[i]
    F_full = rebuild_F_from_1F1(f1f1_pred, omega, y)
    F_pred.append(F_full)
F_pred = np.array(F_pred)

plt.figure(figsize=(10, 6))
for y in y_values:
    idx = np.where(X_train[:, 1] == y)[0]
    true = [rebuild_F_from_1F1(hyp1f1_only(X_train[i][0], y), X_train[i][0], y) for i in idx]
    plt.plot(omega_values, np.abs(F_pred[idx]), label=f'y={y}')
    plt.plot(omega_values, np.abs(true), '--', label=f'true y={y}')
plt.xlabel("omega")
plt.ylabel("|F(omega)|")
plt.legend()
plt.grid(True)
plt.title("Predicted vs True Magnification |F(ω,y)| with Data + Physics Loss")
plt.show()