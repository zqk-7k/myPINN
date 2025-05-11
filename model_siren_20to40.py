import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 高精度 F(omega, y) 定义
def amplification_factor(omega, y):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)
    x_m = (y_mp + mpmath.sqrt(y_mp ** 2 + 4)) / 2
    phi_m = (x_m - y_mp) ** 2 / 2 - mpmath.log(x_m)
    phase = mpmath.pi * omega_mp / 4 + 1j * (omega_mp / 2) * (mpmath.log(omega_mp / 2) - 2 * phi_m)
    gamma_term = mpmath.gamma(1 - 1j * omega_mp / 2)
    a = 1j * omega_mp / 2
    b = 1
    z = 1j * omega_mp * y_mp ** 2 / 2
    hyp1f1_term = mpmath.hyp1f1(a, b, z)
    return complex(mpmath.exp(phase) * gamma_term * hyp1f1_term)

# 构建训练集：仅限 y > 20
omega_values = np.linspace(0.01, 0.1, 10000)
y_train = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
X_train = np.array([[omega, y] for omega in omega_values for y in y_train])
F_train = np.array([amplification_factor(omega, y) for omega, y in X_train])
F_train_real = np.real(F_train)
F_train_imag = np.imag(F_train)

# SIREN 模型结构：周期激活
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30):
        super().__init__()
        self.in_features = in_features
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class SIREN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            SineLayer(2, 64),
            SineLayer(64, 64),
            SineLayer(64, 64),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# 构建模型与训练数据
model = SIREN().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

X_train_tensor = torch.tensor(X_train, dtype=torch.float64).to(device)
F_train_tensor = torch.tensor(np.column_stack((F_train_real, F_train_imag)), dtype=torch.float64).to(device)

# 训练
num_epochs = 2000
batch_size = 64
num_samples = X_train_tensor.shape[0]

for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X_train_tensor[indices]
        batch_y = F_train_tensor[indices]

        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= num_samples
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6e}")

# 保存模型
torch.save(model.state_dict(), "siren_y_gt_20.pth")

# 测试数据
# y_test = [20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]
y_test = [21, 23, 25, 27, 29, 30, 32, 34, 36, 38, 40]
X_test = np.array([[omega, y] for omega in omega_values for y in y_test])
F_test = np.array([amplification_factor(omega, y) for omega, y in X_test])
F_test_real = np.real(F_test)
F_test_imag = np.imag(F_test)
X_test_tensor = torch.tensor(X_test, dtype=torch.float64).to(device)
F_test_tensor = torch.tensor(np.column_stack((F_test_real, F_test_imag)), dtype=torch.float64).to(device)

# 预测
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor).cpu().numpy()
F_pred_real = outputs[:, 0]
F_pred_imag = outputs[:, 1]
F_pred = F_pred_real + 1j * F_pred_imag

# 分别保存每个 y 的图像
for y in y_test:
    idx = np.where(X_test[:, 1] == y)[0]
    F_true = F_test[idx]
    F_out = F_pred[idx]

    plt.figure(figsize=(8, 5))
    plt.plot(omega_values, np.abs(F_out), label='Predicted')
    plt.plot(omega_values, np.abs(F_true), '--', label='True')
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$|F(\omega)|$')
    plt.title(f'SIREN Predicted vs True (y = {y})')
    plt.grid(True)
    plt.legend()
    filename = f"res20/siren_y{int(y)}_result.png"
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ 已保存图像：{filename}")