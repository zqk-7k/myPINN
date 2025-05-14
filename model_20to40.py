import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Num GPUs Available:", torch.cuda.device_count())

# 测试集使用[3, 8, 12, 16, 20, 24, 28, 32]
# 定义公式 (10) 中的放大因子 F(omega)
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
    F = mpmath.exp(phase) * gamma_term * hyp1f1_term
    return complex(F)

# 训练数据：y = [1, 5, 10, 40]
omega_values = np.linspace(0.01, 0.1, 2000)

# # y ∈ [1, 5) 区间：密集采样
# y_dense = np.linspace(1.0, 5.0, 21)  # 每 0.2 取点
#
# # y ∈ [5, 20] 区间：稀疏采样
# y_sparse = np.linspace(5.0, 20.0, 16)  # 每 1.0 左右
#
# # 合并去重（避免重复 5.0）
# y_values = sorted(set(np.concatenate([y_dense, y_sparse]).tolist()))

# y_train = np.linspace(20.0, 40.0, 41)
y_train = np.linspace(35.0, 40.0, 11)



# y_values = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0,
#            3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
#            10.0, 12.0, 14.0, 16.0 ,18.0, 20.0,
#            30.0, 40.0]
X_train = np.array([[omega, y] for omega in omega_values for y in y_train])
F_train = np.array([amplification_factor(omega, y) for omega, y in X_train])
F_train_real = np.real(F_train)
F_train_imag = np.imag(F_train)

# 定义模型
class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

model = PINNModel().to(device)
criterion = nn.MSELoss()
X_train_tensor = torch.tensor(X_train, dtype=torch.float64).to(device)
F_train_tensor = torch.tensor(np.column_stack((F_train_real, F_train_imag)), dtype=torch.float32).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练
num_epochs = 20000
batch_size = 32
num_samples = X_train_tensor.shape[0]

for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i + batch_size]
        batch_x = X_train_tensor[indices]
        batch_y = F_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)
    epoch_loss /= num_samples
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.6f}")

# ✅ 保存模型
torch.save(model.state_dict(), "pinn_model_weights.pth")

# ====== 测试阶段使用新的 y 值 ======

# 训练数据：y = [1, 5, 10, 40]
omega_values = np.linspace(0.01, 0.1, 2000)

# # y ∈ [1, 5) 区间：密集采样
# y_dense = np.linspace(1.0, 5.0, 21)  # 每 0.2 取点
#
# # y ∈ [5, 20] 区间：稀疏采样
y_test_values = np.linspace(35.0, 40.0, 11)

# y_test_values = [5, 8, 10, 20, 28, 32]
# y_test_values = [1, 5, 10, 40, 20]
X_test = np.array([[omega, y] for omega in omega_values for y in y_test_values])
F_test = np.array([amplification_factor(omega, y) for omega, y in X_test])
F_test_real = np.real(F_test)
F_test_imag = np.imag(F_test)

X_test_tensor = torch.tensor(X_test, dtype=torch.float64).to(device)
F_test_tensor = torch.tensor(np.column_stack((F_test_real, F_test_imag)), dtype=torch.float32).to(device)

# 模型预测
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, F_test_tensor)
    print(f'Test Loss (new y set): {test_loss.item():.6f}')

    F_pred_tensor = test_outputs.cpu().numpy()
F_pred_real = F_pred_tensor[:, 0]
F_pred_imag = F_pred_tensor[:, 1]
F_pred = F_pred_real + 1j * F_pred_imag

# 评估指标
def compute_metrics(F_true, F_pred):
    error = np.abs(F_pred - F_true)
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    return mse, rmse, mae

metrics = {}
for y in y_test_values:
    idx = np.where(X_test[:, 1] == y)[0]
    F_true_y = F_test[idx]
    F_pred_y = F_pred[idx]
    mse, rmse, mae = compute_metrics(F_true_y, F_pred_y)
    metrics[y] = (mse, rmse, mae)
    print(f"y = {y}: MSE = {mse:.6e}, RMSE = {rmse:.6e}, MAE = {mae:.6e}")

import os
os.makedirs("res", exist_ok=True)

# 单独可视化每个 y 的预测与真实值
for y in y_test_values:
    idx_y = np.where(X_test[:, 1] == y)[0]
    F_values = F_pred[idx_y]
    F_true_values = F_test[idx_y]
    mse, rmse, mae = compute_metrics(F_true_values, F_values)

    plt.figure(figsize=(8, 5))
    plt.plot(omega_values, np.abs(F_values), label=f'Predicted |F|')
    plt.plot(omega_values, np.abs(F_true_values), '--', label=f'Real |F|')
    plt.xlabel(r'$\omega$', fontsize=14)
    plt.ylabel(r'$|F(\omega)|$', fontsize=14)
    plt.title(f'Prediction vs Real (y = {y})\nRMSE={rmse:.2e}', fontsize=15)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"res20/MLP_y{int(y)}_result.png")
    plt.close()
    print(f"✅ 图像已保存为 res/siren_y{int(y)}_result.png")
