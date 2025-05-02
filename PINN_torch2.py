import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 没有物理损失，仅有模型损失。预测之后，可以按不同的y值计算评价指标

# 检查是否有GPU可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Num GPUs Available:", torch.cuda.device_count())


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


# 生成训练数据
omega_values = np.linspace(0.01, 0.1, 1000)
# y_values = [1, 5, 10, 40]
y_values = [40]
X_train = np.array([[omega, y] for omega in omega_values for y in y_values])
F_train = np.array([amplification_factor(omega, y) for omega, y in X_train])
F_train_real = np.real(F_train)
F_train_imag = np.imag(F_train)


# 定义 PyTorch 神经网络模型
class PINNModel(nn.Module):
    def __init__(self):
        super(PINNModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2)  # 输出实部和虚部
        )

    def forward(self, x):
        return self.net(x)


# 创建模型并转移到设备上
model = PINNModel().to(device)

# 定义均方误差损失函数
criterion = nn.MSELoss()

# 将训练数据转换为 PyTorch 张量，并转移到设备上, 输入值为 omega 和 y
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# 将真实值拼接成 (real, imag) 两列
F_train_tensor = torch.tensor(np.column_stack((F_train_real, F_train_imag)), dtype=torch.float32).to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 训练模型
num_epochs = 1000
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

# 测试数据使用训练数据（数据有限）
X_test_tensor = X_train_tensor
F_test_tensor = F_train_tensor

# 模型评估
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, F_test_tensor)
    print(f'Test Loss (overall): {test_loss.item():.6f}')

# 使用模型进行预测
with torch.no_grad():
    F_pred_tensor = model(X_test_tensor).cpu().numpy()
F_pred_real = F_pred_tensor[:, 0]
F_pred_imag = F_pred_tensor[:, 1]
F_pred = F_pred_real + 1j * F_pred_imag


# 分别计算 y=1,5,10,40 时的评价指标
def compute_metrics(F_true, F_pred):
    # 计算复数预测的误差，这里计算模长误差
    error = np.abs(F_pred - F_true)
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    return mse, rmse, mae


metrics = {}
for y in y_values:
    # 筛选出当前 y 对应的样本索引
    idx = np.where(X_train[:, 1] == y)[0]
    F_true_y = F_train[idx]
    F_pred_y = F_pred[idx]
    mse, rmse, mae = compute_metrics(F_true_y, F_pred_y)
    metrics[y] = (mse, rmse, mae)
    print(f"y = {y}: MSE = {mse:.6e}, RMSE = {rmse:.6e}, MAE = {mae:.6e}")

# 绘制结果及误差
plt.figure(figsize=(10, 6))
for idx, y in enumerate(y_values):
    # 每个 y 对应的数据点：索引 idx, idx+num_y, idx+2*num_y, ...
    F_values = F_pred[idx::len(y_values)]
    # 对应的真实值
    idx_y = np.where(X_train[:, 1] == y)[0]
    F_true_values = F_train[idx_y]

    # 计算误差（模长差值）
    error = np.abs(F_values - F_true_values)
    mse, rmse, mae = compute_metrics(F_true_values, F_values)

    plt.plot(omega_values, np.abs(F_values), label=f'y = {y} (RMSE={rmse:.2e})')
    # 可选：将真实曲线也画上，便于直观比较
    plt.plot(omega_values, np.abs(F_true_values), '--', label=f'Real y = {y}')

plt.xlabel(r'$\omega$', fontsize=14)
plt.ylabel(r'$|F(\omega)|$', fontsize=14)
plt.title('Predicted vs Real Amplification Factor $F(\omega)$', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
