import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 加入了kummer方程进行损失的约束

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
omega_values = np.linspace(0.01, 0.1, 100)  # 为加快调试，这里用较少点
y_values = [1, 5, 10, 40]
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

# 原数据损失（MSE损失）
data_criterion = nn.MSELoss()

# 将训练数据转换为 PyTorch 张量，并转移到设备上, 输入为 omega 和 y
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
# 注意：真实值拼接为两列（实部，虚部）
F_train_tensor = torch.tensor(np.column_stack((F_train_real, F_train_imag)), dtype=torch.float32).to(device)

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 超参数：物理约束损失的权重
lambda_phys = 1e-3  # 可根据实际情况调整

num_epochs = 1000
batch_size = 32
num_samples = X_train_tensor.shape[0]

for epoch in range(num_epochs):
    permutation = torch.randperm(num_samples)
    epoch_loss = 0.0
    for i in range(0, num_samples, batch_size):
        indices = permutation[i:i + batch_size]
        # 注意：为计算导数，需对输入的 omega 分量设置 requires_grad
        batch_x = X_train_tensor[indices].clone().detach().to(device)
        batch_x.requires_grad = True
        batch_y = F_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(batch_x)  # outputs shape: (batch,2)
        # 数据损失
        loss_data = data_criterion(outputs, batch_y)

        # --- 物理损失部分 ---
        # 将预测输出拼接为复数形式： F = F_r + i F_i
        F_r = outputs[:, 0]
        F_i = outputs[:, 1]
        # 分别计算关于 omega 的一阶导数（只对第一列求导）
        grad_F_r = torch.autograd.grad(F_r, batch_x, grad_outputs=torch.ones_like(F_r), create_graph=True)[0][:, 0]
        grad_F_i = torch.autograd.grad(F_i, batch_x, grad_outputs=torch.ones_like(F_i), create_graph=True)[0][:, 0]
        # 计算二阶导数
        grad2_F_r = torch.autograd.grad(grad_F_r, batch_x, grad_outputs=torch.ones_like(grad_F_r), create_graph=True)[
                        0][:, 0]
        grad2_F_i = torch.autograd.grad(grad_F_i, batch_x, grad_outputs=torch.ones_like(grad_F_i), create_graph=True)[
                        0][:, 0]

        # 提取 omega 和 y
        omega = batch_x[:, 0]  # shape (batch,)
        y = batch_x[:, 1]  # shape (batch,)

        # 计算 z = i * omega * y^2/2.
        # 由于 PyTorch不直接支持复数计算，这里将 z 的实部和虚部分开：
        # z = 0 + i*(omega*y^2/2)
        z_real = torch.zeros_like(omega)
        z_imag = omega * y ** 2 / 2.0

        # 物理约束要求： R = z * d^2F/dz^2 + (1-z)*dF/dz - (i*omega/2)*F = 0.
        # 利用链式法则有：
        # dF/dz = (dF/domega)/(dz/domega)  其中 dz/domega = i*y^2/2  => 1/(dz/domega)= -2i/(y^2)
        # d^2F/dz^2 = d^2F/domega^2 / (dz/domega)^2 = d^2F/domega^2 / (i*y^2/2)^2 = -4*F''/y^4.
        # 故定义：
        # term1 = z * (-4/y^4)*F'',   term2 = (1-z)*(-2i/y^2)*F',   term3 = -i*omega/2 *F.
        # 将 F, F', F'' 表示为复数（由其实部和虚部分别表示），并分离实部和虚部计算：

        # 定义 F', F''（关于 omega 的导数），均为复数形式：
        # F' = grad_F_r + i*grad_F_i,  F'' = grad2_F_r + i*grad2_F_i.
        # 计算 term1：
        # term1 = -4*F''/y^4 * z, 其中 z = i*omega*y^2/2.
        # 令 z = 0 + i*(omega*y^2/2)：
        # term1 = -4/(y^4) * (grad2_F_r + i*grad2_F_i) * i*omega*y^2/2
        #       = - (2i*omega)/(y^2) * (grad2_F_r + i*grad2_F_i).
        # 分离实部和虚部：
        term1_r = 2 * omega / (y ** 2) * grad2_F_i  # 实部： 2ω/y² * grad2_F_i
        term1_i = -2 * omega / (y ** 2) * grad2_F_r  # 虚部： -2ω/y² * grad2_F_r

        # 计算 term2：
        # term2 = (1 - z)*(-2i/y^2)*F'，其中 z = i*omega*y^2/2.
        # 注意：1-z = 1 - i*omega*y^2/2.
        # 先计算 (1-z)*F':
        # (1-z)*(grad_F_r + i grad_F_i) = grad_F_r + i grad_F_i - i*omega*y^2/2*grad_F_r + (omega*y^2/2)*grad_F_i.
        # 再乘上 -2i/y^2：
        # term2 = -2i/y^2 * [grad_F_r + i grad_F_i - i*omega*y^2/2*grad_F_r + (omega*y^2/2)*grad_F_i].
        # 展开后分离实部和虚部，可得：
        term2_r = 2 * grad_F_i / (y ** 2) - omega * grad_F_r
        term2_i = -2 * grad_F_r / (y ** 2) - omega * grad_F_i

        # 计算 term3：
        # term3 = -i*omega/2 * F = -i*omega/2*(F_r + iF_i) = (omega/2*F_i) - i*(omega/2*F_r)
        term3_r = omega * F_i / 2.0
        term3_i = -omega * F_r / 2.0

        # 总残差 R = term1 + term2 + term3，分离实部和虚部：
        R_r = term1_r + term2_r + term3_r
        R_i = term1_i + term2_i + term3_i

        loss_phys = torch.mean(R_r ** 2 + R_i ** 2)

        # 总损失 = 数据损失 + lambda_phys * 物理损失
        loss = loss_data + lambda_phys * loss_phys

        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    epoch_loss /= num_samples
    if (epoch + 1) % 100 == 0:
        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Total Loss: {epoch_loss:.6e}, Data Loss: {loss_data.item():.6e}, Phys Loss: {loss_phys.item():.6e}")

# 测试数据使用训练数据（数据有限）
X_test_tensor = X_train_tensor
F_test_tensor = F_train_tensor

model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = data_criterion(test_outputs, F_test_tensor)
    print(f'Test Data Loss (overall): {test_loss.item():.6e}')

# 使用模型进行预测
with torch.no_grad():
    F_pred_tensor = model(X_test_tensor).cpu().numpy()
F_pred_real = F_pred_tensor[:, 0]
F_pred_imag = F_pred_tensor[:, 1]
F_pred = F_pred_real + 1j * F_pred_imag


# 分别计算 y=1,5,10,40 时的评价指标
def compute_metrics(F_true, F_pred):
    error = np.abs(F_pred - F_true)
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(error))
    return mse, rmse, mae


metrics = {}
for y in y_values:
    idx = np.where(X_train[:, 1] == y)[0]
    F_true_y = F_train[idx]
    F_pred_y = F_pred[idx]
    mse, rmse, mae = compute_metrics(F_true_y, F_pred_y)
    metrics[y] = (mse, rmse, mae)
    print(f"y = {y}: MSE = {mse:.6e}, RMSE = {rmse:.6e}, MAE = {mae:.6e}")

# 绘制结果及误差
plt.figure(figsize=(10, 6))
for idx, y in enumerate(y_values):
    F_values = F_pred[idx::len(y_values)]
    idx_y = np.where(X_train[:, 1] == y)[0]
    F_true_values = F_train[idx_y]
    error = np.abs(F_values - F_true_values)
    _, rmse, _ = compute_metrics(F_true_values, F_values)
    plt.plot(omega_values, np.abs(F_values), label=f'y = {y} (RMSE={rmse:.2e})')
    plt.plot(omega_values, np.abs(F_true_values), '--', label=f'Real y = {y}')
plt.xlabel(r'$\omega$', fontsize=14)
plt.ylabel(r'$|F(\omega)|$', fontsize=14)
plt.title('Predicted vs Real Amplification Factor $F(\omega)$ with Physics Constraint', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()
