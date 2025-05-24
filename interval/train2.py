import numpy as np
import mpmath
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from model import SIRENModel

from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GWLensDataset(Dataset):
    def __init__(self, omega_values, y_train_values, normalize_func):
        self.X = []
        self.Y = []
        for y in y_train_values:
            for omega in omega_values:
                o_n, y_n = normalize_func(omega, y)
                self.X.append([o_n, y_n])
                f = amplification_factor(omega, y)
                self.Y.append([f.real, f.imag])
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.Y = torch.tensor(self.Y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]



def visualize_y_prediction(model, y_vis, omega_values, normalize_func, epoch, save_dir="progress_plots2"):
    """
    每隔若干 epoch 画出某个 y 值对应的 F(omega) 拟合效果。

    Args:
        model (torch.nn.Module): 已训练的模型
        y_vis (float): 要可视化的 y 值（未归一化）
        omega_values (np.ndarray): ω 的取值数组
        normalize_func (function): 归一化函数 normalize(omega, y)
        epoch (int): 当前 epoch，便于保存文件名
        save_dir (str): 图像保存目录
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # 归一化输入
    o_n, y_n = normalize_func(omega_values, y_vis)
    Xin = torch.tensor(np.stack([o_n, [y_n] * len(o_n)], axis=1), dtype=torch.float32).to(model.device)

    with torch.no_grad():
        Ypred = model(Xin).cpu().numpy()

    Fpred = Ypred[:, 0] + 1j * Ypred[:, 1]
    Ftrue = np.array([amplification_factor(ω, y_vis) for ω in omega_values])

    # 绘图
    plt.figure(figsize=(40, 4))
    plt.plot(omega_values, np.abs(Ftrue), '--', label='Real |F|')
    plt.plot(omega_values, np.abs(Fpred), label='Pred |F|')
    plt.title(f"Epoch {epoch} | y = {y_vis}")
    plt.xlabel("ω")
    plt.ylabel("|F(ω)|")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/y{y_vis:.2f}_epoch{epoch}.png", dpi=200)
    plt.close()
    print(f"📈 可视化图保存: {save_dir}/y{y_vis:.2f}_epoch{epoch}.png")


# 固定随机种子以保证每次运行一致（可选）
np.random.seed(42)
# === 物理真值函数 ===
def amplification_factor(omega, y):
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(float(y))
    x_m = (y_mp + mpmath.sqrt(y_mp**2 + 4)) / 2
    phi_m = (x_m - y_mp)**2 / 2 - mpmath.log(x_m)
    phase = mpmath.pi*omega_mp/4 + 1j*(omega_mp/2)*(mpmath.log(omega_mp/2) - 2*phi_m)
    gamma_term = mpmath.gamma(1 - 1j*omega_mp/2)
    z = 1j*omega_mp*y_mp**2/2
    hyp1 = mpmath.hyp1f1(1j*omega_mp/2, 1, z)
    return complex(mpmath.exp(phase) * gamma_term * hyp1)

# === 数据准备 ===
omega_values = np.linspace(0.00670904, 43.9322, 5000)

# 在 3 个区间均匀+对数密集采样 y
# y_small  = np.linspace(0.05, 1.0 , num=500)
y_mid    = np.linspace   (1.0 , 3.0 , num=200)
# y_large  = np.linspace   (3.0 , 10.0, num=20)
y_train_values = np.unique(np.concatenate([y_mid])).astype(np.float32)

# 归一化函数
omega_min, omega_max = omega_values.min(), omega_values.max()
y_min,       y_max       = 1.0,        3.0
def normalize(omega, y):
    omega_n = 2*(omega - omega_min)/(omega_max-omega_min) - 1
    y_n     = 2*(y     - y_min    )/(y_max    -y_min       ) - 1
    return omega_n, y_n

# 构造训练集
X_list, Y_list = [], []
for y in y_train_values:
    for omega in omega_values:
        o_n, y_n = normalize(omega, y)
        X_list.append([o_n, y_n])
        f = amplification_factor(omega, y)
        Y_list.append([f.real, f.imag])

# X_train = torch.tensor(X_list, dtype=torch.float32, device=device)
# Y_train = torch.tensor(Y_list, dtype=torch.float32, device=device)
# 构建 Dataset 和 DataLoader
dataset = GWLensDataset(omega_values, y_train_values, normalize)
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True, pin_memory=True)


# === 模型、优化器、损失 ===
model = SIRENModel(
    in_dim=2,
    fourier_feats=20,
    hidden_dim=512,
    out_dim=2,
    depth=8,
    w0=80.0
).to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-6, weight_decay=1e-4)
criterion = nn.MSELoss()

# === 训练 ===
os.makedirs("checkpoints2", exist_ok=True)
for epoch in range(1, 10001):
    model.train()
    total_loss = 0.0
    for batch_X, batch_Y in dataloader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        optimizer.zero_grad()
        pred = model(batch_X)
        loss = criterion(pred, batch_Y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(batch_X)
    avg_loss = total_loss / len(dataloader.dataset)

    if epoch % 500 == 0:
        print(f"Epoch {epoch:5d}  Loss = {avg_loss:.3e}")
        torch.save(model.state_dict(), f"checkpoints2/siren_w0{model.net[0].act.w0:.0f}_ep{epoch}.pth")
    if epoch % 2000 == 0:
        visualize_y_prediction(
            model=model,
            y_vis=6.0,
            omega_values=omega_values,
            normalize_func=normalize,
            epoch=epoch
        )


# 随机选取不重复的 50 个 y 值
y_test_values = np.random.choice(y_train_values, size=50, replace=False)

# 排序（可选，让图更整齐）
y_test_values = np.sort(y_test_values).astype(np.float32)
# === 测试 & 可视化 ===
os.makedirs("res_curves2", exist_ok=True)
for y in y_test_values:   # 举例你关心的 y
    # 构造测试输入并归一化
    o_n, y_n = normalize(omega_values, y)
    Xin = torch.tensor(np.stack([o_n, [y_n]*len(o_n)], axis=1), dtype=torch.float32).to(device)
    with torch.no_grad():
        Ypred = model(Xin).cpu().numpy()
    Fpred = Ypred[:,0] + 1j*Ypred[:,1]
    Ftrue = np.array([amplification_factor(ω, y) for ω in omega_values])

    plt.figure(figsize=(40,4))
    plt.plot(omega_values, np.abs(Ftrue), '--', label='Real |F|')
    plt.plot(omega_values, np.abs(Fpred),  label='Pred |F|')
    plt.title(f"y={y}, SIREN w0={model.net[0].act.w0}")
    plt.xlabel('ω')
    plt.ylabel('|F(ω)|')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"res_curves2/y_{y:.3f}.png", dpi=200)
    plt.close()
    print(f"✅ Saved res_curves2/y_{y:.3f}.png")
