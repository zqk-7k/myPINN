import torch
import torch.nn as nn
import math


# —— Fourier Feature 映射 ——
class FourierFeature(nn.Module):
    def __init__(self, input_dim, mapping_size=20, scale=10.0):
        """
        input_dim: 原始输入维度（2）
        mapping_size: 每个维度映射的频率个数
        scale: 频率数值尺度（可调）
        """
        super().__init__()
        # B: [mapping_size, input_dim]
        self.register_buffer('B', torch.randn(mapping_size, input_dim) * scale)

    def forward(self, x):
        # x: [N, input_dim]
        # x_proj: [N, mapping_size] = 2π * x @ B^T
        x_proj = 2 * math.pi * x @ self.B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# —— SIREN 基本层 ——
class Sine(nn.Module):
    def __init__(self, w0=80.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SIRENLayer(nn.Module):
    def __init__(self, in_features, out_features, is_first=False, w0=80.0):
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.act = Sine(w0)
        self.init_weights(w0)

    def init_weights(self, w0):
        with torch.no_grad():
            if self.is_first:
                # first layer initialization
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                bound = math.sqrt(6 / self.in_features) / w0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return self.act(self.linear(x))


# —— 改进后的 SIRENModel ——
class SIRENModel(nn.Module):
    def __init__(
            self,
            in_dim=2,
            fourier_feats=20,
            hidden_dim=512,
            out_dim=2,
            depth=8,
            w0=80.0
    ):
        """
        in_dim: 原始输入维度 (ω,y)
        fourier_feats: 每个输入维度的 Fourier 特征个数
        hidden_dim: 隐藏层宽度
        depth: 总的 SIREN 层数（不含 Fourier 映射和最后线性层）
        w0: SIREN 首层频率
        """
        super().__init__()
        # 1. Fourier 映射层
        self.ff = FourierFeature(in_dim, mapping_size=fourier_feats, scale=10.0)
        current_dim = fourier_feats * 2  # sin+cos 拼接后维度

        # 2. SIREN 网络
        layers = []
        # 第一层：SIRENLayer with is_first=True
        layers.append(SIRENLayer(current_dim, hidden_dim, is_first=True, w0=w0))
        # 中间层
        for _ in range(depth - 1):
            layers.append(SIRENLayer(hidden_dim, hidden_dim, w0=w0))
        # 输出层
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [N, 2] 包含 (ω_norm, y_norm)
        x = self.ff(x)  # → [N, fourier_feats*2]
        return self.net(x)  # → [N, 2]
