import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
import mpmath


# 定义公式 (10) 中的放大因子 F(omega)
def amplification_factor(omega, y):
    # 转换为 mpmath 的浮点数格式（提高精度）
    omega_mp = mpmath.mpf(omega)
    y_mp = mpmath.mpf(y)

    # 计算 phi_m(y)
    x_m = (y_mp + mpmath.sqrt(y_mp ** 2 + 4)) / 2
    phi_m = (x_m - y_mp) ** 2 / 2 - mpmath.log(x_m)

    # 计算复数相位项
    phase = mpmath.pi * omega_mp / 4 + 1j * (omega_mp / 2) * (mpmath.log(omega_mp / 2) - 2 * phi_m)

    # 计算伽马函数和合流超几何函数（使用 mpmath）
    gamma_term = mpmath.gamma(1 - 1j * omega_mp / 2)
    a = 1j * omega_mp / 2
    b = 1
    z = 1j * omega_mp * y_mp ** 2 / 2
    hyp1f1_term = mpmath.hyp1f1(a, b, z)

    # 合并结果并转换为 Python 复数
    F = mpmath.exp(phase) * gamma_term * hyp1f1_term
    return complex(F)


# 定义 omega 的范围
omega_values = np.linspace(0.01, 0.1, 100)

# 定义 y 的取值
y_values = [1, 5, 10, 40]

# 绘制结果
plt.figure(figsize=(10, 6))
for y in y_values:
    F_values = [amplification_factor(omega, y) for omega in omega_values]
    a = np.abs(F_values)
    plt.plot(omega_values, np.abs(F_values), label=f'y = {y}')
    # plt.plot(np.abs(F_values), omega_values, label=f'y = {y}')

# 设置图形属性
plt.xlabel(r'$\omega$', fontsize=14)
plt.ylabel(r'$|F(\omega)|$', fontsize=14)
plt.title('Amplification Factor $F(\omega)$ for Different $y$', fontsize=16)
plt.grid(True)
plt.legend()
plt.show()