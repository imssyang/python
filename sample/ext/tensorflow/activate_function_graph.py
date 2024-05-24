import numpy as np
import matplotlib.pyplot as plt

# 定义激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

# 定义输入范围
x = np.linspace(-10, 10, 400)

# 计算激活函数的输出
sigmoid_y = sigmoid(x)
tanh_y = tanh(x)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
elu_y = elu(x)

# 创建图像
plt.figure(figsize=(10, 6))

# Sigmoid
plt.subplot(2, 3, 1)
plt.plot(x, sigmoid_y)
plt.title('Sigmoid')
plt.grid(True)
plt.ylim(-0.1, 1.1)
plt.text(-9, 0.9, r'$\sigma(x) = \frac{1}{1 + e^{-x}}$', fontsize=12)
plt.text(-9, 0.75, 'Range: (0, 1)', fontsize=12)

# Tanh
plt.subplot(2, 3, 2)
plt.plot(x, tanh_y)
plt.title('Tanh')
plt.grid(True)
plt.ylim(-1.1, 1.1)
plt.text(-9, 0.7, r'$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$', fontsize=12)
plt.text(-9, 0.35, 'Range: (-1, 1)', fontsize=12)

# ReLU
plt.subplot(2, 3, 3)
plt.plot(x, relu_y)
plt.title('ReLU')
plt.grid(True)
plt.ylim(-1, 10)
plt.text(-9, 8, r'$\mathrm{ReLU}(x) = \max(0, x)$', fontsize=12)
plt.text(-9, 6.5, 'Range: [0, ∞)', fontsize=12)

# Leaky ReLU
plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu_y)
plt.title('Leaky ReLU')
plt.grid(True)
plt.ylim(-1, 10)
plt.text(-9, 8, r'$\mathrm{Leaky ReLU}(x) = x\ \mathrm{if}\ x > 0$', fontsize=12)
plt.text(-9, 6.5, r'$\mathrm{Leaky ReLU}(x) = \alpha x\ \mathrm{if}\ x \leq 0$', fontsize=12)
plt.text(-9, 5, 'Range: (-∞, ∞)', fontsize=12)
plt.text(-9, 3.5, r'$\alpha = 0.01$', fontsize=12)

# ELU
plt.subplot(2, 3, 5)
plt.plot(x, elu_y)
plt.title('ELU')
plt.grid(True)
plt.ylim(-2, 10)
plt.text(-9, 8, r'$\mathrm{ELU}(x) = x\ \mathrm{if}\ x > 0$', fontsize=12)
plt.text(-9, 6.5, r'$\mathrm{ELU}(x) = \alpha (e^x - 1) \mathrm{if}\ x \leq 0$', fontsize=12)
plt.text(-9, 5, 'Range: (-∞, ∞)', fontsize=12)
plt.text(-9, 3.5, r'$\alpha = 1.0$', fontsize=12)

plt.tight_layout()
plt.show()

