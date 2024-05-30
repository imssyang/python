
import numpy as np

def numerical_gradient(f, x):
    """计算函数f在x点的数值梯度"""
    grad = np.zeros_like(x)
    h = 1e-7  # 一个非常小的值
    for idx in range(x.size):
        temp_val = x[idx]

        # 计算 f(x+h)
        x[idx] = temp_val + h
        fxh1 = f(x)

        # 计算 f(x-h)
        x[idx] = temp_val - h
        fxh2 = f(x)

        # 数值梯度
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = temp_val  # 还原值

    return grad

def gradient_check(f, grad_f, x):
    """比较数值梯度和反向传播梯度"""
    num_grad = numerical_gradient(f, x)
    backprop_grad = grad_f(x)
    diff = np.linalg.norm(num_grad - backprop_grad) / (np.linalg.norm(num_grad) + np.linalg.norm(backprop_grad))
    return diff

# 示例函数和其梯度
def loss_function(theta):
    return np.sum(theta ** 2)  # 简单的二次函数

def loss_gradient(theta):
    return 2 * theta  # 二次函数的梯度

# 参数初始化
theta = np.random.randn(3)  # 假设有三个参数

# 进行梯度检验
difference = gradient_check(loss_function, loss_gradient, theta)
print(f'Gradient difference: {difference}')

# 差异应该非常小，例如小于 1e-7
assert difference < 1e-7, "梯度检验失败，反向传播实现可能有误"

