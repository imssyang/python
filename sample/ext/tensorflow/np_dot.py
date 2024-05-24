import torch
import time
import numpy as np

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# 计算点积
result = np.dot(a, b)
print(result)  # 输出 32 (1*4 + 2*5 + 3*6)


# 定义输入张量和权重、偏置
w = torch.randn(1000, 1000)
x = torch.randn(1000, 1000)
b = torch.randn(1000, 1)

# 使用矩阵乘法计算 z
start_time = time.time()
z_torch = torch.matmul(w, x) + b
torch_time = time.time() - start_time

# 转换为 numpy 数组
w_np = w.numpy()
x_np = x.numpy()
b_np = b.numpy()

# 使用for循环计算 z
start_time = time.time()
z_manual = np.zeros((1000, 1000))
for i in range(1000):
    for j in range(1000):
        z_manual[i, j] = np.dot(w_np[i, :], x_np[:, j]) + b_np[i]
manual_time = time.time() - start_time

# 输出执行时间
print(f"PyTorch execution time: {torch_time:.6f} seconds")
print(f"For loop execution time: {manual_time:.6f} seconds")

# 验证两个结果是否一致
print(f"Results are equal: {np.allclose(z_torch.numpy(), z_manual)}")

