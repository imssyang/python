import torch
import torch.nn.functional as F

# 定义输入、权重、偏置和目标
x = torch.tensor([1.0], requires_grad=True)
w = torch.tensor([0.5], requires_grad=True)
b = torch.tensor([0.0], requires_grad=True)
y = torch.tensor([1.0])

# 前向传播
z = w * x + b
a = torch.sigmoid(z)
loss = F.mse_loss(a, y)

# 反向传播
loss.backward()

# 输出梯度
print(f"Gradient of x: {x.grad.item()}")
print(f"Gradient of w: {w.grad.item()}")
print(f"Gradient of b: {b.grad.item()}")

