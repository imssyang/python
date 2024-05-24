import torch

# 定义输入和参数
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
w = torch.tensor([0.1, 0.2, 0.3], requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
print("x:", x)
print("w:", w)
print("b:", b)

# 前向传播
z = torch.dot(x, w) + b
a = torch.sigmoid(z)

# 计算损失
loss = (a - 1)**2

# 反向传播
loss.backward()

# 输出梯度
print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("b.grad:", b.grad)

