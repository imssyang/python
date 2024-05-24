import torch

# 定义输入张量
x1 = torch.tensor([1.0], requires_grad=True)
x2 = torch.tensor([2.0], requires_grad=True)
x3 = torch.tensor([3.0], requires_grad=True)

# 计算输出
z = (x1 + 2 * x2) * x3

# 反向传播
z.backward()

# 输出梯度
print("Gradient of x1:", x1.grad.item())  # 输出 3.0
print("Gradient of x2:", x2.grad.item())  # 输出 6.0
print("Gradient of x3:", x3.grad.item())  # 输出 5.0

