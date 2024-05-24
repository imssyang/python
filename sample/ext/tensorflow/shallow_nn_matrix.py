import torch
import torch.nn.functional as F

# 输入向量
X = torch.tensor([1.0, 2.0, 3.0])

# 第一层的权重和偏置
W1 = torch.tensor([[0.2, 0.4, 0.6],
                   [0.8, 0.1, 0.3]])
b1 = torch.tensor([0.1, 0.2])

# 第二层的权重和偏置
W2 = torch.tensor([0.5, 0.7])
b2 = torch.tensor(0.3)

# 第一层线性组合
Z1 = torch.matmul(W1, X) + b1
print("Z1:", Z1)

# 第一层激活函数 (ReLU)
h1 = F.relu(Z1)
print("h1:", h1)

# 第二层线性组合
Z2 = torch.matmul(W2, h1) + b2
print("Z2:", Z2)

# 最终输出
y = Z2.item()
print("Output y:", y)
