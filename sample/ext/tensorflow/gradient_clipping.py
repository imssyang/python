
import torch
import torch.nn as nn
import torch.optim as optim

# 假设我们有一个简单的线性模型
model = nn.Linear(10, 1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些随机数据
inputs = torch.randn(100, 10)
targets = torch.randn(100, 1)

# 训练步骤
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

