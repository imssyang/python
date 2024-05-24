import numpy as np

# 生成一个简单的二分类数据集
np.random.seed(0)  # 为了结果可重复
num_samples = 100
num_features = 2

# 生成随机数据点
X = np.random.randn(num_samples, num_features)
# 生成随机标签 (0 或 1)
y = (np.random.rand(num_samples) > 0.5).astype(int)

# 添加偏置特征（列）到输入数据矩阵X
X = np.hstack((X, np.ones((num_samples, 1))))

# 初始化参数
w = np.random.randn(num_features + 1)
b = 0.0
alpha = 0.01
num_iterations = 1000

for iteration in range(num_iterations):
    # 计算预测值
    z = np.dot(X, w) + b
    y_hat = 1 / (1 + np.exp(-z))

    # 计算损失
    loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    # 计算梯度
    dw = np.dot(X.T, (y_hat - y)) / num_samples
    db = np.mean(y_hat - y)

    # 更新参数
    w -= alpha * dw
    b -= alpha * db

    # 打印损失值（可选）
    if iteration % 100 == 0:
        print(f"Iteration {iteration}, Loss: {loss}")

print("Training complete.")
print(f"Final weights: {w}")
print(f"Final bias: {b}")

