import numpy as np
import matplotlib.pyplot as plt

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # 减去最大值以防止溢出
    return exp_x / np.sum(exp_x, axis=0)

# 定义输入向量的范围
x = np.linspace(-2.0, 2.0, 100)
y = np.linspace(-2.0, 2.0, 100)

X, Y = np.meshgrid(x, y)
Z = np.array([softmax([x, y, 0]) for x, y in zip(np.ravel(X), np.ravel(Y))])
Z = Z[:, 0].reshape(X.shape)  # 取 softmax 的第一个输出作为高度值

# 创建图像
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制3D曲面
ax.plot_surface(X, Y, Z, cmap='viridis')

ax.set_title('Softmax Function')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('softmax(x1)')

plt.show()

