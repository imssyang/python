import numpy as np
import matplotlib.pyplot as plt

# 定义交叉熵损失函数
def cross_entropy_loss(y, y_hat):
    if y == 1:
        return -np.log(y_hat)
    else:
        return -np.log(1 - y_hat)

# 定义概率范围
y_hat = np.linspace(0.01, 0.99, 100)

# 计算损失
loss_y1 = [cross_entropy_loss(1, p) for p in y_hat]
loss_y0 = [cross_entropy_loss(0, p) for p in y_hat]

# 绘制图像
plt.figure(figsize=(10, 6))
plt.plot(y_hat, loss_y1, label=r'$L(\hat{y}) = -log(\hat{y})$ when y=1', color='blue')
plt.plot(y_hat, loss_y0, label=r'$L(\hat{y}) = -log(1-\hat{y})$ when y=0', color='red')
plt.xlabel('Predicted Probability $\hat{y}$')
plt.ylabel('Cross-Entropy Loss')
plt.title('Cross-Entropy Loss Function')
plt.legend()
plt.grid(True)
plt.savefig('cross_entropy_function.png')
plt.show()

