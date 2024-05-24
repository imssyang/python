import numpy as np
import matplotlib.pyplot as plt

# 定义 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 生成输入数据
x = np.linspace(-10, 10, 100)
y = sigmoid(x)

# 绘制图形
plt.plot(x, y, label=r'$p = \sigma(z) = \frac{1}{1 + e^{-z}}$')
plt.xlabel('z')
plt.ylabel('p')
plt.title('Sigmoid Function')
plt.legend()
plt.grid(True)
plt.savefig('sigmoid_function.png')
plt.show()

