
import numpy as np

# 定义一个简单的二次损失函数
def loss_function(theta):
    return theta ** 2

# 定义损失函数的梯度
def gradient(theta):
    return 2 * theta

# 初始化参数
theta = 10.0
learning_rate = 0.01
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
m = 0.0
v = 0.0
num_iterations = 1000

# Adam 优化算法
for t in range(1, num_iterations + 1):
    grad = gradient(theta)
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2

    # 偏差修正
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)

    # 更新参数
    theta -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    if t % 100 == 0:
        print(f"Iteration {t}: theta = {theta}, loss = {loss_function(theta)}")

print(f"Final parameters: theta = {theta}")

