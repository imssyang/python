# 假设有一个损失函数 J(w) = w^2 - 10w + 25 = (w-5)^2 需要最小化，目标值 w=5。
import os
os.environ["TF_USE_LEGACY_KERAS"] = "True"

import numpy as np
import tensorflow as tf

# 确保我们使用的是 TensorFlow 2.x
assert tf.__version__.startswith('2.')

# 定义参数 w
w = tf.Variable(0.0, dtype=tf.float32)

# 定义损失函数
def cost_fn():
    coefficients = np.array([1., -10., 25.])
    return coefficients[0] * w**2 + coefficients[1] * w + coefficients[2]

# 使用梯度下降优化器，学习率为 0.01
optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=0.01)

# 训练步骤
for i in range(1000):
    optimizer.minimize(cost_fn, var_list=[w])
    if i % 100 == 0:  # 每 100 次打印一次 w 的值
        print(f'Step {i}: w = {w.numpy()}')
print(f'Final w = {w.numpy()}')

# Step 0: w = 0.09999999403953552
# Step 100: w = 4.350164413452148
# Step 200: w = 4.913819789886475
# Step 300: w = 4.988570690155029
# Step 400: w = 4.998484134674072
# Step 500: w = 4.9997992515563965
# Step 600: w = 4.999971866607666
# Step 700: w = 4.999988555908203
# Step 800: w = 4.999988555908203
# Step 900: w = 4.999988555908203
# Final w = 4.999988555908203

