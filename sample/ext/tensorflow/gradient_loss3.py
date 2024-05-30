import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD

# 确保我们使用的是 TensorFlow 2.x
assert tf.__version__.startswith('2.')

# 定义损失函数系数
coefficients = np.array([[1.], [-10.], [25.]])

# 定义 Keras 模型
class CustomModel(Model):
    def __init__(self):
        super(CustomModel, self).__init__()
        # 定义一个可训练的权重 w，初始化为 0
        self.w = tf.Variable(0.0, dtype=tf.float32, trainable=True)

    def call(self, inputs):
        x0, x1, x2 = inputs
        return x0 * self.w**2 + x1 * self.w + x2

# 创建模型实例
model = CustomModel()

# 定义优化器
optimizer = SGD(learning_rate=0.01)

# 定义训练步骤
@tf.function
def train_step():
    with tf.GradientTape() as tape:
        cost = model(coefficients)
    gradients = tape.gradient(cost, [model.w])
    optimizer.apply_gradients(zip(gradients, [model.w]))
    return cost

# 训练模型
for i in range(1000):
    cost = train_step()
    if i % 100 == 0:
        print(f'Step {i}: w = {model.w.numpy()}, cost = {cost.numpy()}')

print(f'Final w = {model.w.numpy()}')

