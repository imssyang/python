# 假设有一个损失函数 J(w) = w^2 - 10w + 25 = (w-5)^2 需要最小化，目标值 w=5。
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

coefficients = np.array([[1.], [-10.], [25.]])

w = tf.Variable(0, dtype=tf.float32) # 定义参数 w
x = tf.placeholder(tf.float32, [3, 1])
cost = x[0][0]*w**2 + x[1][0]*w + x[2][0] # 定义损失函数
train = tf.train.GradientDescentOptimizer(0.01).minimize(cost) # 设置梯度下降法的学习率为 0.01
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
print(session.run(w)) # 0.0
for i in range(1000):
    session.run(train, feed_dict={x: coefficients})
    print(session.run(w)) # 0.099999994 0.19799998 0.29403996 0.38815916 ...
print(session.run(w)) # 4.9999886

