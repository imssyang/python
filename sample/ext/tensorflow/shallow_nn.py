import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error

# 构建浅层神经网络模型
model = Sequential([
    Dense(10, activation='relu', input_shape=(3,)),  # 隐藏层，10个神经元，输入维度为3
    Dense(1)  # 输出层，1个神经元（回归任务）
])

# 编译模型
model.compile(optimizer='adam', loss='mean_squared_error')

# 假设有一些训练数据
import numpy as np
X_train = np.random.rand(100, 3)  # 100个样本，3个特征
y_train = np.random.rand(100, 1)  # 100个目标值

# 训练模型
model.fit(X_train, y_train, epochs=10)

# 使用模型进行预测
X_test = np.random.rand(10, 3)
predictions = model.predict(X_test)
print(predictions)


y_test = np.random.rand(10, 1)  # 假设有真实的目标值
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

