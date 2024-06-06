import tensorflow as tf
import numpy as np

# 创建一些样本数据
num_classes = 3
num_features = 4
num_samples = 10

np.random.seed(42)
X_data = np.random.rand(num_samples, num_features).astype(np.float32)
y_data = np.random.randint(num_classes, size=num_samples)

# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((X_data, y_data)).batch(5)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(num_classes, activation=None)
])

# 选择优化器和损失函数
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 训练模型
epochs = 100
for epoch in range(epochs):
    for X_batch, y_batch in dataset:
        with tf.GradientTape() as tape:
            logits = model(X_batch)
            loss = loss_fn(y_batch, logits)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch + 1}, Loss: {loss.numpy()}')

# 预测
logits = model(X_data)
predictions = tf.argmax(logits, axis=1)
print(f'Predictions: {predictions.numpy()}')
print(f'True labels: {y_data}')

