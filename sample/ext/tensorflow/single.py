import os
import certifi
import ssl
import tensorflow as tf

# Set cert file to download dataset
os.environ['SSL_CERT_FILE'] = certifi.where()

# Load dataset and convert int to float
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build machine learning model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()

# Define loss function for training
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

# Config and compile model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Adjust model parameters
model.fit(x_train, y_train, epochs=5)

# Check model performance
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])
probability_model(x_test[:5])
