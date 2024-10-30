import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist

# بارگذاری داده‌های MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# پیش‌پردازش داده‌ها
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# تبدیل برچسب‌ها  (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# ساخت مدل شبکه عصبی
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# کامپایل مدل
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# آموزش مدل
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

# ارزیابی مدل
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)