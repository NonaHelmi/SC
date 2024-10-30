import numpy as np
from .perceptronnetwork import Perceptron


# داده‌های آموزشی (AND gate)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# برچسب‌های خروجی (AND gate)
y = np.array([0, 0, 0, 1])

# ایجاد و آموزش پرسپترون
perceptron = Perceptron(learning_rate=0.1, n_iter=10)
perceptron.fit(X, y)

# پیش‌بینی بر روی داده‌های آموزشی
predictions = perceptron.predict(X)

print("Predictions:", predictions)
