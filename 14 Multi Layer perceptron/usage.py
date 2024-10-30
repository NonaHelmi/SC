import numpy as np
from .MLP_Network import MLP


# داده‌های آموزشی (OR gate)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# برچسب‌های خروجی (OR gate)
y = np.array([[0], [1], [1], [1]])

# ایجاد و آموزش شبکه عصبی چند لایه
mlp = MLP(layers=[2, 3, 1], learning_rate=0.1, n_iter=10000)
mlp.fit(X, y)

# پیش‌بینی بر روی داده‌های آموزشی
predictions = mlp.predict(X)

# تبدیل خروجی به 0 و 1
predictions = np.where(predictions >= 0.5, 1, 0)

print("Predictions:\n", predictions)