import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # تعداد نمونه‌ها و ویژگی‌ها
        n_samples, n_features = X.shape

        # وزن‌ها و بایاس‌ها را مقداردهی اولیه می‌کنیم
        self.weights = np.zeros(n_features)
        self.bias = 0

        # آموزش مدل
        for _ in range(self.n_iter):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.activation_function(linear_output)

                # به‌روزرسانی وزن‌ها و بایاس
                update = self.learning_rate * (y[idx] - y_predicted)
                self.weights += update * x_i
                self.bias += update

    def activation_function(self, x):
        # تابع فعال‌سازی (در اینجا، تابع Heaviside)
        return np.where(x >= 0, 1, 0)

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.activation_function(linear_output)
        return y_predicted