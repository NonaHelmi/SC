import numpy as np

class MLP:
    def __init__(self, layers, learning_rate=0.01, n_iter=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = []
        self.biases = []

        # مقداردهی اولیه وزن‌ها و بایاس‌ها
        for i in range(len(layers) - 1):
            weight = np.random.rand(layers[i], layers[i + 1]) * 0.01
            bias = np.zeros((1, layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def activation_function(self, x):
        return 1 / (1 + np.exp(-x))  # تابع سیگموئید

    def activation_function_derivative(self, x):
        return x * (1 - x)  # مشتق تابع سیگموئید

    def fit(self, X, y):
        for _ in range(self.n_iter):
            for i in range(X.shape[0]):
                # مرحله پیشرو (forward propagation)
                inputs = X[i].reshape(1, -1)
                outputs = [inputs]

                for weight, bias in zip(self.weights, self.biases):
                    inputs = self.activation_function(np.dot(inputs, weight) + bias)
                    outputs.append(inputs)

                # مرحله پسرو (backward propagation)
                error = y[i].reshape(1, -1) - outputs[-1]
                for j in reversed(range(len(self.weights))):
                    delta = error * self.activation_function_derivative(outputs[j + 1])
                    self.weights[j] += self.learning_rate * np.dot(outputs[j].T, delta)
                    self.biases[j] += self.learning_rate * delta
                    error = np.dot(delta, self.weights[j].T)

    def predict(self, X):
        for weight, bias in zip(self.weights, self.biases):
            X = self.activation_function(np.dot(X, weight) + bias)
        return X