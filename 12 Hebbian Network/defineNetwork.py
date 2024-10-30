import numpy as np


class HebbianNetwork:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        # Initialize weights randomly
        self.weights = np.random.rand(input_size, output_size)
        self.learning_rate = learning_rate

    def train(self, inputs, outputs):
        # Update weights based on the Hebbian learning rule
        for i in range(len(inputs)):
            input_vector = inputs[i]
            output_vector = outputs[i]
            self.weights += self.learning_rate * np.outer(input_vector, output_vector)

    def predict(self, input_vector):
        # Predict output by calculating the dot product of input and weights
        return np.dot(input_vector, self.weights)