from .defineNetwork import HebbianNetwork
import numpy as np


# Define input and output patterns
inputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
outputs = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])

# Initialize the Hebbian network
network = HebbianNetwork(input_size=2, output_size=2)

# Train the network
network.train(inputs, outputs)

# Test the network with a sample input
test_input = np.array([1, 0])
predicted_output = network.predict(test_input)

print(f"Predicted output for input {test_input}: {predicted_output}")