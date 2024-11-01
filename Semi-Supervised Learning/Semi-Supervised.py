import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.semi_supervised import LabelPropagation

# Load the iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Create a mask for the labels (only a few labels will be known)
random_state = np.random.RandomState(42)
random_unlabeled_points = random_state.rand(len(y)) < 0.8  # 80% unlabeled
y[random_unlabeled_points] = -1  # Marking unlabeled points with -1

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# Create and fit the LabelPropagation model
label_propagation_model = LabelPropagation()
label_propagation_model.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = label_propagation_model.predict(X_test)

# Calculate accuracy only for the labeled test data
# We need to filter out the unlabeled points in y_test
labeled_indices = y_test != -1
accuracy = np.mean(y_pred[labeled_indices] == y_test[labeled_indices])  # Only consider labeled test data
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the results
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50)
plt.title("Semi-Supervised Learning with Label Propagation")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
