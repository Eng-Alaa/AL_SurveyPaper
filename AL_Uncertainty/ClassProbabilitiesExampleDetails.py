import numpy as np

# Define the training data
X_train = np.array([[10, 1],
                    [15, 0],
                    [5, 1],
                    [8, 0]])

y_train = np.array([1, 1, 0, 0])

# Initialize the weights
weights = np.zeros(X_train.shape[1] + 1)  # Including bias term

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# Set the learning rate and number of epochs
learning_rate = 0.01
num_epochs = 100

# Training the logistic regression model using gradient descent
for epoch in range(num_epochs):
    # Compute linear combination
    Z = np.dot(X_train, weights[1:]) + weights[0]

    # Apply sigmoid function
    predictions = sigmoid(Z)

    # Calculate the loss
    loss = binary_cross_entropy(y_train, predictions)

    # Compute gradients
    gradient = np.dot(X_train.T, predictions - y_train) / len(y_train)
    bias_gradient = np.mean(predictions - y_train)

    # Update weights
    weights[1:] -= learning_rate * gradient
    weights[0] -= learning_rate * bias_gradient

# Define a test data point
X_test = np.array([12, 1])

# Make predictions for the test data point
linear_combination = np.dot(X_test, weights[1:]) + weights[0]
predicted_prob = sigmoid(linear_combination)

# Print the class probabilities
print("Class Probabilities:")
print("Class 0:", 1 - predicted_prob)
print("Class 1:", predicted_prob)