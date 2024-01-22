import numpy as np

def model_probability(y, x, theta):
    # Assume logistic regression model
    z = theta[0] + theta[1] * x[0] + theta[2] * x[1]
    p = 1 / (1 + np.exp(-z))
    return p ** y * (1 - p) ** (1 - y)

# Define the sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Define the binary cross-entropy loss function
def binary_cross_entropy(y_true, y_pred):
    return -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def Update_weights(X, y, weights, num_epochs,learning_rate):
    # Training the logistic regression model using gradient descent
    for epoch in range(num_epochs):
        # Compute linear combination
        Z = np.dot(X, weights[1:]) + weights[0]

        # Apply sigmoid function
        predictions = sigmoid(Z)

        # Calculate the loss
        loss = binary_cross_entropy(y, predictions)

        # Compute gradients
        gradient = np.dot(X.T, predictions - y) / len(y)
        bias_gradient = np.mean(predictions - y)

        # Update weights
        weights[1:] -= learning_rate * gradient
        weights[0] -= learning_rate * bias_gradient
    return weights


np.random.seed(42)

# Example training data
X = np.array([[1, 1], [2, 2], [3, 3]])
y = np.array([0, 1, 0])
# Initialize the weights
# Set the learning rate and number of epochs
numModels=10
learning_rate = 0.01
num_epochs = 100
Theta=[]
for i in range(0,numModels):
    weights = np.random.randn(X.shape[1] + 1)  # Including bias term
    Theta.append(Update_weights(X, y, weights, num_epochs,learning_rate))
# Example parameter space
#Theta = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]])

# Calculate likelihoods
likelihoods = []
for theta in Theta:
    likelihood = np.prod([model_probability(y[i], X[i], theta) for i in range(len(X))])
    likelihoods.append(likelihood)

# Maximum Likelihood Estimation (MLE)
theta_ml = Theta[np.argmax(likelihoods)]

# Calculate normalized likelihoods
normalized_likelihoods = [likelihood / likelihoods[np.argmax(likelihoods)] for likelihood in likelihoods]

print("Likelihoods:", likelihoods)
print("Maximum Likelihood Estimate theta_ml:", theta_ml)
print("Normalized Likelihoods:", normalized_likelihoods)