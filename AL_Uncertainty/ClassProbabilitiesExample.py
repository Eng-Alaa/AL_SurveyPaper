import numpy as np
from sklearn.linear_model import LogisticRegression

# Define the training data
X_train = np.array([[2.0], [3.0], [4.0], [5.0]])  # Input features
y_train = np.array([0, 0, 1, 1])  # Corresponding class labels

# Create and train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Define the test point
X_test = np.array([[4.5]])

# Predict class probabilities for the test point
#proba = model.predict_proba(X_test)

# Get the model parameters
intercept = model.intercept_[0]
coefficients = model.coef_[0]

# Calculate the linear combination
linear_combination = intercept + coefficients * X_test

# Apply the logistic function to obtain class probabilities
proba = 1 / (1 + np.exp(-linear_combination))

# Print the class probabilities
print("Class Probabilities:")
print("Class 0:", 1 - proba)
print("Class 1:", proba)