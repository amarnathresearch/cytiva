#logistic regression from scratch, x1 as hours studied, x2 as attendance percentage and y as pass(1) or fail(0)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# Logistic Regression class
class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = sigmoid(linear_model)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)
# Sample dataset
X = np.array([[2, 80], [4, 60], [6, 70], [8, 90], [10, 50], [12, 85], [14, 95], [16, 40]])
y = np.array([0, 0, 0, 1, 0, 1, 1, 0])
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Create and train the model
model = LogisticRegressionScratch(learning_rate=0.001, n_iterations=10000)
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')
# Visualize the results
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')
plt.xlabel('Hours Studied')
plt.ylabel('Attendance Percentage')
plt.title('Logistic Regression from Scratch')
plt.show()