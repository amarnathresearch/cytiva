#logistic regression
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
# Sample dataset
data = {'Hours_Studied': [2, 4, 6, 8, 10, 12, 14, 16],
        'Attendance_Percentage': [80, 60, 70, 90, 50, 85, 95, 40],
        'Pass': [0, 0, 0, 1, 0, 1, 1, 0]}
df = pd.DataFrame(data)
X = df[['Hours_Studied', 'Attendance_Percentage']].values
y = df['Pass'].values
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# Create and train the model
model = LogisticRegression()
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
plt.title('Logistic Regression using scikit-learn')
plt.show()