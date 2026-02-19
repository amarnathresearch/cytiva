import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = pd.read_csv('insurance.csv')
# Preprocess data
#have x1 as smoker or not, x2 as age, x3 as bmi and y as charges
data = pd.get_dummies(data, drop_first=True)
X = data[['age', 'bmi', 'smoker_yes']]
y = data['charges']
# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)
# Make predictions
y_pred = model.predict(X_test)
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')
# Print model coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
# Visualize the results
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color='blue')
plt.xlabel('Actual Charges')
plt.ylabel('Predicted Charges')
plt.title('Actual vs Predicted Charges')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k----', lw=2)
plt.show()

#test with new data
new_data = pd.DataFrame({
    'age': [30, 45],
    'bmi': [22.0, 30.5],
    'smoker_yes': [0, 1]
})
new_pred = model.predict(new_data)
print(f'Predictions for new data: {new_pred}')