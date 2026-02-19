import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# RANDOM FOREST CLASSIFICATION - FRUIT DATA
# =====================================================

# Load the fruit dataset
df = pd.read_csv('/kaggle/input/datasets/amarnathr/fruitsnew/fruits.csv')

print("=" * 60)
print("RANDOM FOREST CLASSIFICATION - FRUIT DATA")
print("=" * 60)
print("\nDataset shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nFruit distribution:")
print(df['target'].value_counts())
print("\nDataset Statistics:")
print(df.describe())

# Encode categorical features
le_color = LabelEncoder()
df['color_encoded'] = le_color.fit_transform(df['color'])

# Create feature matrix X and target vector y
X = df[['color_encoded', 'diameter']].values
y = df['target'].values

# Encode target labels
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print("\n" + "=" * 60)
print("FEATURE ENCODING")
print("=" * 60)
print("\nColor Encoding:")
for i, color in enumerate(le_color.classes_):
    print(f"{color}: {i}")
print("\nTarget Encoding:")
for i, fruit in enumerate(le_target.classes_):
    print(f"{fruit}: {i}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train Random Forest Classifier
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train, y_train)

print("\nModel trained successfully!")
print(f"Number of trees: {rf_model.n_estimators}")
print(f"Max depth: {rf_model.max_depth}")

# Make predictions
y_pred_train = rf_model.predict(X_train)
y_pred_test = rf_model.predict(X_test)

# Evaluate model
print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(
    y_test, y_pred_test,
    target_names=le_target.classes_
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance
feature_importance = rf_model.feature_importances_
feature_names = ['Color', 'Diameter']

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Feature Importance
axes[0, 0].barh(feature_names, feature_importance, color='steelblue')
axes[0, 0].set_xlabel('Importance')
axes[0, 0].set_title('Feature Importance')
axes[0, 0].grid(axis='x', alpha=0.3)

# 2. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)
axes[0, 1].set_title('Confusion Matrix (Test Set)')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Accuracy Comparison
accuracies = [train_accuracy, test_accuracy]
axes[1, 0].bar(['Training', 'Testing'], accuracies, color=['green', 'orange'])
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].set_title('Model Accuracy Comparison')
axes[1, 0].set_ylim([0, 1])
axes[1, 0].grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(accuracies):
    axes[1, 0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')

# 4. Fruit Distribution by Diameter
fruit_types = le_target.classes_
colors_map = {0: 'red', 1: 'purple', 2: 'yellow', 3: 'orange'}
for i, fruit in enumerate(fruit_types):
    mask = df['target'] == fruit
    axes[1, 1].scatter(df[mask]['diameter'], [i] * mask.sum(),
                       label=fruit, alpha=0.6, s=100)

axes[1, 1].set_ylabel('Fruit Type')
axes[1, 1].set_yticks(range(len(fruit_types)))
axes[1, 1].set_yticklabels(fruit_types)
axes[1, 1].set_xlabel('Diameter (cm)')
axes[1, 1].set_title('Fruit Distribution by Diameter')
axes[1, 1].grid(alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('random_forest_fruit_data.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'random_forest_fruit_data.png'")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print("\nPredictions on test samples:")
for i in range(min(5, len(X_test))):
    color_idx = int(X_test[i][0])
    diameter = X_test[i][1]
    true_fruit = le_target.inverse_transform([y_test[i]])[0]
    pred_fruit = le_target.inverse_transform([y_pred_test[i]])[0]
    color_name = le_color.inverse_transform([color_idx])[0]
    confidence = rf_model.predict_proba(X_test[i].reshape(1, -1)).max()
    
    print(f"\nColor: {color_name}, Diameter: {diameter:.1f}cm")
    print(f"True Fruit: {true_fruit}, Predicted: {pred_fruit}, Confidence: {confidence:.4f}")

# Prediction on new data
print("\n" + "=" * 60)
print("PREDICTION ON NEW DATA")
print("=" * 60)

# Example: green fruit with diameter 2.5
new_color = 'green'
new_diameter = 2.5

new_color_encoded = le_color.transform([new_color])[0]
new_sample = np.array([[new_color_encoded, new_diameter]])

prediction = rf_model.predict(new_sample)[0]
confidence = rf_model.predict_proba(new_sample)[0]

predicted_fruit = le_target.inverse_transform([prediction])[0]

print(f"\nNew sample: Color={new_color}, Diameter={new_diameter}cm")
print(f"Predicted Fruit: {predicted_fruit}")
print("\nProbabilities for each fruit type:")
for fruit, prob in zip(le_target.classes_, confidence):
    print(f"{fruit}: {prob:.4f}")
