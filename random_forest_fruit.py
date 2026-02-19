import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# RANDOM FOREST CLASSIFICATION - FRUIT DATASET
# =====================================================

# Create a sample fruit dataset
# Features: Weight (g), Color_Score (0-10), Sweetness (0-10), Firmness (0-10)
np.random.seed(42)

n_samples = 200

# Apple data
apple_weight = np.random.normal(180, 30, n_samples // 3)
apple_color = np.random.normal(7, 1.5, n_samples // 3)
apple_sweetness = np.random.normal(6, 1.5, n_samples // 3)
apple_firmness = np.random.normal(8, 1, n_samples // 3)

# Banana data
banana_weight = np.random.normal(120, 20, n_samples // 3)
banana_color = np.random.normal(5, 1.5, n_samples // 3)
banana_sweetness = np.random.normal(8, 1.2, n_samples // 3)
banana_firmness = np.random.normal(5, 1, n_samples // 3)

# Orange data
orange_weight = np.random.normal(150, 25, n_samples // 3)
orange_color = np.random.normal(8, 1, n_samples // 3)
orange_sweetness = np.random.normal(7, 1.5, n_samples // 3)
orange_firmness = np.random.normal(6, 1.2, n_samples // 3)

# Combine data
X = np.vstack([
    np.column_stack([apple_weight, apple_color, apple_sweetness, apple_firmness]),
    np.column_stack([banana_weight, banana_color, banana_sweetness, banana_firmness]),
    np.column_stack([orange_weight, orange_color, orange_sweetness, orange_firmness])
])

y = np.hstack([
    np.zeros(n_samples // 3),
    np.ones(n_samples // 3),
    np.full(n_samples // 3, 2)
])

# Create DataFrame
df = pd.DataFrame(X, columns=['Weight (g)', 'Color_Score', 'Sweetness', 'Firmness'])
df['Fruit'] = y
fruit_names = {0: 'Apple', 1: 'Banana', 2: 'Orange'}
df['Fruit_Name'] = df['Fruit'].map(fruit_names)

print("=" * 60)
print("RANDOM FOREST CLASSIFICATION - FRUIT DATASET")
print("=" * 60)
print("\nDataset shape:", df.shape)
print("\nFirst 10 rows of the dataset:")
print(df.head(10))
print("\nFruit distribution:")
print(df['Fruit_Name'].value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train Random Forest Classifier
print("\n" + "=" * 60)
print("TRAINING RANDOM FOREST MODEL")
print("=" * 60)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
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
print(classification_report(y_test, y_pred_test, target_names=['Apple', 'Banana', 'Orange']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance
feature_importance = rf_model.feature_importances_
feature_names = ['Weight (g)', 'Color_Score', 'Sweetness', 'Firmness']

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
            xticklabels=['Apple', 'Banana', 'Orange'],
            yticklabels=['Apple', 'Banana', 'Orange'])
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

# 4. Weight vs Sweetness (colored by fruit)
colors = {0: 'red', 1: 'yellow', 2: 'orange'}
for fruit_id, fruit_name in fruit_names.items():
    mask = df['Fruit'] == fruit_id
    axes[1, 1].scatter(df[mask]['Weight (g)'], df[mask]['Sweetness'], 
                      label=fruit_name, alpha=0.6, s=100, color=colors[fruit_id])
axes[1, 1].set_xlabel('Weight (g)')
axes[1, 1].set_ylabel('Sweetness')
axes[1, 1].set_title('Fruit Distribution: Weight vs Sweetness')
axes[1, 1].legend()
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('random_forest_fruit.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'random_forest_fruit.png'")

# Sample prediction
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)
print("\nSample fruits from test set:")
sample_indices = np.random.choice(len(X_test), 5, replace=False)
for idx in sample_indices:
    true_label = fruit_names[y_test[idx]]
    pred_label = fruit_names[y_pred_test[idx]]
    confidence = rf_model.predict_proba(X_test[idx].reshape(1, -1)).max()
    print(f"\nWeight: {X_test[idx][0]:.1f}g, Color: {X_test[idx][1]:.1f}, Sweetness: {X_test[idx][2]:.1f}, Firmness: {X_test[idx][3]:.1f}")
    print(f"True Fruit: {true_label}, Predicted: {pred_label}, Confidence: {confidence:.4f}")
