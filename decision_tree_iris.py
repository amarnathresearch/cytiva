import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# DECISION TREE CLASSIFICATION - IRIS DATASET
# =====================================================

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

print("=" * 60)
print("DECISION TREE CLASSIFICATION - IRIS DATASET")
print("=" * 60)
print(f"\nTotal Samples: {len(df)}")
print("Dataset shape:", df.shape)
print("\nFirst 10 rows:")
print(df.head(10))
print("\nDataset Info:")
print(df.info())
print("\nIris species distribution:")
print(df['target_name'].value_counts())
print("\nDataset Statistics:")
print(df.describe())

# Create feature matrix X and target vector y
X = df.iloc[:, :-2]  # All feature columns except target and target_name
y = df['target']

feature_names = iris.feature_names
target_names = iris.target_names

print("\n" + "=" * 60)
print("FEATURES AND TARGETS")
print("=" * 60)
print("\nFeatures:")
for i, name in enumerate(feature_names):
    print(f"{i+1}. {name}")
print("\nTarget classes:")
for i, name in enumerate(target_names):
    print(f"{i}: {name}")

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Train Decision Tree Classifier
print("\n" + "=" * 60)
print("TRAINING DECISION TREE MODEL")
print("=" * 60)

dt_model = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)
dt_model.fit(X_train, y_train)

print("\nModel trained successfully!")
print(f"Max depth: {dt_model.max_depth}")
print(f"Number of leaves: {dt_model.get_n_leaves()}")
print(f"Tree depth: {dt_model.get_depth()}")

# Make predictions
y_pred_train = dt_model.predict(X_train)
y_pred_test = dt_model.predict(X_test)

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
    target_names=target_names
))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(cm)

# Feature Importance
feature_importance = dt_model.feature_importances_

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
for name, importance in zip(feature_names, feature_importance):
    print(f"{name}: {importance:.4f}")

# Visualizations
fig = plt.figure(figsize=(16, 12))

# 1. Decision Tree Visualization
ax1 = plt.subplot(2, 2, 1)
plot_tree(dt_model, 
          feature_names=feature_names,
          class_names=target_names,
          filled=True,
          ax=ax1,
          fontsize=8)
ax1.set_title('Decision Tree Structure', fontsize=12, fontweight='bold')

# 2. Feature Importance
ax2 = plt.subplot(2, 2, 2)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
ax2.barh(feature_names, feature_importance, color=colors)
ax2.set_xlabel('Importance', fontsize=10)
ax2.set_title('Feature Importance', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# 3. Confusion Matrix Heatmap
ax3 = plt.subplot(2, 2, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3,
            xticklabels=target_names,
            yticklabels=target_names,
            cbar_kws={'label': 'Count'})
ax3.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label', fontsize=10)
ax3.set_xlabel('Predicted Label', fontsize=10)

# 4. Accuracy Comparison
ax4 = plt.subplot(2, 2, 4)
accuracies = [train_accuracy, test_accuracy]
bars = ax4.bar(['Training', 'Testing'], accuracies, color=['#2ca02c', '#ff7f0e'], width=0.6)
ax4.set_ylabel('Accuracy', fontsize=10)
ax4.set_title('Model Accuracy Comparison', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1])
ax4.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(accuracies):
    ax4.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('decision_tree_iris.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'decision_tree_iris.png'")

# Sample predictions
print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print("\nPredictions on test samples:")
for i in range(min(5, len(X_test))):
    sepal_length = X_test.iloc[i][0]
    sepal_width = X_test.iloc[i][1]
    petal_length = X_test.iloc[i][2]
    petal_width = X_test.iloc[i][3]
    
    true_class = target_names[y_test.iloc[i]]
    pred_class = target_names[y_pred_test[i]]
    confidence = dt_model.predict_proba(X_test.iloc[i].values.reshape(1, -1)).max()
    
    print(f"\nSample {i+1}:")
    print(f"  Sepal Length: {sepal_length:.2f}, Sepal Width: {sepal_width:.2f}")
    print(f"  Petal Length: {petal_length:.2f}, Petal Width: {petal_width:.2f}")
    print(f"  True: {true_class}, Predicted: {pred_class}, Confidence: {confidence:.4f}")

# Prediction on new data
print("\n" + "=" * 60)
print("PREDICTION ON NEW DATA")
print("=" * 60)

# Example: new iris sample
new_sample_values = np.array([[5.5, 3.0, 4.2, 1.3]])
prediction = dt_model.predict(new_sample_values)[0]
confidence = dt_model.predict_proba(new_sample_values)[0]
predicted_species = target_names[prediction]

print(f"\nNew sample:")
print(f"  Sepal Length: 5.5, Sepal Width: 3.0")
print(f"  Petal Length: 4.2, Petal Width: 1.3")
print(f"  Predicted Species: {predicted_species}")
print("\nProbabilities for each species:")
for species, prob in zip(target_names, confidence):
    print(f"  {species}: {prob:.4f}")
