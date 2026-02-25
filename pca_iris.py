import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('IRIS.csv')

# Extract features (first 4 columns)
feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
X = df[feature_columns].values
species = df['species'].values

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])
print("Number of samples:", X.shape[0])
print()

# Standardize the features (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("Data after standardization (first 5 samples):")
print("=" * 60)
print(X_scaled[:5])
print()

# Apply PCA with all components
pca = PCA(n_components=4)
X_pca = pca.fit_transform(X_scaled)

# Get eigenvalues (explained variance)
eigenvalues = pca.explained_variance_
print("=" * 60)
print("Eigenvalues (Explained Variance):")
print("=" * 60)
for i, eigenvalue in enumerate(eigenvalues):
    print(f"PC{i+1}: {eigenvalue:.6f}")
print()

# Get eigenvectors (principal components)
eigenvectors = pca.components_
print("=" * 60)
print("Eigenvectors (Principal Components):")
print("=" * 60)
for i, eigenvector in enumerate(eigenvectors):
    print(f"PC{i+1}: {eigenvector}")
print()

# Get explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_
print("=" * 60)
print("Explained Variance Ratio:")
print("=" * 60)
for i, ratio in enumerate(explained_variance_ratio):
    print(f"PC{i+1}: {ratio:.6f} ({ratio*100:.2f}%)")
print(f"Total: {sum(explained_variance_ratio):.6f}")
print()

# Cumulative explained variance
cumulative_variance = np.cumsum(explained_variance_ratio)
print("=" * 60)
print("Cumulative Explained Variance:")
print("=" * 60)
for i, cum_var in enumerate(cumulative_variance):
    print(f"PC1-PC{i+1}: {cum_var:.6f} ({cum_var*100:.2f}%)")
print()

# Reconstruction using PC1 only
print("=" * 60)
print("Reconstruction using PC1 only:")
print("=" * 60)
pca_1 = PCA(n_components=1)
X_pca_1 = pca_1.fit_transform(X_scaled)
X_reconstructed_1 = pca_1.inverse_transform(X_pca_1)
reconstruction_error_1 = np.mean(np.sum((X_scaled - X_reconstructed_1)**2, axis=1))
print(f"Reconstruction error (MSE): {reconstruction_error_1:.6f}")
print(f"Original data (first 3 samples):\n{X_scaled[:3]}")
print(f"\nReconstructed data (first 3 samples):\n{X_reconstructed_1[:3]}")
print()

# Reconstruction using PC1 and PC2
print("=" * 60)
print("Reconstruction using PC1 and PC2:")
print("=" * 60)
pca_2 = PCA(n_components=2)
X_pca_2 = pca_2.fit_transform(X_scaled)
X_reconstructed_2 = pca_2.inverse_transform(X_pca_2)
reconstruction_error_2 = np.mean(np.sum((X_scaled - X_reconstructed_2)**2, axis=1))
print(f"Reconstruction error (MSE): {reconstruction_error_2:.6f}")
print(f"Original data (first 3 samples):\n{X_scaled[:3]}")
print(f"\nReconstructed data (first 3 samples):\n{X_reconstructed_2[:3]}")
print()

# Reconstruction using PC1, PC2, and PC3
print("=" * 60)
print("Reconstruction using PC1, PC2, and PC3:")
print("=" * 60)
pca_3 = PCA(n_components=3)
X_pca_3 = pca_3.fit_transform(X_scaled)
X_reconstructed_3 = pca_3.inverse_transform(X_pca_3)
reconstruction_error_3 = np.mean(np.sum((X_scaled - X_reconstructed_3)**2, axis=1))
print(f"Reconstruction error (MSE): {reconstruction_error_3:.6f}")
print(f"Original data (first 3 samples):\n{X_scaled[:3]}")
print(f"\nReconstructed data (first 3 samples):\n{X_reconstructed_3[:3]}")
print()

# Visualization - Scree plot
plt.figure(figsize=(12, 5))

# Scree plot
plt.subplot(1, 2, 1)
plt.plot(range(1, 5), eigenvalues, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue (Explained Variance)')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)

# Cumulative explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, 5), cumulative_variance*100, 'ro-', linewidth=2, markersize=8)
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance (%)')
plt.title('Cumulative Explained Variance')
plt.grid(True, alpha=0.3)
plt.ylim([0, 105])

plt.tight_layout()
plt.savefig('pca_iris_analysis.png', dpi=100, bbox_inches='tight')
print("Saved plot as 'pca_iris_analysis.png'")

# 2D visualization
plt.figure(figsize=(10, 6))
colors = {'Iris-setosa': 'red', 'Iris-versicolor': 'green', 'Iris-virginica': 'blue'}
for species_name in colors:
    indices = species == species_name
    plt.scatter(X_pca_2[indices, 0], X_pca_2[indices, 1], 
                label=species_name, alpha=0.7, s=100, color=colors[species_name])

plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
plt.title('Iris Dataset - PCA (PC1 vs PC2)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_iris_2d.png', dpi=100, bbox_inches='tight')
print("Saved plot as 'pca_iris_2d.png'")

plt.show()

# Classification using Decision Tree on PC1 and PC2
print("\n" + "=" * 60)
print("Classification using Decision Tree on PC1 and PC2:")
print("=" * 60)

# Apply PCA with 2 components for classification
pca_2_clf = PCA(n_components=2)
X_pca_2_all = pca_2_clf.fit_transform(X_scaled)

# Encode species labels to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(species)

# Ensure y is a 1D array
y = np.asarray(y).ravel()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca_2_all, y, test_size=0.3, random_state=42)

# Ensure arrays are properly formatted
X_train = np.asarray(X_train)
X_test = np.asarray(X_test)
y_train = np.asarray(y_train).ravel()
y_test = np.asarray(y_test).ravel()

# Train Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42, max_depth=5)
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)
y_pred = np.asarray(y_pred).ravel()

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy Score: {accuracy:.4f}")
print()

# Confusion matrix
print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print()

# Classification report
print("Classification Report:")
target_names = [str(c) for c in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# Feature importance
print("\nFeature Importance:")
print(f"PC1: {dt_classifier.feature_importances_[0]:.4f}")
print(f"PC2: {dt_classifier.feature_importances_[1]:.4f}")

# Visualize Decision Tree predictions
plt.figure(figsize=(12, 6))

# Original data with true labels
plt.subplot(1, 2, 1)
for i, species_name in enumerate(le.classes_):
    mask = (y_test == i)
    if np.any(mask):
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                    label=f'{species_name} (True)', alpha=0.7, s=100)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
plt.title('Test Data - True Labels')
plt.legend()
plt.grid(True, alpha=0.3)

# Data with predicted labels
plt.subplot(1, 2, 2)
for i, species_name in enumerate(le.classes_):
    mask = (y_pred == i)
    if np.any(mask):
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                    label=f'{species_name} (Predicted)', alpha=0.7, s=100)
plt.xlabel(f'PC1 ({explained_variance_ratio[0]*100:.2f}%)')
plt.ylabel(f'PC2 ({explained_variance_ratio[1]*100:.2f}%)')
plt.title('Test Data - Predicted Labels')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_iris_classification.png', dpi=100, bbox_inches='tight')
print("\nSaved plot as 'pca_iris_classification.png'")
