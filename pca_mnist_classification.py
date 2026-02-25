import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.datasets import load_digits
import seaborn as sns

# Load MNIST-like dataset (scikit-learn digits dataset - 8x8 images of digits 0-9)
print("=" * 60)
print("PCA for MNIST Classification using Decision Tree")
print("=" * 60)

digits = load_digits()
X = digits.data  # 1797 samples, 64 features (8x8 pixels)
y = digits.target  # labels 0-9

print(f"Dataset shape: {X.shape}")
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features (pixels): {X.shape[1]}")
print(f"Number of classes: {len(np.unique(y))}")
print()

# Visualize sample images from the dataset
print("=" * 60)
print("Sample Images from MNIST Dataset:")
print("=" * 60)
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in range(10):
    image = X[i].reshape(8, 8)
    axes[i].imshow(image, cmap='gray')
    axes[i].set_title(f'Digit: {y[i]}')
    axes[i].axis('off')

plt.suptitle('Sample Handwritten Digits from MNIST', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print()

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA with different number of components
print("=" * 60)
print("PCA Analysis:")
print("=" * 60)

pca_full = PCA(n_components=64)
pca_full.fit(X_scaled)

# Get explained variance ratio
explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

print(f"Explained Variance by each component:")
for i in range(10):
    print(f"PC{i+1}: {explained_variance_ratio[i]:.6f} ({explained_variance_ratio[i]*100:.2f}%)")
print()

print(f"Cumulative Explained Variance:")
print(f"First 5 components: {cumulative_variance[4]:.6f} ({cumulative_variance[4]*100:.2f}%)")
print(f"First 10 components: {cumulative_variance[9]:.6f} ({cumulative_variance[9]*100:.2f}%)")
print(f"First 20 components: {cumulative_variance[19]:.6f} ({cumulative_variance[19]*100:.2f}%)")
print(f"First 30 components: {cumulative_variance[29]:.6f} ({cumulative_variance[29]*100:.2f}%)")
print()

# Visualize Scree plot and cumulative variance
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, 65), explained_variance_ratio, 'bo-', linewidth=1, markersize=4)
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('Scree Plot - MNIST Dataset')
ax1.grid(True, alpha=0.3)

ax2.plot(range(1, 65), cumulative_variance*100, 'ro-', linewidth=2, markersize=4)
ax2.axhline(y=95, color='g', linestyle='--', label='95% variance')
ax2.set_xlabel('Number of Principal Components')
ax2.set_ylabel('Cumulative Explained Variance (%)')
ax2.set_title('Cumulative Explained Variance - MNIST Dataset')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Classification with different number of PCA components
print("=" * 60)
print("Classification Performance:")
print("=" * 60)

components_to_test = [5, 10, 15, 20, 30, 50, 64]
results = {}

for n_comp in components_to_test:
    print(f"\nUsing {n_comp} principal components...")
    
    # Apply PCA
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
    
    # Train Decision Tree classifier
    dt = DecisionTreeClassifier(max_depth=20, random_state=42)
    dt.fit(X_train, y_train)
    
    # Predict
    y_pred = dt.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    results[n_comp] = accuracy
    print(f"  Accuracy: {accuracy:.4f}")

print()

# Plot accuracy vs components
plt.figure(figsize=(10, 6))
plt.plot(list(results.keys()), list(results.values()), 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Principal Components')
plt.ylabel('Classification Accuracy')
plt.title('Decision Tree Classification Accuracy vs PCA Components')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Use optimal number of components (30 - good balance)
n_optimal = 30
print("=" * 60)
print(f"Detailed Classification with {n_optimal} components:")
print("=" * 60)

pca_optimal = PCA(n_components=n_optimal)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_pca_optimal, y, test_size=0.2, random_state=42)

# Train Decision Tree
dt_optimal = DecisionTreeClassifier(max_depth=20, random_state=42)
dt_optimal.fit(X_train, y_train)

# Predict
y_pred_optimal = dt_optimal.predict(X_test)

# Accuracy
accuracy_optimal = accuracy_score(y_test, y_pred_optimal)
print(f"\nAccuracy: {accuracy_optimal:.4f}")
print()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_optimal)
print("Confusion Matrix:")
print(cm)
print()

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred_optimal))

# Visualize Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title(f'Confusion Matrix - MNIST with {n_optimal} PCA Components')
plt.tight_layout()
plt.show()

# Visualize some principal components as images
print("=" * 60)
print("Visualization of Principal Components:")
print("=" * 60)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.ravel()

for i in range(10):
    # Reshape component to 8x8 image
    component_image = pca_optimal.components_[i].reshape(8, 8)
    axes[i].imshow(component_image, cmap='gray')
    axes[i].set_title(f'PC{i+1} ({explained_variance_ratio[i]*100:.2f}%)')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Visualize some original digits and their reconstruction
print("\nVisualizing original digits and reconstruction...")

fig, axes = plt.subplots(3, 5, figsize=(15, 9))

# Test with first 5 samples from each visualization
test_indices = [0, 1, 2, 3, 4]

for idx, test_idx in enumerate(test_indices):
    # Original image
    original_image = X[test_idx].reshape(8, 8)
    axes[0, idx].imshow(original_image, cmap='gray')
    axes[0, idx].set_title(f'Original (Digit {y[test_idx]})')
    axes[0, idx].axis('off')
    
    # PCA transformed and reconstructed
    with_pca = X_pca_optimal[:, :30]
    pca_reconstructed = pca_optimal.inverse_transform(pca_optimal.transform(X_scaled[test_idx:test_idx+1]))
    reconstructed_image = pca_reconstructed[0].reshape(8, 8)
    axes[1, idx].imshow(reconstructed_image, cmap='gray')
    axes[1, idx].set_title(f'Reconstructed ({n_optimal} PC)')
    axes[1, idx].axis('off')
    
    # Difference
    difference = np.abs(original_image - reconstructed_image)
    axes[2, idx].imshow(difference, cmap='hot')
    axes[2, idx].set_title(f'Difference')
    axes[2, idx].axis('off')

plt.tight_layout()
plt.show()

# Summary
print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print(f"Original dataset size: {X.shape[1]} features")
print(f"Reduced to: {n_optimal} principal components")
print(f"Dimensionality reduction: {(1 - n_optimal/X.shape[1])*100:.2f}%")
print(f"Variance retained: {cumulative_variance[n_optimal-1]*100:.2f}%")
print(f"Classification Accuracy: {accuracy_optimal:.4f}")
print()
print("All visualizations displayed successfully!")
