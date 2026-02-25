import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 1], [1, 3], [4, 5]])

# Fit PCA
pca = PCA(n_components=2)
pca.fit(X)

# Get eigenvalues (explained variance)
eigenvalues = pca.explained_variance_
print("Eigenvalues:")
print(eigenvalues)
print()

# Get eigenvectors (principal components)
eigenvectors = pca.components_
print("Eigenvectors (Principal Components):")
print(eigenvectors)
print()

# Get explained variance ratio (probability for each eigenvalue)
explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio (Probability):")
print(explained_variance_ratio)
print()

# Get scores (transformed data for each eigenvector)
scores = pca.fit_transform(X)
print("Scores (Transformed Data):")
print(scores)
print()

# Summary with combined information
print("=" * 60)
print("Summary of Eigenvalues with Probabilities and Scores:")
print("=" * 60)
for i in range(len(eigenvalues)):
    print(f"\nEigenvalue {i+1}: {eigenvalues[i]:.6f}")
    print(f"Eigenvector {i+1}: {eigenvectors[i]}")
    print(f"Explained Variance Ratio (Probability): {explained_variance_ratio[i]:.6f}")
    print(f"Scores for this component: {scores[:, i]}")

# Reconstruct original data using only PC1
print("\n" + "=" * 60)
print("Reconstruction using only PC1:")
print("=" * 60)
pc1_scores = scores[:, 0].reshape(-1, 1)  # Get only PC1 scores
pc1_eigenvector = eigenvectors[0].reshape(1, -1)  # Get only PC1 eigenvector
reconstructed_data_pc1 = pc1_scores @ pc1_eigenvector
print(f"\nOriginal data:\n{X}")
print(f"\nReconstructed data using PC1 alone:\n{reconstructed_data_pc1}")

# Reconstruct original data using PC1 and PC2
print("\n" + "=" * 60)
print("Reconstruction using PC1 and PC2:")
print("=" * 60)
pc1_pc2_scores = scores[:, [0, 1]]  # Get PC1 and PC2 scores
pc1_pc2_eigenvectors = eigenvectors[[0, 1], :]  # Get PC1 and PC2 eigenvectors
reconstructed_data_pc1_pc2 = pc1_pc2_scores @ pc1_pc2_eigenvectors
print(f"\nOriginal data:\n{X}")
print(f"\nReconstructed data using PC1 and PC2:\n{reconstructed_data_pc1_pc2}")