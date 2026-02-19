import numpy as np
from sklearn.decomposition import PCA
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
pca = PCA(n_components=2)
pca.fit(X)
print(pca.explained_variance_ratio_)
print(pca.singular_values_)
X_pca = pca.transform(X)
print(X_pca)
# Plotting the PCA result
import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', marker='o')
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()

# Plot original data
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c='green', marker='s')
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()

# Inverse transform to original space
X_original = pca.inverse_transform(X_pca)
print(X_original)

# Plotting the inverse transformed data
#add original data points for comparison
plt.figure()
plt.scatter(X_original[:, 0], X_original[:, 1], c='red', marker='x', label='Inverse')
plt.scatter(X[:, 0], X[:, 1], c='green', marker='s', label='Original')
plt.legend()
plt.title('Inverse Transformed Data with Original')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.grid()
plt.show()
# Explained variance for each principal component
explained_variance = pca.explained_variance_
print(explained_variance)