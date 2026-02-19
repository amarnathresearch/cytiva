"""
K-Means Clustering Example using scikit-learn
This script demonstrates k-means clustering on sample data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate sample data
# Create 300 samples with 4 clusters
X, y_true = make_blobs(n_samples=300, centers=4, n_features=2, 
                        cluster_std=0.60, random_state=42)

print("Dataset shape:", X.shape)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

# Initialize and fit K-Means model
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
y_kmeans = kmeans.fit_predict(X)

print("\nK-Means Results:")
print("Cluster centers:\n", kmeans.cluster_centers_)
print("Inertia (sum of squared distances):", kmeans.inertia_)
print("Number of iterations:", kmeans.n_iter_)

# Get cluster labels and centers
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Print cluster information
print("\nCluster Information:")
for i in range(4):
    cluster_points = np.sum(labels == i)
    print(f"Cluster {i}: {cluster_points} points")

# Visualization
plt.figure(figsize=(10, 6))

# Plot the clusters
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis', alpha=0.6)

# Plot the cluster centers
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.8, 
            marker='X', edgecolors='black', linewidth=2, label='Centroids')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering (k=4)')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/opt/cytiva/kmeans_plot.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nPlot saved as 'kmeans_plot.png'")

# Elbow method - find optimal number of clusters
inertias = []
silhouette_scores = []
K_range = range(1, 11)

from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_temp.fit(X)
    inertias.append(kmeans_temp.inertia_)
    if k > 1:
        silhouette_scores.append(silhouette_score(X, kmeans_temp.labels_))

# Plot Elbow curve
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_range, inertias, 'bo-')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia')
ax1.set_title('Elbow Method')
ax1.grid(True, alpha=0.3)

ax2.plot(range(2, 11), silhouette_scores, 'go-')
ax2.set_xlabel('Number of Clusters (k)')
ax2.set_ylabel('Silhouette Score')
ax2.set_title('Silhouette Score Analysis')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/opt/cytiva/elbow_silhouette.png', dpi=100, bbox_inches='tight')
plt.show()

print("Elbow and Silhouette plots saved as 'elbow_silhouette.png'")

# Predict cluster for new points
new_points = np.array([[0, 0], [5, 5]])
predictions = kmeans.predict(new_points)
print("\nNew point predictions:")
for point, cluster in zip(new_points, predictions):
    print(f"Point {point} belongs to cluster {cluster}")
