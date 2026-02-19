import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the IRIS dataset
df = pd.read_csv('IRIS.csv')

# Extract only the feature columns (ignore species)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# Apply K-means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X)

# Add cluster labels to dataframe
df['cluster'] = clusters

# Display the results
print("K-means Clustering Results (k=3)")
print("=" * 50)
print(f"\nCluster Centers:\n{kmeans.cluster_centers_}")
print(f"\nInertia (sum of squared distances): {kmeans.inertia_:.2f}")
print(f"\nNumber of samples in each cluster:")
print(df['cluster'].value_counts().sort_index())

# Display first few rows with cluster assignments
print("\nFirst 10 rows with cluster assignments:")
print(df.head(10))

# Optional: Visualize clusters (2D projection using first two features)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            c='red', marker='X', s=200, edgecolors='black', label='Centroids')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title('K-means Clustering (k=3) - IRIS Dataset')
plt.colorbar(scatter, label='Cluster')
plt.legend()
plt.grid(True)
plt.savefig('kmeans_clusters.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'kmeans_clusters.png'")

# =====================================================
# AGGLOMERATIVE CLUSTERING (Hierarchical Clustering)
# =====================================================
print("\n\n" + "=" * 50)
print("Agglomerative Clustering Results (k=3)")
print("=" * 50)

# Apply Agglomerative Clustering with k=3
agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')
agg_clusters = agg_clustering.fit_predict(X)

# Add cluster labels to dataframe
df['agg_cluster'] = agg_clusters

# Display the results
print(f"\nNumber of samples in each cluster:")
print(df['agg_cluster'].value_counts().sort_index())

# Display first few rows with cluster assignments
print("\nFirst 10 rows with agglomerative cluster assignments:")
print(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'agg_cluster']].head(10))

# Visualize agglomerative clusters
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Scatter plot of agglomerative clusters
scatter = axes[0].scatter(X[:, 0], X[:, 1], c=agg_clusters, cmap='viridis', edgecolors='k')
axes[0].set_xlabel('sepal_length')
axes[0].set_ylabel('sepal_width')
axes[0].set_title('Agglomerative Clustering (k=3) - IRIS Dataset')
axes[0].grid(True)
plt.colorbar(scatter, ax=axes[0], label='Cluster')

# Dendrogram
Z = linkage(X, method='ward')
dendrogram(Z, ax=axes[1], no_labels=True)
axes[1].set_title('Dendrogram - Hierarchical Clustering')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Distance')

plt.tight_layout()
plt.savefig('agglomerative_clusters.png', dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualization saved as 'agglomerative_clusters.png'")

# Comparison of K-means and Agglomerative Clustering
print("\n\n" + "=" * 50)
print("Comparison: K-means vs Agglomerative Clustering")
print("=" * 50)
comparison_df = df[['sepal_length', 'sepal_width', 'cluster', 'agg_cluster']].head(20)
print("\nFirst 20 samples - Cluster assignments comparison:")
print(comparison_df)
