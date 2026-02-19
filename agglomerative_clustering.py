import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Load the IRIS dataset
df = pd.read_csv('IRIS.csv')

# Extract only the feature columns (ignore species)
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values

# =====================================================
# AGGLOMERATIVE CLUSTERING (Hierarchical Clustering)
# =====================================================
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

# Display full cluster assignments
print("\n" + "=" * 50)
print("All cluster assignments:")
print(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species', 'agg_cluster']])
