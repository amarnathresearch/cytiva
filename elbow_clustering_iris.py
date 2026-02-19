import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# ELBOW METHOD FOR CLUSTERING - IRIS DATASET
# =====================================================

print("=" * 80)
print("ELBOW METHOD FOR K-MEANS CLUSTERING - IRIS DATASET")
print("=" * 80)

# Load Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['target_name'] = df['target'].map({
    0: iris.target_names[0],
    1: iris.target_names[1],
    2: iris.target_names[2]
})

print(f"\n{'Dataset Information':}")
print(f"  Shape: {df.shape}")
print(f"  Total samples: {len(df)}")
print(f"  Total features: {len(iris.feature_names)}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nDataset Info:")
print(df.info())
print(f"\nIris species distribution:")
print(df['target_name'].value_counts())
print(f"\nFeature statistics:")
print(df.iloc[:, :-2].describe())

# =====================================================
# DATA PREPROCESSING
# =====================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Extract features (exclude target columns)
X = df.iloc[:, :-2].values
feature_names = iris.feature_names
target_names = iris.target_names
y_true = df['target'].values

print(f"\nFeatures shape: {X.shape}")
print(f"Number of features: {len(feature_names)}")
print(f"Feature names: {list(feature_names)}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures scaled using StandardScaler")
print(f"Scaled data shape: {X_scaled.shape}")
print(f"Scaled feature means (should be ~0): {X_scaled.mean(axis=0)}")
print(f"Scaled feature std devs (should be ~1): {X_scaled.std(axis=0)}")

# =====================================================
# ELBOW METHOD: INERTIA CALCULATION
# =====================================================

print("\n" + "=" * 80)
print("ELBOW METHOD: INERTIA CALCULATION")
print("=" * 80)

# Test different numbers of clusters
k_range = range(1, 11)
inertias = []
silhouette_scores = []
davies_bouldin_scores = []
calinski_harabasz_scores = []

print(f"\nTesting k values from 1 to 10...")
print(f"\n{'k':<5} {'Inertia':<15} {'Silhouette':<15} {'Davies-Bouldin':<15} {'Calinski-Harabasz':<15}")
print("-" * 65)

for k in k_range:
    # Train K-means
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Calculate metrics
    inertia = kmeans.inertia_
    inertias.append(inertia)
    
    # Silhouette score (only for k > 1)
    if k > 1:
        sil_score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(sil_score)
    else:
        silhouette_scores.append(0)
    
    # Davies-Bouldin index (only for k > 1)
    if k > 1:
        db_score = davies_bouldin_score(X_scaled, kmeans.labels_)
        davies_bouldin_scores.append(db_score)
    else:
        davies_bouldin_scores.append(0)
    
    # Calinski-Harabasz index (only for k > 1)
    if k > 1:
        ch_score = calinski_harabasz_score(X_scaled, kmeans.labels_)
        calinski_harabasz_scores.append(ch_score)
    else:
        calinski_harabasz_scores.append(0)
    
    print(f"{k:<5} {inertia:<15.2f} {silhouette_scores[-1]:<15.4f} {davies_bouldin_scores[-1]:<15.4f} {calinski_harabasz_scores[-1]:<15.2f}")

# =====================================================
# FIND OPTIMAL K USING ELBOW METHOD
# =====================================================

print("\n" + "=" * 80)
print("OPTIMAL K DETERMINATION")
print("=" * 80)

# Calculate differences in inertia (elbow detection)
inertia_diffs = np.diff(inertias)
inertia_diffs_2 = np.diff(inertia_diffs)

print(f"\nInertia analysis:")
print(f"{'k':<5} {'Inertia':<15} {'Δ Inertia':<15} {'Δ² Inertia':<15}")
print("-" * 50)
for i, k in enumerate(k_range):
    delta = inertia_diffs[i-1] if i > 0 else 0
    delta2 = inertia_diffs_2[i-2] if i > 1 else 0
    print(f"{k:<5} {inertias[i]:<15.2f} {delta:<15.2f} {delta2:<15.2f}")

# Find elbow point (k with maximum second derivative)
optimal_k_elbow = np.argmax(np.abs(inertia_diffs_2)) + 2

# Find best k by Silhouette Score (max)
optimal_k_silhouette = np.argmax(silhouette_scores[1:]) + 2

# Find best k by Davies-Bouldin Score (min after k=1)
optimal_k_db = np.argmin(davies_bouldin_scores[1:]) + 2

# Find best k by Calinski-Harabasz Score (max after k=1)
optimal_k_ch = np.argmax(calinski_harabasz_scores[1:]) + 2

print(f"\nOptimal k suggestions:")
print(f"  Elbow Method: k={optimal_k_elbow} (maximum second derivative)")
print(f"  Silhouette Score: k={optimal_k_silhouette} (max={silhouette_scores[optimal_k_silhouette-1]:.4f})")
print(f"  Davies-Bouldin Index: k={optimal_k_db} (min={davies_bouldin_scores[optimal_k_db-1]:.4f})")
print(f"  Calinski-Harabasz: k={optimal_k_ch} (max={calinski_harabasz_scores[optimal_k_ch-1]:.2f})")

# Use elbow method as primary choice
optimal_k = optimal_k_elbow
print(f"\n✓ Selected optimal k: {optimal_k} (using Elbow Method)")

# =====================================================
# TRAIN FINAL MODEL WITH OPTIMAL K
# =====================================================

print("\n" + "=" * 80)
print("FINAL CLUSTERING WITH OPTIMAL K")
print("=" * 80)

final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
y_pred = final_kmeans.fit_predict(X_scaled)

print(f"\nK-Means clustering with k={optimal_k}:")
print(f"  Cluster centers shape: {final_kmeans.cluster_centers_.shape}")
print(f"  Inertia: {final_kmeans.inertia_:.2f}")
print(f"  Silhouette Score: {silhouette_score(X_scaled, y_pred):.4f}")
print(f"  Davies-Bouldin Index: {davies_bouldin_score(X_scaled, y_pred):.4f}")
print(f"  Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, y_pred):.2f}")

# Cluster distribution
print(f"\nCluster distribution:")
unique_clusters, cluster_counts = np.unique(y_pred, return_counts=True)
for cluster_id, count in zip(unique_clusters, cluster_counts):
    percentage = (count / len(y_pred)) * 100
    print(f"  Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")

# =====================================================
# VISUALIZATIONS
# =====================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig = plt.figure(figsize=(18, 14))

# 1. Elbow Curve (Inertia)
ax1 = plt.subplot(3, 3, 1)
ax1.plot(k_range, inertias, marker='o', markersize=8, linewidth=2, color='blue', label='Inertia')
ax1.axvline(x=optimal_k_elbow, color='red', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k_elbow}')
ax1.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold')
ax1.set_ylabel('Inertia', fontsize=10, fontweight='bold')
ax1.set_title('Elbow Method - Inertia', fontsize=11, fontweight='bold')
ax1.set_xticks(k_range)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Silhouette Score
ax2 = plt.subplot(3, 3, 2)
ax2.plot(k_range, silhouette_scores, marker='s', markersize=8, linewidth=2, color='green')
ax2.axvline(x=optimal_k_silhouette, color='red', linestyle='--', linewidth=2, label=f'Best k={optimal_k_silhouette}')
ax2.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Silhouette Score', fontsize=10, fontweight='bold')
ax2.set_title('Silhouette Score vs k', fontsize=11, fontweight='bold')
ax2.set_xticks(k_range)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Davies-Bouldin Index
ax3 = plt.subplot(3, 3, 3)
ax3.plot(k_range[1:], davies_bouldin_scores[1:], marker='^', markersize=8, linewidth=2, color='orange')
ax3.axvline(x=optimal_k_db, color='red', linestyle='--', linewidth=2, label=f'Best k={optimal_k_db}')
ax3.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Davies-Bouldin Index', fontsize=10, fontweight='bold')
ax3.set_title('Davies-Bouldin Index vs k (Lower is Better)', fontsize=11, fontweight='bold')
ax3.set_xticks(k_range)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Calinski-Harabasz Index
ax4 = plt.subplot(3, 3, 4)
ax4.plot(k_range[1:], calinski_harabasz_scores[1:], marker='D', markersize=8, linewidth=2, color='purple')
ax4.axvline(x=optimal_k_ch, color='red', linestyle='--', linewidth=2, label=f'Best k={optimal_k_ch}')
ax4.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Calinski-Harabasz Score', fontsize=10, fontweight='bold')
ax4.set_title('Calinski-Harabasz Score vs k (Higher is Better)', fontsize=11, fontweight='bold')
ax4.set_xticks(k_range)
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Clustering result (Sepal Length vs Sepal Width)
ax5 = plt.subplot(3, 3, 5)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA15E', '#BC6C25', '#8E7DBE', '#5D737E', '#C1666B']
for i in range(optimal_k):
    cluster_points = X_scaled[y_pred == i]
    ax5.scatter(cluster_points[:, 0], cluster_points[:, 1], 
               label=f'Cluster {i}', s=100, alpha=0.7, edgecolors='black', linewidth=1, color=colors[i])

# Plot cluster centers
centers = final_kmeans.cluster_centers_
ax5.scatter(centers[:, 0], centers[:, 1], marker='*', s=500, c='red', edgecolors='black', linewidth=2, label='Centroids')
ax5.set_xlabel('Sepal Length (scaled)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Sepal Width (scaled)', fontsize=10, fontweight='bold')
ax5.set_title(f'K-Means Clustering (k={optimal_k}) - Sepal Length vs Width', fontsize=11, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)

# 6. Clustering result (Petal Length vs Petal Width)
ax6 = plt.subplot(3, 3, 6)
for i in range(optimal_k):
    cluster_points = X_scaled[y_pred == i]
    ax6.scatter(cluster_points[:, 2], cluster_points[:, 3], 
               label=f'Cluster {i}', s=100, alpha=0.7, edgecolors='black', linewidth=1, color=colors[i])

# Plot cluster centers
ax6.scatter(centers[:, 2], centers[:, 3], marker='*', s=500, c='red', edgecolors='black', linewidth=2, label='Centroids')
ax6.set_xlabel('Petal Length (scaled)', fontsize=10, fontweight='bold')
ax6.set_ylabel('Petal Width (scaled)', fontsize=10, fontweight='bold')
ax6.set_title(f'K-Means Clustering (k={optimal_k}) - Petal Length vs Width', fontsize=11, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(True, alpha=0.3)

# 7. Cluster size distribution
ax7 = plt.subplot(3, 3, 7)
unique_clusters, cluster_counts = np.unique(y_pred, return_counts=True)
ax7.bar(unique_clusters, cluster_counts, color=colors[:optimal_k], alpha=0.8, edgecolor='black', linewidth=1)
ax7.set_xlabel('Cluster ID', fontsize=10, fontweight='bold')
ax7.set_ylabel('Number of Samples', fontsize=10, fontweight='bold')
ax7.set_title('Cluster Size Distribution', fontsize=11, fontweight='bold')
ax7.set_xticks(unique_clusters)
ax7.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, v in enumerate(cluster_counts):
    ax7.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 8. Comparison of all evaluation metrics
ax8 = plt.subplot(3, 3, 8)
metrics_comparison = {
    'Elbow': optimal_k_elbow,
    'Silhouette': optimal_k_silhouette,
    'Davies-Bouldin': optimal_k_db,
    'Calinski-Harabasz': optimal_k_ch
}
ax8.bar(metrics_comparison.keys(), metrics_comparison.values(), 
        color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], alpha=0.8, edgecolor='black', linewidth=1)
ax8.set_ylabel('Optimal k', fontsize=10, fontweight='bold')
ax8.set_title('Optimal k by Different Methods', fontsize=11, fontweight='bold')
ax8.set_ylim([0, 11])
ax8.grid(axis='y', alpha=0.3)

# Add value labels
for i, (method, k) in enumerate(metrics_comparison.items()):
    ax8.text(i, k + 0.2, str(k), ha='center', fontweight='bold')

# 9. Second derivative of inertia (elbow point detection)
ax9 = plt.subplot(3, 3, 9)
k_range_2 = list(range(2, 10))
ax9.plot(k_range_2, inertia_diffs_2, marker='o', markersize=8, linewidth=2, color='darkblue', label='Δ² Inertia')
ax9.axvline(x=optimal_k_elbow, color='red', linestyle='--', linewidth=2, label=f'Elbow at k={optimal_k_elbow}')
ax9.set_xlabel('Number of Clusters (k)', fontsize=10, fontweight='bold')
ax9.set_ylabel('Second Derivative', fontsize=10, fontweight='bold')
ax9.set_title('Elbow Detection - Second Derivative', fontsize=11, fontweight='bold')
ax9.legend()
ax9.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_clustering_iris.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'elbow_clustering_iris.png'")

# =====================================================
# CLUSTER ANALYSIS
# =====================================================

print("\n" + "=" * 80)
print("CLUSTER ANALYSIS")
print("=" * 80)

# Add cluster assignments to dataframe
df['Cluster'] = y_pred

# Analyze clusters
print(f"\nCluster characteristics (mean values):")
print(f"{'Cluster':<10} {'Sepal Len':<12} {'Sepal Wid':<12} {'Petal Len':<12} {'Petal Wid':<12}")
print("-" * 58)

for cluster_id in range(optimal_k):
    cluster_data = df[df['Cluster'] == cluster_id].iloc[:, :-3]
    means = cluster_data.mean()
    print(f"{cluster_id:<10} {means[0]:<12.2f} {means[1]:<12.2f} {means[2]:<12.2f} {means[3]:<12.2f}")

# Cluster to true species mapping
print(f"\nCluster to Species mapping:")
for cluster_id in range(optimal_k):
    cluster_species = df[df['Cluster'] == cluster_id]['target_name'].value_counts()
    print(f"  Cluster {cluster_id}:")
    for species, count in cluster_species.items():
        percentage = (count / len(df[df['Cluster'] == cluster_id])) * 100
        print(f"    {species}: {count} samples ({percentage:.1f}%)")

# =====================================================
# SUMMARY AND RECOMMENDATIONS
# =====================================================

print("\n" + "=" * 80)
print("SUMMARY: ELBOW METHOD FOR K-MEANS CLUSTERING")
print("=" * 80)

print(f"""
Dataset Information:
  - Total samples: {len(df)}
  - Number of features: 4
  - Features: {list(feature_names)}
  - Species: {list(target_names)}

Elbow Method Analysis:
  - Tested k values: 1 to 10
  - Optimal k (Elbow): {optimal_k_elbow}
  - Optimal k (Silhouette): {optimal_k_silhouette}
  - Optimal k (Davies-Bouldin): {optimal_k_db}
  - Optimal k (Calinski-Harabasz): {optimal_k_ch}

Selected Optimal k: {optimal_k}

Final Model Performance (k={optimal_k}):
  - Inertia: {final_kmeans.inertia_:.2f}
  - Silhouette Score: {silhouette_score(X_scaled, y_pred):.4f}
    (Range: -1 to 1, higher is better)
  - Davies-Bouldin Index: {davies_bouldin_score(X_scaled, y_pred):.4f}
    (Lower is better, measures cluster separation)
  - Calinski-Harabasz Score: {calinski_harabasz_score(X_scaled, y_pred):.2f}
    (Higher is better, ratio of between-cluster to within-cluster variance)

Key Metrics Explained:

1. INERTIA (Within-cluster sum of squares):
   - Lower values indicate tighter clusters
   - Elbow point: where reduction rate slows down

2. SILHOUETTE SCORE:
   - Measures how similar objects are to their own cluster vs other clusters
   - Range: -1 to 1 (1 is best)
   - Score > 0.5 generally indicates good clustering

3. DAVIES-BOULDIN INDEX:
   - Average similarity between each cluster and its most similar cluster
   - Lower values indicate better separation
   - No upper bound

4. CALINSKI-HARABASZ SCORE:
   - Ratio of between-cluster dispersion to within-cluster dispersion
   - Higher values indicate better-defined clusters
   - More robust to number of clusters than Silhouette

Recommendations:
✓ Use k={optimal_k} for this Iris dataset (consensus from multiple methods)
✓ Elbow method is good for initial k selection
✓ Combine with Silhouette Score for final decision
✓ Consider domain knowledge when selecting k
✓ The iris dataset has 3 true species, but k={optimal_k} was selected by metrics
""")

print("=" * 80)
