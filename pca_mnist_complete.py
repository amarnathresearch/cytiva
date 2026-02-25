"""
PCA for MNIST Digit Classification using Decision Tree
This script demonstrates Principal Component Analysis (PCA) for dimensionality reduction
and classification of MNIST handwritten digits using a Decision Tree classifier.
All visualizations are displayed inline.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_explore_data():
    """Load MNIST-like dataset and explore its properties"""
    print("\n" + "=" * 70)
    print("STEP 1: LOAD AND EXPLORE MNIST DATASET")
    print("=" * 70)
    
    digits = load_digits()
    X = digits.data  # 1797 samples, 64 features (8x8 pixels)
    y = digits.target  # labels 0-9
    
    print(f"\nDataset shape: {X.shape}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features (pixels): {X.shape[1]}")
    print(f"Number of classes (digits): {len(np.unique(y))}")
    print(f"Classes: {np.unique(y)}")
    print(f"Feature value range: [{X.min():.2f}, {X.max():.2f}]")
    
    return X, y


def standardize_data(X):
    """Standardize the features before PCA"""
    print("\n" + "=" * 70)
    print("STEP 2: STANDARDIZE THE DATA")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"\nData standardization completed")
    print(f"Mean of scaled data: {X_scaled.mean():.6f}")
    print(f"Std of scaled data: {X_scaled.std():.6f}")
    
    return X_scaled


def apply_pca_and_analyze(X_scaled):
    """Apply PCA with all components and analyze variance"""
    print("\n" + "=" * 70)
    print("STEP 3: APPLY PCA AND ANALYZE VARIANCE")
    print("=" * 70)
    
    # Apply PCA with all components
    pca_full = PCA(n_components=64)
    X_pca_full = pca_full.fit_transform(X_scaled)
    
    # Get explained variance ratio
    explained_variance_ratio = pca_full.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\nFirst 10 Principal Components:")
    for i in range(10):
        print(f"PC{i+1:2d}: {explained_variance_ratio[i]:.6f} ({explained_variance_ratio[i]*100:6.2f}%)")
    
    print("\n" + "=" * 70)
    print("CUMULATIVE EXPLAINED VARIANCE")
    print("=" * 70)
    print(f"First 5 components  : {cumulative_variance[4]:.6f} ({cumulative_variance[4]*100:6.2f}%)")
    print(f"First 10 components : {cumulative_variance[9]:.6f} ({cumulative_variance[9]*100:6.2f}%)")
    print(f"First 15 components : {cumulative_variance[14]:.6f} ({cumulative_variance[14]*100:6.2f}%)")
    print(f"First 20 components : {cumulative_variance[19]:.6f} ({cumulative_variance[19]*100:6.2f}%)")
    print(f"First 30 components : {cumulative_variance[29]:.6f} ({cumulative_variance[29]*100:6.2f}%)")
    print(f"First 50 components : {cumulative_variance[49]:.6f} ({cumulative_variance[49]*100:6.2f}%)")
    
    return pca_full, explained_variance_ratio, cumulative_variance


def visualize_variance(explained_variance_ratio, cumulative_variance):
    """Visualize scree plot and cumulative variance"""
    print("\n" + "=" * 70)
    print("STEP 4: VISUALIZE SCREE PLOT AND CUMULATIVE VARIANCE")
    print("=" * 70)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Scree plot
    ax1.plot(range(1, 65), explained_variance_ratio, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
    ax1.set_title('Scree Plot - MNIST Dataset', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 65)
    
    # Cumulative explained variance
    ax2.plot(range(1, 65), cumulative_variance*100, 'ro-', linewidth=2, markersize=6, label='Cumulative Variance')
    ax2.axhline(y=95, color='g', linestyle='--', linewidth=2, label='95% variance threshold')
    ax2.axvline(x=30, color='orange', linestyle=':', linewidth=2, label='30 components (optimal)')
    ax2.set_xlabel('Number of Principal Components', fontsize=12)
    ax2.set_ylabel('Cumulative Explained Variance (%)', fontsize=12)
    ax2.set_title('Cumulative Explained Variance - MNIST Dataset', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 65)
    ax2.set_ylim([0, 105])
    
    plt.tight_layout()
    plt.show()
    print("✓ Scree plot and cumulative variance visualization displayed")


def classify_with_multiple_components(X_scaled, y):
    """Train Decision Tree with different PCA components"""
    print("\n" + "=" * 70)
    print("STEP 5: CLASSIFICATION WITH MULTIPLE PCA COMPONENTS")
    print("=" * 70)
    
    components_to_test = [5, 10, 15, 20, 30, 50, 64]
    results = {}
    
    print("\nTraining Decision Tree with different PCA components...")
    for n_comp in components_to_test:
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
        
        print(f"Components: {n_comp:2d} | Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    return results


def plot_accuracy_results(results):
    """Plot classification accuracy vs number of components"""
    print("\n" + "=" * 70)
    print("STEP 6: PLOT ACCURACY RESULTS")
    print("=" * 70)
    
    plt.figure(figsize=(12, 6))
    plt.plot(list(results.keys()), list(results.values()), 'bo-', linewidth=3, markersize=10, label='Accuracy')
    plt.axvline(x=30, color='red', linestyle='--', linewidth=2, label='Optimal (30 components)')
    plt.xlabel('Number of Principal Components', fontsize=12, fontweight='bold')
    plt.ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    plt.title('Decision Tree Classification Accuracy vs PCA Components', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.xticks(list(results.keys()))
    plt.ylim([0.9, 1.0])
    
    # Add value labels on points
    for comp, acc in results.items():
        plt.annotate(f'{acc:.3f}', xy=(comp, acc), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Accuracy plot displayed")


def detailed_classification(X_scaled, y, n_optimal=30):
    """Perform detailed classification with optimal components"""
    print("\n" + "=" * 70)
    print(f"STEP 7: DETAILED CLASSIFICATION WITH {n_optimal} OPTIMAL COMPONENTS")
    print("=" * 70)
    
    # Apply PCA with optimal components
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
    print(f"\nAccuracy Score: {accuracy_optimal:.4f} ({accuracy_optimal*100:.2f}%)")
    print(f"Number of test samples: {len(y_test)}")
    print(f"Correct predictions: {np.sum(y_pred_optimal == y_test)}")
    print(f"Incorrect predictions: {np.sum(y_pred_optimal != y_test)}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_optimal)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_optimal, digits=4))
    
    return pca_optimal, cm, y_test, y_pred_optimal, accuracy_optimal


def plot_confusion_matrix(cm, accuracy_optimal, n_optimal=30):
    """Visualize confusion matrix as heatmap"""
    print("\n" + "=" * 70)
    print("STEP 8: DISPLAY CONFUSION MATRIX HEATMAP")
    print("=" * 70)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, 
                xticklabels=range(10), yticklabels=range(10),
                cbar_kws={'label': 'Number of Samples'},
                annot_kws={'size': 11, 'weight': 'bold'})
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title(f'Confusion Matrix - MNIST with {n_optimal} PCA Components\nAccuracy: {accuracy_optimal:.4f}', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    print("✓ Confusion matrix heatmap displayed")


def visualize_components(pca_full, explained_variance_ratio):
    """Visualize principal components as images"""
    print("\n" + "=" * 70)
    print("STEP 9: VISUALIZE PRINCIPAL COMPONENTS AS IMAGES")
    print("=" * 70)
    print("Each principal component is reshaped to an 8x8 image")
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.ravel()
    
    for i in range(10):
        # Reshape component to 8x8 image
        component_image = pca_full.components_[i].reshape(8, 8)
        axes[i].imshow(component_image, cmap='gray')
        axes[i].set_title(f'PC{i+1}\nVariance: {explained_variance_ratio[i]*100:.2f}%', 
                         fontsize=11, fontweight='bold')
        axes[i].axis('off')
    
    plt.suptitle('First 10 Principal Components Visualization', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    print("✓ Principal components visualization displayed")


def compare_original_vs_reconstructed(X, X_scaled, y, pca_optimal, n_optimal=30):
    """Compare original vs reconstructed digits"""
    print("\n" + "=" * 70)
    print("STEP 10: COMPARE ORIGINAL VS RECONSTRUCTED DIGITS")
    print("=" * 70)
    print(f"Reconstruction using {n_optimal} principal components")
    
    fig, axes = plt.subplots(3, 10, figsize=(18, 8))
    
    # Visualize first 10 samples
    for idx in range(10):
        # Original image
        original_image = X[idx].reshape(8, 8)
        axes[0, idx].imshow(original_image, cmap='gray')
        axes[0, idx].set_title(f'{y[idx]}', fontsize=11, fontweight='bold')
        axes[0, idx].axis('off')
        if idx == 0:
            axes[0, idx].set_ylabel('Original', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
        
        # PCA transformed and reconstructed
        pca_reconstructed = pca_optimal.inverse_transform(pca_optimal.transform(X_scaled[idx:idx+1]))
        reconstructed_image = pca_reconstructed[0].reshape(8, 8)
        axes[1, idx].imshow(reconstructed_image, cmap='gray')
        axes[1, idx].axis('off')
        if idx == 0:
            axes[1, idx].set_ylabel('Reconstructed', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
        
        # Difference visualization
        difference = np.abs(original_image - reconstructed_image)
        axes[2, idx].imshow(difference, cmap='hot')
        axes[2, idx].axis('off')
        if idx == 0:
            axes[2, idx].set_ylabel('Difference', fontsize=12, fontweight='bold', rotation=0, ha='right', va='center')
    
    plt.suptitle(f'Digit Reconstruction with {n_optimal} PCA Components\n(Original → Reconstructed → | Difference |)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()
    print("✓ Reconstruction comparison visualization displayed")


def print_summary(X, n_optimal, cumulative_variance, accuracy_optimal):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Original dataset size: {X.shape[1]} features")
    print(f"Reduced to: {n_optimal} principal components")
    print(f"Dimensionality reduction: {(1 - n_optimal/X.shape[1])*100:.2f}%")
    print(f"Variance retained: {cumulative_variance[n_optimal-1]*100:.2f}%")
    print(f"Classification Accuracy (Decision Tree): {accuracy_optimal:.4f}")
    print("\n✓ All steps completed successfully!")
    print("=" * 70)


def main():
    """Main function to run the entire PCA MNIST classification pipeline"""
    print("\n" + "=" * 70)
    print("PCA FOR MNIST DIGIT CLASSIFICATION USING DECISION TREE")
    print("=" * 70)
    
    # Load and explore data
    X, y = load_and_explore_data()
    
    # Standardize data
    X_scaled = standardize_data(X)
    
    # Apply PCA and analyze variance
    pca_full, explained_variance_ratio, cumulative_variance = apply_pca_and_analyze(X_scaled)
    
    # Visualize variance
    visualize_variance(explained_variance_ratio, cumulative_variance)
    
    # Classify with multiple components
    results = classify_with_multiple_components(X_scaled, y)
    
    # Plot accuracy results
    plot_accuracy_results(results)
    
    # Detailed classification with optimal components
    n_optimal = 30
    pca_optimal, cm, y_test, y_pred_optimal, accuracy_optimal = detailed_classification(X_scaled, y, n_optimal)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, accuracy_optimal, n_optimal)
    
    # Visualize principal components
    visualize_components(pca_full, explained_variance_ratio)
    
    # Compare original vs reconstructed
    compare_original_vs_reconstructed(X, X_scaled, y, pca_optimal, n_optimal)
    
    # Print summary
    print_summary(X, n_optimal, cumulative_variance, accuracy_optimal)


if __name__ == "__main__":
    main()
