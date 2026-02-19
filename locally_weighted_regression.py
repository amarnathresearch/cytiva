import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import load_iris
import seaborn as sns

# =====================================================
# LOCALLY WEIGHTED LINEAR REGRESSION (LWLR)
# =====================================================

class LocallyWeightedRegression:
    """
    Locally Weighted Linear Regression (LWLR)
    A non-parametric regression algorithm that weights nearby points more heavily
    """
    
    def __init__(self, tau=1.0):
        """
        Parameters:
        tau: bandwidth parameter (controls the width of the weight function)
             larger tau = more points influence prediction
        """
        self.tau = tau
        self.X_train = None
        self.y_train = None
        self.weights = None
        
    def _gaussian_kernel(self, query_point, training_points):
        """
        Compute Gaussian kernel weights
        w_i = exp(-(||x_i - x||^2) / (2 * tau^2))
        """
        distances = np.linalg.norm(training_points - query_point, axis=1)
        weights = np.exp(-(distances ** 2) / (2 * (self.tau ** 2)))
        return weights
    
    def fit(self, X, y):
        """
        Store training data (LWLR is lazy - no explicit model fitting)
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y).reshape(-1, 1)
        print(f"Training data stored: {self.X_train.shape[0]} samples, {self.X_train.shape[1]} features")
    
    def predict(self, X):
        """
        Make predictions using locally weighted regression
        """
        X = np.array(X)
        predictions = []
        
        for query_point in X:
            # Compute weights for all training points
            weights = self._gaussian_kernel(query_point, self.X_train)
            weights = weights.reshape(-1, 1)
            
            # Create weight matrix W (diagonal matrix)
            W = np.diag(weights.flatten())
            
            # Compute weighted least squares solution
            # (X^T * W * X)^-1 * X^T * W * y
            try:
                XT = self.X_train.T
                theta = np.linalg.inv(XT @ W @ self.X_train) @ XT @ W @ self.y_train
                prediction = query_point @ theta
                predictions.append(prediction[0])
            except np.linalg.LinAlgError:
                # If matrix is singular, use least squares without weights
                prediction = np.linalg.lstsq(self.X_train, self.y_train, rcond=None)[0]
                predictions.append((query_point @ prediction)[0])
        
        return np.array(predictions)
    
    def score(self, X, y):
        """Compute R² score"""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# =====================================================
# SYNTHETIC DATASET FOR REGRESSION
# =====================================================

print("=" * 60)
print("LOCALLY WEIGHTED LINEAR REGRESSION")
print("=" * 60)

# Create synthetic dataset
np.random.seed(42)
n_samples = 200

# Generate features
X = np.random.uniform(-10, 10, (n_samples, 2))

# Generate target (non-linear relationship)
y = 2*X[:, 0] + 3*X[:, 1] + 0.5*X[:, 0]*X[:, 1] + np.random.normal(0, 5, n_samples)

print(f"\nDataset created:")
print(f"  Total samples: {n_samples}")
print(f"  Features: 2")
print(f"  Target variable: Continuous")
print(f"\nX shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nFeature 1 range: [{X[:, 0].min():.2f}, {X[:, 0].max():.2f}]")
print(f"Feature 2 range: [{X[:, 1].min():.2f}, {X[:, 1].max():.2f}]")
print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training set: {X_train.shape[0]} samples")
print(f"  Testing set: {X_test.shape[0]} samples")

# =====================================================
# TRAIN LOCALLY WEIGHTED REGRESSION
# =====================================================

print("\n" + "=" * 60)
print("TRAINING LOCALLY WEIGHTED REGRESSION")
print("=" * 60)

# Train with different tau values
tau_values = [0.5, 1.0, 2.0, 5.0]
models = {}

for tau in tau_values:
    print(f"\nTraining LWLR with tau = {tau}...")
    lwlr = LocallyWeightedRegression(tau=tau)
    lwlr.fit(X_train, y_train)
    models[tau] = lwlr
    print(f"  Training score (R²): {lwlr.score(X_train, y_train):.4f}")

# =====================================================
# EVALUATION
# =====================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

best_tau = None
best_r2 = -np.inf

evaluation_results = []

for tau in tau_values:
    lwlr = models[tau]
    
    # Predictions
    y_pred_train = lwlr.predict(X_train)
    y_pred_test = lwlr.predict(X_test)
    
    # Metrics
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    print(f"\nTau = {tau}:")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")
    print(f"  Training MSE: {train_mse:.4f}")
    print(f"  Testing MSE: {test_mse:.4f}")
    print(f"  Training MAE: {train_mae:.4f}")
    print(f"  Testing MAE: {test_mae:.4f}")
    
    evaluation_results.append({
        'Tau': tau,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    })
    
    if test_r2 > best_r2:
        best_r2 = test_r2
        best_tau = tau

print(f"\n✓ Best tau: {best_tau} (Test R² = {best_r2:.4f})")

# =====================================================
# VISUALIZATIONS
# =====================================================

fig = plt.figure(figsize=(16, 12))

# 1. Predicted vs Actual (Test Set - Best Model)
ax1 = plt.subplot(2, 3, 1)
best_model = models[best_tau]
y_pred_best = best_model.predict(X_test)
ax1.scatter(y_test, y_pred_best, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
min_val = min(y_test.min(), y_pred_best.min())
max_val = max(y_test.max(), y_pred_best.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Values', fontsize=10)
ax1.set_ylabel('Predicted Values', fontsize=10)
ax1.set_title(f'Predicted vs Actual (Best Model: tau={best_tau})', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals (Test Set)
ax2 = plt.subplot(2, 3, 2)
residuals = y_test - y_pred_best
ax2.scatter(y_pred_best, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=10)
ax2.set_ylabel('Residuals', fontsize=10)
ax2.set_title('Residual Plot (Test Set)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Performance Metrics Comparison
ax3 = plt.subplot(2, 3, 3)
df_results = pd.DataFrame(evaluation_results)
x_pos = np.arange(len(tau_values))
width = 0.35
bars1 = ax3.bar(x_pos - width/2, df_results['Train R²'], width, label='Train R²', alpha=0.8)
bars2 = ax3.bar(x_pos + width/2, df_results['Test R²'], width, label='Test R²', alpha=0.8)
ax3.set_xlabel('Tau Value', fontsize=10)
ax3.set_ylabel('R² Score', fontsize=10)
ax3.set_title('R² Score Comparison Across Tau Values', fontsize=12, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([str(t) for t in tau_values])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0, 1])

# 4. MSE Comparison
ax4 = plt.subplot(2, 3, 4)
bars1 = ax4.bar(x_pos - width/2, df_results['Train MSE'], width, label='Train MSE', alpha=0.8, color='orange')
bars2 = ax4.bar(x_pos + width/2, df_results['Test MSE'], width, label='Test MSE', alpha=0.8, color='red')
ax4.set_xlabel('Tau Value', fontsize=10)
ax4.set_ylabel('MSE', fontsize=10)
ax4.set_title('MSE Comparison Across Tau Values', fontsize=12, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels([str(t) for t in tau_values])
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Residuals Distribution
ax5 = plt.subplot(2, 3, 5)
ax5.hist(residuals, bins=20, color='purple', alpha=0.7, edgecolor='black')
ax5.set_xlabel('Residuals', fontsize=10)
ax5.set_ylabel('Frequency', fontsize=10)
ax5.set_title('Distribution of Residuals', fontsize=12, fontweight='bold')
ax5.grid(axis='y', alpha=0.3)
ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)

# 6. MAE Comparison
ax6 = plt.subplot(2, 3, 6)
bars1 = ax6.bar(x_pos - width/2, df_results['Train MAE'], width, label='Train MAE', alpha=0.8, color='teal')
bars2 = ax6.bar(x_pos + width/2, df_results['Test MAE'], width, label='Test MAE', alpha=0.8, color='coral')
ax6.set_xlabel('Tau Value', fontsize=10)
ax6.set_ylabel('MAE', fontsize=10)
ax6.set_title('MAE Comparison Across Tau Values', fontsize=12, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([str(t) for t in tau_values])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('locally_weighted_regression.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'locally_weighted_regression.png'")

# =====================================================
# PREDICTIONS ON NEW DATA
# =====================================================

print("\n" + "=" * 60)
print("PREDICTIONS ON NEW DATA")
print("=" * 60)

# Example predictions
test_samples = [
    [5.0, 3.0],
    [-2.5, 4.5],
    [8.0, -6.0],
    [0.0, 0.0],
    [-5.0, -5.0]
]

print(f"\nPredictions using best model (tau={best_tau}):\n")
for sample in test_samples:
    sample_array = np.array(sample).reshape(1, -1)
    prediction = best_model.predict(sample_array)[0]
    print(f"  Input: Feature1={sample[0]:6.2f}, Feature2={sample[1]:6.2f} → Prediction: {prediction:8.2f}")

# =====================================================
# MODEL SUMMARY
# =====================================================

print("\n" + "=" * 60)
print("LOCALLY WEIGHTED REGRESSION SUMMARY")
print("=" * 60)
print(f"""
Algorithm: Locally Weighted Linear Regression (LWLR)
Type: Non-parametric, lazy learning algorithm

Key Characteristics:
- Makes predictions by weighting nearby training examples
- Uses Gaussian kernel for weighting: w_i = exp(-(d_i²)/(2*tau²))
- No explicit training phase (lazy learning)
- Good for non-linear relationships

Hyperparameter:
- Tau (τ): Controls bandwidth of the weight function
  * Smaller tau: More focused on nearby points (low bias, high variance)
  * Larger tau: More points influence prediction (high bias, low variance)

Best Model Configuration:
- Tau: {best_tau}
- Test R² Score: {best_r2:.4f}
- Training Set Size: {X_train.shape[0]} samples
- Feature Dimensions: {X_train.shape[1]}

Advantages:
✓ Can model non-linear relationships
✓ Simple and intuitive
✓ Good for multi-dimensional data
✓ No explicit model assumptions

Disadvantages:
✗ Computationally expensive for prediction (must compute weights for all training points)
✗ Sensitive to choice of tau (bandwidth parameter)
✗ Requires storing all training data
✗ Performance degrades with high-dimensional data (curse of dimensionality)
""")

print("=" * 60)
