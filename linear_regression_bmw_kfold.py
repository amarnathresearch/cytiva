import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# LINEAR REGRESSION WITH K-FOLD CV - BMW SALES DATASET
# =====================================================

print("=" * 80)
print("LINEAR REGRESSION WITH K-FOLD CROSS-VALIDATION - BMW SALES DATA")
print("=" * 80)

# Load BMW dataset
# Download from: https://www.kaggle.com/datasets/payaldhokane/bmw-global-sales-and-market-data
try:
    df = pd.read_csv('/opt/cytiva/bmw_sales_data.csv')
except FileNotFoundError:
    print("\nDataset not found at /opt/cytiva/bmw_sales_data.csv")
    print("Please download from: https://www.kaggle.com/datasets/payaldhokane/bmw-global-sales-and-market-data")
    print("And place it in /opt/cytiva/ directory")
    exit()

print(f"\n{'Dataset Information':}")
print(f"  Shape: {df.shape}")
print(f"  Total samples: {len(df)}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nDataset Info:")
print(df.info())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nBasic statistics:")
print(df.describe())

# =====================================================
# DATA PREPROCESSING
# =====================================================

print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

df_processed = df.copy()

# Handle missing values
print("\nHandling missing values...")
for col in df_processed.columns:
    if df_processed[col].isnull().sum() > 0:
        if df_processed[col].dtype == 'object':
            df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
            print(f"  {col}: Filled with mode")
        else:
            df_processed[col].fillna(df_processed[col].median(), inplace=True)
            print(f"  {col}: Filled with median")

# Identify target variable (assume 'Units Sold' or similar column exists)
target_cols = [col for col in df_processed.columns if 'units' in col.lower() or 'sold' in col.lower() or 'sales' in col.lower()]

if not target_cols:
    print("\nWarning: Could not automatically identify target column.")
    print("Available columns:", df_processed.columns.tolist())
    print("Using the last numeric column as target...")
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    target_col = numeric_cols[-1]
else:
    target_col = target_cols[0]

print(f"\nTarget variable selected: '{target_col}'")

# =====================================================
# FEATURE ENGINEERING
# =====================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING")
print("=" * 80)

# Separate features and target
y = df_processed[target_col].values
X_cols = [col for col in df_processed.columns if col != target_col]
X = df_processed[X_cols].copy()

print(f"\nTarget variable: {target_col}")
print(f"Number of features: {len(X_cols)}")
print(f"Features: {X_cols}")

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: {len(le.classes_)} unique values")

print(f"\nX shape after encoding: {X.shape}")
print(f"y shape: {y.shape}")
print(f"\nTarget variable statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Median: {np.median(y):.2f}")
print(f"  Std Dev: {y.std():.2f}")
print(f"  Min: {y.min():.2f}")
print(f"  Max: {y.max():.2f}")

# =====================================================
# FEATURE SCALING
# =====================================================

print("\n" + "=" * 80)
print("FEATURE SCALING")
print("=" * 80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeatures scaled using StandardScaler")
print(f"X_scaled shape: {X_scaled.shape}")

# =====================================================
# K-FOLD CROSS-VALIDATION SETUP
# =====================================================

print("\n" + "=" * 80)
print("K-FOLD CROSS-VALIDATION SETUP")
print("=" * 80)

k_values = [3, 5, 10]
cv_results = {}

for k in k_values:
    print(f"\n{'=' * 50}")
    print(f"K-FOLD CROSS-VALIDATION (k={k})")
    print(f"{'=' * 50}")
    
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Initialize linear regression model
    lr_model = LinearRegression()
    
    # Define scoring metrics
    scoring = {
        'r2': 'r2',
        'neg_mse': 'neg_mean_squared_error',
        'neg_mae': 'neg_mean_absolute_error',
        'neg_mape': 'neg_mean_absolute_percentage_error'
    }
    
    # Perform cross-validation
    cv_scores = cross_validate(lr_model, X_scaled, y, cv=kf, scoring=scoring, return_train_score=True)
    
    # Extract scores
    train_r2_scores = cv_scores['train_r2']
    test_r2_scores = cv_scores['test_r2']
    train_mse_scores = -cv_scores['train_neg_mse']
    test_mse_scores = -cv_scores['test_neg_mse']
    train_mae_scores = -cv_scores['train_neg_mae']
    test_mae_scores = -cv_scores['test_neg_mae']
    
    # Store results
    cv_results[k] = {
        'kf': kf,
        'train_r2': train_r2_scores,
        'test_r2': test_r2_scores,
        'train_mse': train_mse_scores,
        'test_mse': test_mse_scores,
        'train_mae': train_mae_scores,
        'test_mae': test_mae_scores
    }
    
    # Print detailed results for each fold
    print(f"\nFold Results:")
    print(f"{'Fold':<8} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12} {'Train MAE':<12} {'Test MAE':<12}")
    print("-" * 80)
    
    for fold in range(k):
        print(f"{fold+1:<8} {train_r2_scores[fold]:<12.4f} {test_r2_scores[fold]:<12.4f} "
              f"{train_mse_scores[fold]:<12.2f} {test_mse_scores[fold]:<12.2f} "
              f"{train_mae_scores[fold]:<12.2f} {test_mae_scores[fold]:<12.2f}")
    
    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'Metric':<25} {'Mean':<12} {'Std Dev':<12} {'Min':<12} {'Max':<12}")
    print("-" * 61)
    
    print(f"{'Train R²':<25} {train_r2_scores.mean():<12.4f} {train_r2_scores.std():<12.4f} "
          f"{train_r2_scores.min():<12.4f} {train_r2_scores.max():<12.4f}")
    print(f"{'Test R²':<25} {test_r2_scores.mean():<12.4f} {test_r2_scores.std():<12.4f} "
          f"{test_r2_scores.min():<12.4f} {test_r2_scores.max():<12.4f}")
    print(f"{'Train MSE':<25} {train_mse_scores.mean():<12.2f} {train_mse_scores.std():<12.2f} "
          f"{train_mse_scores.min():<12.2f} {train_mse_scores.max():<12.2f}")
    print(f"{'Test MSE':<25} {test_mse_scores.mean():<12.2f} {test_mse_scores.std():<12.2f} "
          f"{test_mse_scores.min():<12.2f} {test_mse_scores.max():<12.2f}")
    print(f"{'Train MAE':<25} {train_mae_scores.mean():<12.2f} {train_mae_scores.std():<12.2f} "
          f"{train_mae_scores.min():<12.2f} {train_mae_scores.max():<12.2f}")
    print(f"{'Test MAE':<25} {test_mae_scores.mean():<12.2f} {test_mae_scores.std():<12.2f} "
          f"{test_mae_scores.min():<12.2f} {test_mae_scores.max():<12.2f}")

# =====================================================
# TRAIN FINAL MODEL ON ENTIRE DATASET
# =====================================================

print("\n" + "=" * 80)
print("FINAL MODEL TRAINING (Full Dataset)")
print("=" * 80)

final_model = LinearRegression()
final_model.fit(X_scaled, y)

y_pred = final_model.predict(X_scaled)

final_r2 = r2_score(y, y_pred)
final_mse = mean_squared_error(y, y_pred)
final_mae = mean_absolute_error(y, y_pred)
final_rmse = np.sqrt(final_mse)

print(f"\nFinal Model Performance (Full Dataset):")
print(f"  R² Score: {final_r2:.4f}")
print(f"  MSE: {final_mse:.2f}")
print(f"  RMSE: {final_rmse:.2f}")
print(f"  MAE: {final_mae:.2f}")

print(f"\nIntercept: {final_model.intercept_:.4f}")
print(f"\nFeature Coefficients (Top 20):")
coef_df = pd.DataFrame({
    'Feature': X_cols,
    'Coefficient': final_model.coef_
}).sort_values('Coefficient', key=abs, ascending=False)

print(coef_df.head(20).to_string(index=False))

# =====================================================
# VISUALIZATIONS
# =====================================================

print("\n" + "=" * 80)
print("GENERATING VISUALIZATIONS...")
print("=" * 80)

fig = plt.figure(figsize=(18, 14))

# 1. Predictions vs Actual
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(y, y_pred, alpha=0.6, s=50, color='blue', edgecolors='black', linewidth=0.5)
min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Units Sold', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted Units Sold', fontsize=10, fontweight='bold')
ax1.set_title('Predicted vs Actual Values', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Residuals Plot
ax2 = plt.subplot(3, 3, 2)
residuals = y - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, s=50, color='green', edgecolors='black', linewidth=0.5)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Values', fontsize=10, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax2.set_title('Residual Plot', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Residuals Distribution
ax3 = plt.subplot(3, 3, 3)
ax3.hist(residuals, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('Residuals', fontsize=10, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax3.set_title('Distribution of Residuals', fontsize=11, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 4. K-Fold CV: Test R² Comparison
ax4 = plt.subplot(3, 3, 4)
k_values_list = list(cv_results.keys())
test_r2_means = [cv_results[k]['test_r2'].mean() for k in k_values_list]
test_r2_stds = [cv_results[k]['test_r2'].std() for k in k_values_list]

ax4.bar(range(len(k_values_list)), test_r2_means, yerr=test_r2_stds, capsize=10,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1)
ax4.set_xlabel('K-Fold Configuration', fontsize=10, fontweight='bold')
ax4.set_ylabel('Mean Test R² Score', fontsize=10, fontweight='bold')
ax4.set_title('K-Fold CV: Test R² Comparison', fontsize=11, fontweight='bold')
ax4.set_xticks(range(len(k_values_list)))
ax4.set_xticklabels([f'k={k}' for k in k_values_list])
ax4.grid(axis='y', alpha=0.3)

# 5. K-Fold CV: Test MSE Comparison
ax5 = plt.subplot(3, 3, 5)
test_mse_means = [cv_results[k]['test_mse'].mean() for k in k_values_list]
test_mse_stds = [cv_results[k]['test_mse'].std() for k in k_values_list]

ax5.bar(range(len(k_values_list)), test_mse_means, yerr=test_mse_stds, capsize=10,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1)
ax5.set_xlabel('K-Fold Configuration', fontsize=10, fontweight='bold')
ax5.set_ylabel('Mean Test MSE', fontsize=10, fontweight='bold')
ax5.set_title('K-Fold CV: Test MSE Comparison', fontsize=11, fontweight='bold')
ax5.set_xticks(range(len(k_values_list)))
ax5.set_xticklabels([f'k={k}' for k in k_values_list])
ax5.grid(axis='y', alpha=0.3)

# 6. K-Fold CV: Test MAE Comparison
ax6 = plt.subplot(3, 3, 6)
test_mae_means = [cv_results[k]['test_mae'].mean() for k in k_values_list]
test_mae_stds = [cv_results[k]['test_mae'].std() for k in k_values_list]

ax6.bar(range(len(k_values_list)), test_mae_means, yerr=test_mae_stds, capsize=10,
        color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8, edgecolor='black', linewidth=1)
ax6.set_xlabel('K-Fold Configuration', fontsize=10, fontweight='bold')
ax6.set_ylabel('Mean Test MAE', fontsize=10, fontweight='bold')
ax6.set_title('K-Fold CV: Test MAE Comparison', fontsize=11, fontweight='bold')
ax6.set_xticks(range(len(k_values_list)))
ax6.set_xticklabels([f'k={k}' for k in k_values_list])
ax6.grid(axis='y', alpha=0.3)

# 7. Top Feature Coefficients
ax7 = plt.subplot(3, 3, 7)
top_coef = coef_df.head(10)
colors_coef = ['green' if x > 0 else 'red' for x in top_coef['Coefficient']]
ax7.barh(top_coef['Feature'], top_coef['Coefficient'], color=colors_coef, alpha=0.7, edgecolor='black')
ax7.set_xlabel('Coefficient Value', fontsize=10, fontweight='bold')
ax7.set_title('Top 10 Feature Coefficients', fontsize=11, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# 8. K=5 Fold-wise R² Scores
ax8 = plt.subplot(3, 3, 8)
k5_results = cv_results[5]
fold_indices = np.arange(1, 6)
ax8.plot(fold_indices, k5_results['train_r2'], marker='o', label='Train R²', linewidth=2, markersize=8)
ax8.plot(fold_indices, k5_results['test_r2'], marker='s', label='Test R²', linewidth=2, markersize=8)
ax8.set_xlabel('Fold Number', fontsize=10, fontweight='bold')
ax8.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax8.set_title('K-Fold (k=5): Train vs Test R² per Fold', fontsize=11, fontweight='bold')
ax8.set_xticks(fold_indices)
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. Actual vs Predicted Distribution
ax9 = plt.subplot(3, 3, 9)
ax9.hist(y, bins=30, alpha=0.6, label='Actual', color='blue', edgecolor='black')
ax9.hist(y_pred, bins=30, alpha=0.6, label='Predicted', color='orange', edgecolor='black')
ax9.set_xlabel('Units Sold', fontsize=10, fontweight='bold')
ax9.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax9.set_title('Distribution: Actual vs Predicted', fontsize=11, fontweight='bold')
ax9.legend()
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('linear_regression_bmw_kfold.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'linear_regression_bmw_kfold.png'")

# =====================================================
# PREDICTIONS ON NEW DATA
# =====================================================

print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS")
print("=" * 80)

# Show some random samples from the dataset
sample_indices = np.random.choice(len(X), min(5, len(X)), replace=False)
print(f"\nRandom sample predictions:")
print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<12}")
print("-" * 56)

for idx in sample_indices:
    actual = y[idx]
    predicted = y_pred[idx]
    error = actual - predicted
    error_pct = (error / actual) * 100 if actual != 0 else 0
    print(f"{idx:<8} {actual:<12.2f} {predicted:<12.2f} {error:<12.2f} {error_pct:<12.2f}")

# =====================================================
# SUMMARY AND RECOMMENDATIONS
# =====================================================

print("\n" + "=" * 80)
print("SUMMARY: LINEAR REGRESSION WITH K-FOLD CROSS-VALIDATION")
print("=" * 80)

best_k = max(cv_results.keys(), key=lambda k: cv_results[k]['test_r2'].mean())
best_results = cv_results[best_k]

print(f"""
Dataset Information:
  - Total samples: {len(df)}
  - Number of features: {len(X_cols)}
  - Target variable: {target_col}
  - Target range: [{y.min():.2f}, {y.max():.2f}]
  - Target mean: {y.mean():.2f}
  - Target std dev: {y.std():.2f}

Data Preprocessing:
  - Categorical features encoded: {len(categorical_cols)}
  - Numerical features: {len(numerical_cols)}
  - Features scaled using StandardScaler

K-Fold Cross-Validation Results:
  - Best k configuration: k={best_k}
  - Mean Test R²: {best_results['test_r2'].mean():.4f} (±{best_results['test_r2'].std():.4f})
  - Mean Test MSE: {best_results['test_mse'].mean():.2f} (±{best_results['test_mse'].std():.2f})
  - Mean Test MAE: {best_results['test_mae'].mean():.2f} (±{best_results['test_mae'].std():.2f})

Final Model Performance (Full Dataset):
  - R² Score: {final_r2:.4f}
  - MSE: {final_mse:.2f}
  - RMSE: {final_rmse:.2f}
  - MAE: {final_mae:.2f}

Top 5 Important Features:
""")

for idx, (_, row) in enumerate(coef_df.head(5).iterrows(), 1):
    print(f"  {idx}. {row['Feature']}: {row['Coefficient']:.4f}")

print(f"""
Model Insights:
✓ Linear regression model trained with k-fold cross-validation
✓ Multiple k configurations tested (k=3, 5, 10) for robust evaluation
✓ Features standardized to have zero mean and unit variance
✓ Cross-validation scores indicate model generalization performance
✓ Residuals analysis shows model prediction quality

Recommendations:
- Use k={best_k} for future cross-validation (best performance)
- Monitor test scores across folds for consistency
- Consider feature engineering for improved predictions
- Test other regression models for comparison
- Address any high residual outliers for better fit
""")

print("=" * 80)
