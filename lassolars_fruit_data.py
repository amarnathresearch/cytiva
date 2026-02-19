import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoLars, Lasso, Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# LASSOLARS REGRESSION - FRUIT DATASET
# =====================================================

print("=" * 70)
print("LASSOLARS REGRESSION - FRUIT DATASET ANALYSIS")
print("=" * 70)

# Load fruit data
df = pd.read_csv('/opt/cytiva/fruit_data_augmented.csv')

print(f"\n{'Dataset Information':}")
print(f"  Total samples: {len(df)}")
print(f"  Shape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nDataset Info:")
print(df.info())
print(f"\nFruit distribution:")
print(df['target'].value_counts())
print(f"\nColor distribution:")
print(df['color'].value_counts())
print(f"\nDiameter statistics:")
print(df['diameter'].describe())

# Encode categorical features
color_encoder = LabelEncoder()
fruit_encoder = LabelEncoder()

df['color_encoded'] = color_encoder.fit_transform(df['color'])
df['fruit_encoded'] = fruit_encoder.fit_transform(df['target'])

print(f"\nColor encoding:")
for i, color in enumerate(color_encoder.classes_):
    print(f"  {color}: {i}")

print(f"\nFruit encoding:")
for i, fruit in enumerate(fruit_encoder.classes_):
    print(f"  {fruit}: {i}")

# =====================================================
# TASK 1: PREDICT DIAMETER FROM COLOR (Single Feature)
# =====================================================

print("\n" + "=" * 70)
print("TASK 1: PREDICT DIAMETER FROM COLOR")
print("=" * 70)

X_task1 = df[['color_encoded']].values
y_task1 = df['diameter'].values

# Split data
X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_task1, y_task1, test_size=0.2, random_state=42
)

# Scale features
scaler_t1 = StandardScaler()
X_train_t1_scaled = scaler_t1.fit_transform(X_train_t1)
X_test_t1_scaled = scaler_t1.transform(X_test_t1)

print(f"\nData split:")
print(f"  Training set: {X_train_t1.shape[0]} samples")
print(f"  Testing set: {X_test_t1.shape[0]} samples")
print(f"  Features: 1 (color_encoded)")

# Train LassoLars with different alphas
alphas = [0.001, 0.01, 0.1, 0.5, 1.0]
models_t1 = {}

print(f"\nTraining LassoLars models with different alphas...")
for alpha in alphas:
    model = LassoLars(alpha=alpha, random_state=42)
    model.fit(X_train_t1_scaled, y_train_t1)
    models_t1[alpha] = model

# Evaluate Task 1
print(f"\n{'Alpha':<10} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12} {'Coef':<10}")
print("-" * 68)

best_alpha_t1 = None
best_r2_t1 = -np.inf
results_t1 = []

for alpha in alphas:
    model = models_t1[alpha]
    y_pred_train = model.predict(X_train_t1_scaled)
    y_pred_test = model.predict(X_test_t1_scaled)
    
    train_r2 = r2_score(y_train_t1, y_pred_train)
    test_r2 = r2_score(y_test_t1, y_pred_test)
    train_mse = mean_squared_error(y_train_t1, y_pred_train)
    test_mse = mean_squared_error(y_test_t1, y_pred_test)
    
    coef = model.coef_[0] if len(model.coef_) > 0 else 0
    
    print(f"{alpha:<10} {train_r2:<12.4f} {test_r2:<12.4f} {train_mse:<12.4f} {test_mse:<12.4f} {coef:<10.4f}")
    
    results_t1.append({
        'Alpha': alpha,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Coefficient': coef
    })
    
    if test_r2 > best_r2_t1:
        best_r2_t1 = test_r2
        best_alpha_t1 = alpha

print(f"\n✓ Best alpha for Task 1: {best_alpha_t1} (Test R² = {best_r2_t1:.4f})")

# =====================================================
# TASK 2: PREDICT FRUIT TYPE FROM COLOR & DIAMETER
# =====================================================

print("\n" + "=" * 70)
print("TASK 2: PREDICT FRUIT TYPE FROM COLOR & DIAMETER")
print("=" * 70)

X_task2 = df[['color_encoded', 'diameter']].values
y_task2 = df['fruit_encoded'].values

# Split data
X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    X_task2, y_task2, test_size=0.2, random_state=42
)

# Scale features
scaler_t2 = StandardScaler()
X_train_t2_scaled = scaler_t2.fit_transform(X_train_t2)
X_test_t2_scaled = scaler_t2.transform(X_test_t2)

print(f"\nData split:")
print(f"  Training set: {X_train_t2.shape[0]} samples")
print(f"  Testing set: {X_test_t2.shape[0]} samples")
print(f"  Features: 2 (color_encoded, diameter)")

# Train LassoLars with different alphas
models_t2 = {}

print(f"\nTraining LassoLars models with different alphas...")
for alpha in alphas:
    model = LassoLars(alpha=alpha, random_state=42)
    model.fit(X_train_t2_scaled, y_train_t2)
    models_t2[alpha] = model

# Evaluate Task 2
print(f"\n{'Alpha':<10} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12}")
print("-" * 62)

best_alpha_t2 = None
best_r2_t2 = -np.inf
results_t2 = []

for alpha in alphas:
    model = models_t2[alpha]
    y_pred_train = model.predict(X_train_t2_scaled)
    y_pred_test = model.predict(X_test_t2_scaled)
    
    train_r2 = r2_score(y_train_t2, y_pred_train)
    test_r2 = r2_score(y_test_t2, y_pred_test)
    train_mse = mean_squared_error(y_train_t2, y_pred_train)
    test_mse = mean_squared_error(y_test_t2, y_pred_test)
    
    print(f"{alpha:<10} {train_r2:<12.4f} {test_r2:<12.4f} {train_mse:<12.4f} {test_mse:<12.4f}")
    
    results_t2.append({
        'Alpha': alpha,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })
    
    if test_r2 > best_r2_t2:
        best_r2_t2 = test_r2
        best_alpha_t2 = alpha

print(f"\n✓ Best alpha for Task 2: {best_alpha_t2} (Test R² = {best_r2_t2:.4f})")

# =====================================================
# COMPARE WITH OTHER MODELS
# =====================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON (Task 2 - Best Alpha Models)")
print("=" * 70)

# Train comparison models
models_comparison = {
    'LassoLars': LassoLars(alpha=best_alpha_t2, random_state=42),
    'Lasso': Lasso(alpha=best_alpha_t2, random_state=42),
    'Ridge': Ridge(alpha=best_alpha_t2, random_state=42),
    'Linear Regression': LinearRegression()
}

print(f"\nTraining comparison models...")
comparison_results = []

for model_name, model in models_comparison.items():
    model.fit(X_train_t2_scaled, y_train_t2)
    
    y_pred_train = model.predict(X_train_t2_scaled)
    y_pred_test = model.predict(X_test_t2_scaled)
    
    train_r2 = r2_score(y_train_t2, y_pred_train)
    test_r2 = r2_score(y_test_t2, y_pred_test)
    train_mse = mean_squared_error(y_train_t2, y_pred_train)
    test_mse = mean_squared_error(y_test_t2, y_pred_test)
    train_mae = mean_absolute_error(y_train_t2, y_pred_train)
    test_mae = mean_absolute_error(y_test_t2, y_pred_test)
    
    comparison_results.append({
        'Model': model_name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse,
        'Train MAE': train_mae,
        'Test MAE': test_mae
    })
    
    print(f"\n{model_name}:")
    print(f"  Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"  Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"  Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")

# =====================================================
# FEATURE COEFFICIENTS (Task 2)
# =====================================================

print("\n" + "=" * 70)
print("FEATURE COEFFICIENTS ANALYSIS (Task 2 - Best Model)")
print("=" * 70)

best_model_t2 = models_t2[best_alpha_t2]
feature_names = ['Color (Encoded)', 'Diameter']

print(f"\nLassoLars (Alpha={best_alpha_t2}):")
print(f"  Intercept: {best_model_t2.intercept_:.4f}")
for fname, coef in zip(feature_names, best_model_t2.coef_):
    print(f"  {fname}: {coef:.4f}")

# Lasso comparison
lasso_model = models_comparison['Lasso']
print(f"\nLasso (Alpha={best_alpha_t2}):")
print(f"  Intercept: {lasso_model.intercept_:.4f}")
for fname, coef in zip(feature_names, lasso_model.coef_):
    print(f"  {fname}: {coef:.4f}")

# Ridge comparison
ridge_model = models_comparison['Ridge']
print(f"\nRidge (Alpha={best_alpha_t2}):")
print(f"  Intercept: {ridge_model.intercept_:.4f}")
for fname, coef in zip(feature_names, ridge_model.coef_):
    print(f"  {fname}: {coef:.4f}")

# =====================================================
# VISUALIZATIONS
# =====================================================

print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS...")
print("=" * 70)

fig = plt.figure(figsize=(18, 14))

# Task 1 visualizations
best_model_t1 = models_t1[best_alpha_t1]
y_pred_test_t1 = best_model_t1.predict(X_test_t1_scaled)

# 1. Task 1: Predicted vs Actual
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(y_test_t1, y_pred_test_t1, alpha=0.7, s=100, color='blue', edgecolors='black', linewidth=1)
min_val = min(y_test_t1.min(), y_pred_test_t1.min())
max_val = max(y_test_t1.max(), y_pred_test_t1.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Diameter', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted Diameter', fontsize=10, fontweight='bold')
ax1.set_title(f'Task 1: Diameter Prediction (α={best_alpha_t1})', fontsize=11, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Task 1: Residuals
ax2 = plt.subplot(3, 3, 2)
residuals_t1 = y_test_t1 - y_pred_test_t1
ax2.scatter(y_pred_test_t1, residuals_t1, alpha=0.7, s=100, color='green', edgecolors='black', linewidth=1)
ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax2.set_xlabel('Predicted Diameter', fontsize=10, fontweight='bold')
ax2.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax2.set_title('Task 1: Residual Plot', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 3. Task 1: R² vs Alpha
ax3 = plt.subplot(3, 3, 3)
df_t1 = pd.DataFrame(results_t1)
ax3.plot(df_t1['Alpha'], df_t1['Train R²'], marker='o', label='Train R²', linewidth=2, markersize=8)
ax3.plot(df_t1['Alpha'], df_t1['Test R²'], marker='s', label='Test R²', linewidth=2, markersize=8)
ax3.set_xlabel('Alpha', fontsize=10, fontweight='bold')
ax3.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax3.set_title('Task 1: R² vs Alpha', fontsize=11, fontweight='bold')
ax3.set_xscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Task 2 visualizations
best_model_t2_pred = models_t2[best_alpha_t2]
y_pred_test_t2 = best_model_t2_pred.predict(X_test_t2_scaled)

# 4. Task 2: Predicted vs Actual
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test_t2, y_pred_test_t2, alpha=0.7, s=100, color='purple', edgecolors='black', linewidth=1)
min_val2 = min(y_test_t2.min(), y_pred_test_t2.min())
max_val2 = max(y_test_t2.max(), y_pred_test_t2.max())
ax4.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Fruit Type (Encoded)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Predicted Fruit Type', fontsize=10, fontweight='bold')
ax4.set_title(f'Task 2: Fruit Prediction (α={best_alpha_t2})', fontsize=11, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Task 2: Residuals
ax5 = plt.subplot(3, 3, 5)
residuals_t2 = y_test_t2 - y_pred_test_t2
ax5.scatter(y_pred_test_t2, residuals_t2, alpha=0.7, s=100, color='orange', edgecolors='black', linewidth=1)
ax5.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax5.set_xlabel('Predicted Fruit Type', fontsize=10, fontweight='bold')
ax5.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax5.set_title('Task 2: Residual Plot', fontsize=11, fontweight='bold')
ax5.grid(True, alpha=0.3)

# 6. Task 2: R² vs Alpha
ax6 = plt.subplot(3, 3, 6)
df_t2 = pd.DataFrame(results_t2)
ax6.plot(df_t2['Alpha'], df_t2['Train R²'], marker='o', label='Train R²', linewidth=2, markersize=8)
ax6.plot(df_t2['Alpha'], df_t2['Test R²'], marker='s', label='Test R²', linewidth=2, markersize=8)
ax6.set_xlabel('Alpha', fontsize=10, fontweight='bold')
ax6.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax6.set_title('Task 2: R² vs Alpha', fontsize=11, fontweight='bold')
ax6.set_xscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3)

# 7. Model Comparison: Test R²
ax7 = plt.subplot(3, 3, 7)
df_comparison = pd.DataFrame(comparison_results)
ax7.barh(df_comparison['Model'], df_comparison['Test R²'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'], 
         alpha=0.8, edgecolor='black', linewidth=1)
ax7.set_xlabel('Test R² Score', fontsize=10, fontweight='bold')
ax7.set_title('Model Comparison: Test R²', fontsize=11, fontweight='bold')
ax7.grid(axis='x', alpha=0.3)

# 8. Feature Coefficients Comparison
ax8 = plt.subplot(3, 3, 8)
models_for_coef = ['LassoLars', 'Lasso', 'Ridge']
coef_data = []

for model_name in models_for_coef:
    model = models_comparison[model_name]
    coef_data.append(model.coef_)

x = np.arange(len(feature_names))
width = 0.25
colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']

for i, (model_name, coef, color) in enumerate(zip(models_for_coef, coef_data, colors_bar)):
    ax8.bar(x + i*width, coef, width, label=model_name, alpha=0.8, color=color, edgecolor='black', linewidth=1)

ax8.set_xlabel('Features', fontsize=10, fontweight='bold')
ax8.set_ylabel('Coefficient Value', fontsize=10, fontweight='bold')
ax8.set_title('Feature Coefficients Comparison', fontsize=11, fontweight='bold')
ax8.set_xticks(x + width)
ax8.set_xticklabels(feature_names)
ax8.legend()
ax8.grid(axis='y', alpha=0.3)

# 9. Residuals Distribution Task 2
ax9 = plt.subplot(3, 3, 9)
ax9.hist(residuals_t2, bins=10, color='lightcoral', alpha=0.7, edgecolor='black')
ax9.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax9.set_xlabel('Residuals', fontsize=10, fontweight='bold')
ax9.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax9.set_title('Task 2: Residuals Distribution', fontsize=11, fontweight='bold')
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lassolars_fruit_data.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'lassolars_fruit_data.png'")

# =====================================================
# PREDICTIONS ON NEW SAMPLES
# =====================================================

print("\n" + "=" * 70)
print("PREDICTIONS ON NEW SAMPLES")
print("=" * 70)

# Task 1: Predict diameter from color
print(f"\nTask 1: Predicting Diameter from Color (LassoLars α={best_alpha_t1})")
print("-" * 70)

new_colors = ['green', 'red', 'yellow', 'purple']
print(f"\n{'Color':<12} {'Encoded':<10} {'Predicted Diameter':<20}")
print("-" * 42)

for color in new_colors:
    color_idx = color_encoder.transform([color])[0]
    sample = np.array([[color_idx]])
    sample_scaled = scaler_t1.transform(sample)
    pred_diameter = best_model_t1.predict(sample_scaled)[0]
    print(f"{color:<12} {color_idx:<10} {pred_diameter:<20.2f}")

# Task 2: Predict fruit type from color and diameter
print(f"\nTask 2: Predicting Fruit Type from Color & Diameter (LassoLars α={best_alpha_t2})")
print("-" * 70)

new_samples_t2 = [
    ('green', 3.0),
    ('red', 3.0),
    ('purple', 1.1),
    ('yellow', 3.2),
    ('green', 1.2)
]

print(f"\n{'Color':<12} {'Diameter':<10} {'Predicted Value':<20} {'Nearest Fruit':<15}")
print("-" * 57)

for color, diameter in new_samples_t2:
    color_idx = color_encoder.transform([color])[0]
    sample = np.array([[color_idx, diameter]])
    sample_scaled = scaler_t2.transform(sample)
    pred_value = best_model_t2_pred.predict(sample_scaled)[0]
    pred_fruit_idx = round(pred_value)
    if 0 <= pred_fruit_idx < len(fruit_encoder.classes_):
        pred_fruit = fruit_encoder.classes_[pred_fruit_idx]
    else:
        pred_fruit = fruit_encoder.classes_[np.argmin(np.abs(np.arange(len(fruit_encoder.classes_)) - pred_value))]
    print(f"{color:<12} {diameter:<10.2f} {pred_value:<20.4f} {pred_fruit:<15}")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("SUMMARY: LASSOLARS REGRESSION ON FRUIT DATASET")
print("=" * 70)

print(f"""
Dataset Information:
  - Total samples: {len(df)}
  - Fruits: {', '.join(fruit_encoder.classes_)}
  - Colors: {', '.join(color_encoder.classes_)}
  - Diameter range: [{df['diameter'].min():.2f}, {df['diameter'].max():.2f}]

Task 1: Predict Diameter from Color
  - Best alpha: {best_alpha_t1}
  - Test R² Score: {best_r2_t1:.4f}
  - Features: 1 (color_encoded)
  - Target: continuous (diameter)

Task 2: Predict Fruit Type from Color & Diameter
  - Best alpha: {best_alpha_t2}
  - Test R² Score: {best_r2_t2:.4f}
  - Features: 2 (color_encoded, diameter)
  - Target: categorical (fruit type, encoded as continuous)

LassoLars (Least Angle Regression with Lasso):
✓ Combines LARS algorithm efficiency with L1 regularization
✓ Efficient feature selection (sparse coefficients)
✓ Fast computation especially for high-dimensional data
✓ Good for interpretable models
✓ Automatic feature selection (some coefficients may be zero)

Key Hyperparameter:
  Alpha (α): Controls the strength of L1 regularization
  - Larger alpha: More features are zeroed out (sparser model)
  - Smaller alpha: More features are retained

Model Comparison Results:
  Best performing model (Task 2):
  - Model: {df_comparison.loc[df_comparison['Test R²'].idxmax(), 'Model']}
  - Test R²: {df_comparison['Test R²'].max():.4f}
""")

print("=" * 70)
