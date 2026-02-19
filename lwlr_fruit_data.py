import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# LOCALLY WEIGHTED LINEAR REGRESSION ON FRUIT DATASET
# =====================================================

class LocallyWeightedRegression:
    """
    Locally Weighted Linear Regression (LWLR)
    Predicts fruit properties based on local weighted training data
    """
    
    def __init__(self, tau=1.0):
        """
        Parameters:
        tau: bandwidth parameter (controls the width of the weight function)
        """
        self.tau = tau
        self.X_train = None
        self.y_train = None
        
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
        Store training data (LWLR is lazy learning)
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
            try:
                XT = self.X_train.T
                theta = np.linalg.inv(XT @ W @ self.X_train) @ XT @ W @ self.y_train
                prediction = query_point @ theta
                predictions.append(prediction[0])
            except np.linalg.LinAlgError:
                # Fallback to least squares
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
# LOAD AND PREPARE FRUIT DATASET
# =====================================================

print("=" * 70)
print("LOCALLY WEIGHTED REGRESSION - FRUIT DATASET ANALYSIS")
print("=" * 70)

# Load fruit data
df = pd.read_csv('/opt/cytiva/fruit_data_augmented.csv')

print(f"\n{'Dataset Information':}")
print(f"  Total samples: {len(df)}")
print(f"  Shape: {df.shape}")
print(f"\nFirst 10 rows:")
print(df.head(10))
print(f"\nDataset info:")
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
# REGRESSION TASK 1: PREDICT DIAMETER FROM COLOR
# =====================================================

print("\n" + "=" * 70)
print("TASK 1: PREDICT DIAMETER FROM COLOR (Continuous Regression)")
print("=" * 70)

# Prepare features and target
X_task1 = df[['color_encoded']].values
y_task1 = df['diameter'].values

# Split data
X_train_t1, X_test_t1, y_train_t1, y_test_t1 = train_test_split(
    X_task1, y_task1, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training set: {X_train_t1.shape[0]} samples")
print(f"  Testing set: {X_test_t1.shape[0]} samples")

# Train models with different tau values
tau_values = [0.1, 0.5, 1.0, 2.0]
models_t1 = {}

print(f"\nTraining LWLR models...")
for tau in tau_values:
    lwlr = LocallyWeightedRegression(tau=tau)
    lwlr.fit(X_train_t1, y_train_t1)
    models_t1[tau] = lwlr

# Evaluate Task 1
print(f"\n{'Tau':<8} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12}")
print("-" * 56)

best_tau_t1 = None
best_r2_t1 = -np.inf
results_t1 = []

for tau in tau_values:
    lwlr = models_t1[tau]
    y_pred_train = lwlr.predict(X_train_t1)
    y_pred_test = lwlr.predict(X_test_t1)
    
    train_r2 = r2_score(y_train_t1, y_pred_train)
    test_r2 = r2_score(y_test_t1, y_pred_test)
    train_mse = mean_squared_error(y_train_t1, y_pred_train)
    test_mse = mean_squared_error(y_test_t1, y_pred_test)
    
    print(f"{tau:<8} {train_r2:<12.4f} {test_r2:<12.4f} {train_mse:<12.4f} {test_mse:<12.4f}")
    
    results_t1.append({
        'Tau': tau,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })
    
    if test_r2 > best_r2_t1:
        best_r2_t1 = test_r2
        best_tau_t1 = tau

print(f"\n✓ Best tau for Task 1: {best_tau_t1} (Test R² = {best_r2_t1:.4f})")

# =====================================================
# REGRESSION TASK 2: PREDICT FRUIT TYPE FROM COLOR & DIAMETER
# =====================================================

print("\n" + "=" * 70)
print("TASK 2: PREDICT FRUIT TYPE FROM COLOR & DIAMETER (Classification via Regression)")
print("=" * 70)

# Prepare features and target (encode fruit as 0, 1, 2, etc.)
X_task2 = df[['color_encoded', 'diameter']].values
y_task2 = df['fruit_encoded'].values

# Split data
X_train_t2, X_test_t2, y_train_t2, y_test_t2 = train_test_split(
    X_task2, y_task2, test_size=0.2, random_state=42
)

print(f"\nData split:")
print(f"  Training set: {X_train_t2.shape[0]} samples")
print(f"  Testing set: {X_test_t2.shape[0]} samples")
print(f"  Features: 2 (color_encoded, diameter)")

# Train models with different tau values
models_t2 = {}

print(f"\nTraining LWLR models for classification...")
for tau in tau_values:
    lwlr = LocallyWeightedRegression(tau=tau)
    lwlr.fit(X_train_t2, y_train_t2)
    models_t2[tau] = lwlr

# Evaluate Task 2
print(f"\n{'Tau':<8} {'Train R²':<12} {'Test R²':<12} {'Train MSE':<12} {'Test MSE':<12}")
print("-" * 56)

best_tau_t2 = None
best_r2_t2 = -np.inf
results_t2 = []

for tau in tau_values:
    lwlr = models_t2[tau]
    y_pred_train = lwlr.predict(X_train_t2)
    y_pred_test = lwlr.predict(X_test_t2)
    
    train_r2 = r2_score(y_train_t2, y_pred_train)
    test_r2 = r2_score(y_test_t2, y_pred_test)
    train_mse = mean_squared_error(y_train_t2, y_pred_train)
    test_mse = mean_squared_error(y_test_t2, y_pred_test)
    
    print(f"{tau:<8} {train_r2:<12.4f} {test_r2:<12.4f} {train_mse:<12.4f} {test_mse:<12.4f}")
    
    results_t2.append({
        'Tau': tau,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'Train MSE': train_mse,
        'Test MSE': test_mse
    })
    
    if test_r2 > best_r2_t2:
        best_r2_t2 = test_r2
        best_tau_t2 = tau

print(f"\n✓ Best tau for Task 2: {best_tau_t2} (Test R² = {best_r2_t2:.4f})")

# =====================================================
# VISUALIZATIONS
# =====================================================

fig = plt.figure(figsize=(18, 14))

# Task 1: Diameter Prediction
print("\n" + "=" * 70)
print("GENERATING VISUALIZATIONS...")
print("=" * 70)

best_model_t1 = models_t1[best_tau_t1]
y_pred_test_t1 = best_model_t1.predict(X_test_t1)

# 1. Task 1: Predicted vs Actual
ax1 = plt.subplot(3, 3, 1)
ax1.scatter(y_test_t1, y_pred_test_t1, alpha=0.7, s=100, color='blue', edgecolors='black', linewidth=1)
min_val = min(y_test_t1.min(), y_pred_test_t1.min())
max_val = max(y_test_t1.max(), y_pred_test_t1.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax1.set_xlabel('Actual Diameter', fontsize=10, fontweight='bold')
ax1.set_ylabel('Predicted Diameter', fontsize=10, fontweight='bold')
ax1.set_title(f'Task 1: Diameter Prediction (tau={best_tau_t1})', fontsize=11, fontweight='bold')
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

# 3. Task 1: R² Comparison
ax3 = plt.subplot(3, 3, 3)
df_t1 = pd.DataFrame(results_t1)
x_pos = np.arange(len(tau_values))
width = 0.35
ax3.bar(x_pos - width/2, df_t1['Train R²'], width, label='Train R²', alpha=0.8, color='skyblue')
ax3.bar(x_pos + width/2, df_t1['Test R²'], width, label='Test R²', alpha=0.8, color='coral')
ax3.set_xlabel('Tau Value', fontsize=10, fontweight='bold')
ax3.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax3.set_title('Task 1: R² Comparison', fontsize=11, fontweight='bold')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([str(t) for t in tau_values])
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Task 2: Fruit Classification via Regression
best_model_t2 = models_t2[best_tau_t2]
y_pred_test_t2 = best_model_t2.predict(X_test_t2)

# 4. Task 2: Predicted vs Actual
ax4 = plt.subplot(3, 3, 4)
ax4.scatter(y_test_t2, y_pred_test_t2, alpha=0.7, s=100, color='purple', edgecolors='black', linewidth=1)
min_val2 = min(y_test_t2.min(), y_pred_test_t2.min())
max_val2 = max(y_test_t2.max(), y_pred_test_t2.max())
ax4.plot([min_val2, max_val2], [min_val2, max_val2], 'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Fruit Type', fontsize=10, fontweight='bold')
ax4.set_ylabel('Predicted Fruit Type', fontsize=10, fontweight='bold')
ax4.set_title(f'Task 2: Fruit Classification (tau={best_tau_t2})', fontsize=11, fontweight='bold')
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

# 6. Task 2: R² Comparison
ax6 = plt.subplot(3, 3, 6)
df_t2 = pd.DataFrame(results_t2)
ax6.bar(x_pos - width/2, df_t2['Train R²'], width, label='Train R²', alpha=0.8, color='lightgreen')
ax6.bar(x_pos + width/2, df_t2['Test R²'], width, label='Test R²', alpha=0.8, color='salmon')
ax6.set_xlabel('Tau Value', fontsize=10, fontweight='bold')
ax6.set_ylabel('R² Score', fontsize=10, fontweight='bold')
ax6.set_title('Task 2: R² Comparison', fontsize=11, fontweight='bold')
ax6.set_xticks(x_pos)
ax6.set_xticklabels([str(t) for t in tau_values])
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Dataset Scatter: Color vs Diameter
ax7 = plt.subplot(3, 3, 7)
colors_map = {0: 'green', 1: 'purple', 2: 'red', 3: 'yellow'}
color_names = color_encoder.classes_

for color_idx, color_name in enumerate(color_encoder.classes_):
    mask = df['color_encoded'] == color_idx
    ax7.scatter(df[mask]['color_encoded'], df[mask]['diameter'], 
               label=color_name, s=100, alpha=0.7, edgecolors='black', linewidth=1)

ax7.set_xlabel('Color (Encoded)', fontsize=10, fontweight='bold')
ax7.set_ylabel('Diameter', fontsize=10, fontweight='bold')
ax7.set_title('Dataset: Color vs Diameter', fontsize=11, fontweight='bold')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. Residuals Distribution Task 1
ax8 = plt.subplot(3, 3, 8)
ax8.hist(residuals_t1, bins=10, color='lightblue', alpha=0.7, edgecolor='black')
ax8.axvline(x=0, color='r', linestyle='--', linewidth=2)
ax8.set_xlabel('Residuals', fontsize=10, fontweight='bold')
ax8.set_ylabel('Frequency', fontsize=10, fontweight='bold')
ax8.set_title('Task 1: Residuals Distribution', fontsize=11, fontweight='bold')
ax8.grid(axis='y', alpha=0.3)

# 9. Fruit Type Distribution
ax9 = plt.subplot(3, 3, 9)
fruit_counts = df['target'].value_counts()
ax9.bar(fruit_counts.index, fruit_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'], 
        alpha=0.7, edgecolor='black', linewidth=1)
ax9.set_xlabel('Fruit Type', fontsize=10, fontweight='bold')
ax9.set_ylabel('Count', fontsize=10, fontweight='bold')
ax9.set_title('Fruit Distribution in Dataset', fontsize=11, fontweight='bold')
ax9.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lwlr_fruit_data.png', dpi=100, bbox_inches='tight')
plt.show()

print("✓ Visualization saved as 'lwlr_fruit_data.png'")

# =====================================================
# PREDICTIONS ON NEW SAMPLES
# =====================================================

print("\n" + "=" * 70)
print("PREDICTIONS ON NEW SAMPLES")
print("=" * 70)

# Task 1: Predict diameter from color
print(f"\nTask 1: Predicting Diameter from Color")
print("-" * 70)

new_colors = ['green', 'red', 'yellow', 'purple']
print(f"\n{'Color':<12} {'Encoded':<10} {'Predicted Diameter':<20}")
print("-" * 42)

for color in new_colors:
    color_idx = color_encoder.transform([color])[0]
    sample = np.array([[color_idx]])
    pred_diameter = best_model_t1.predict(sample)[0]
    print(f"{color:<12} {color_idx:<10} {pred_diameter:<20.2f}")

# Task 2: Predict fruit type from color and diameter
print(f"\nTask 2: Predicting Fruit Type from Color & Diameter")
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
    pred_value = best_model_t2.predict(sample)[0]
    pred_fruit_idx = round(pred_value)
    if pred_fruit_idx < len(fruit_encoder.classes_):
        pred_fruit = fruit_encoder.classes_[pred_fruit_idx]
    else:
        pred_fruit = fruit_encoder.classes_[-1]
    print(f"{color:<12} {diameter:<10.2f} {pred_value:<20.4f} {pred_fruit:<15}")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 70)
print("SUMMARY: LOCALLY WEIGHTED REGRESSION ON FRUIT DATASET")
print("=" * 70)

print(f"""
Dataset Information:
  - Total samples: {len(df)}
  - Fruits: {', '.join(fruit_encoder.classes_)}
  - Colors: {', '.join(color_encoder.classes_)}
  - Diameter range: [{df['diameter'].min():.2f}, {df['diameter'].max():.2f}]

Task 1: Predict Diameter from Color
  - Best tau: {best_tau_t1}
  - Test R² Score: {best_r2_t1:.4f}
  - Features: 1 (color)
  - Target: continuous (diameter)
  
Task 2: Predict Fruit Type from Color & Diameter
  - Best tau: {best_tau_t2}
  - Test R² Score: {best_r2_t2:.4f}
  - Features: 2 (color, diameter)
  - Target: categorical (fruit type, encoded as continuous)

Locally Weighted Regression (LWLR) on Fruit Data:
✓ Non-parametric approach works well with small datasets
✓ Captures local relationships in the data
✓ Good for both regression and classification (via regression) tasks
✓ Flexible with respect to non-linear patterns
""")

print("=" * 70)
