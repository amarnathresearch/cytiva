"""
Deep RNN Neural Network for Sentiment Analysis
This script demonstrates advanced sentiment analysis using deep RNN architecture
with LSTM, GRU, attention mechanisms, and multiple dense layers.
Dataset: IMDB Movie Reviews
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, \
    roc_curve, auc, precision_recall_curve, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 150
EMBEDDING_DIM = 128
LSTM_UNITS_1 = 128
LSTM_UNITS_2 = 64
LSTM_UNITS_3 = 32
DENSE_UNITS_1 = 256
DENSE_UNITS_2 = 128
DENSE_UNITS_3 = 64


def load_and_preprocess_data():
    """Load and preprocess IMDB dataset"""
    print("\n" + "=" * 70)
    print("STEP 1: LOAD AND PREPROCESS DATA")
    print("=" * 70)
    
    print("\nLoading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    print(f"\nDataset Information:")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    
    # Pad sequences
    print(f"\nPadding sequences to {MAX_SEQUENCE_LENGTH} words...")
    X_train_padded = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test_padded = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    print(f"  Training shape: {X_train_padded.shape}")
    print(f"  Test shape: {X_test_padded.shape}")
    
    # Dataset statistics
    train_lengths = [len(x) for x in X_train]
    print(f"\nSequence Length Statistics:")
    print(f"  Min: {min(train_lengths)}, Max: {max(train_lengths)}")
    print(f"  Mean: {np.mean(train_lengths):.2f}, Median: {np.median(train_lengths):.2f}")
    
    print(f"\nLabel Distribution:")
    print(f"  Positive: {np.sum(y_train)} train, {np.sum(y_test)} test")
    print(f"  Negative: {len(y_train)-np.sum(y_train)} train, {len(y_test)-np.sum(y_test)} test")
    
    # Limit training data for faster training
    X_train_padded = X_train_padded[:10000]
    y_train = y_train[:10000]
    X_test_padded = X_test_padded[:2500]
    y_test = y_test[:2500]
    
    print(f"\nLimited dataset for faster training:")
    print(f"  Training samples: {len(X_train_padded)}")
    print(f"  Test samples: {len(X_test_padded)}")
    
    return X_train_padded, X_test_padded, y_train, y_test


def build_deep_lstm_model():
    """Build deep LSTM model with multiple layers"""
    print("\n" + "=" * 70)
    print("STEP 2: BUILD DEEP LSTM MODEL")
    print("=" * 70)
    
    print("\nBuilding stacked LSTM architecture...")
    
    model = models.Sequential([
        # Embedding Layer
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        layers.Dropout(0.3),
        
        # First LSTM Block
        layers.Bidirectional(layers.LSTM(LSTM_UNITS_1, return_sequences=True)),
        layers.Dropout(0.3),
        
        # Second LSTM Block
        layers.Bidirectional(layers.LSTM(LSTM_UNITS_2, return_sequences=True)),
        layers.Dropout(0.3),
        
        # Third LSTM Block
        layers.Bidirectional(layers.LSTM(LSTM_UNITS_3, return_sequences=False)),
        layers.Dropout(0.3),
        
        # Deep Dense Layers
        layers.Dense(DENSE_UNITS_1, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(DENSE_UNITS_2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(DENSE_UNITS_3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output Layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    print("\nModel Architecture:")
    model.summary()
    
    # Compile with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    print("\n✓ Deep LSTM model compiled successfully")
    
    return model


def build_gru_model():
    """Build deep GRU model (alternative to LSTM)"""
    print("\n" + "=" * 70)
    print("BUILDING DEEP GRU MODEL (ALTERNATIVE)")
    print("=" * 70)
    
    print("\nBuilding stacked GRU architecture...")
    
    model = models.Sequential([
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.GRU(LSTM_UNITS_1, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.GRU(LSTM_UNITS_2, return_sequences=True)),
        layers.Dropout(0.3),
        
        layers.Bidirectional(layers.GRU(LSTM_UNITS_3, return_sequences=False)),
        layers.Dropout(0.3),
        
        layers.Dense(DENSE_UNITS_1, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(DENSE_UNITS_2, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(DENSE_UNITS_3, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✓ Deep GRU model compiled successfully")
    
    return model


def print_model_architecture(model):
    """Print detailed model architecture"""
    print("\n" + "=" * 70)
    print("DETAILED MODEL ARCHITECTURE")
    print("=" * 70)
    
    if not model.built:
        model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    traiable_params = sum([keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([keras.backend.count_params(w) for w in model.non_trainable_weights])
    
    print(f"Trainable Parameters: {traiable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    
    print("\nLayer-by-Layer Details:")
    for i, layer in enumerate(model.layers, 1):
        print(f"\n  Layer {i}: {layer.name}")
        print(f"    Type: {layer.__class__.__name__}")
        print(f"    Parameters: {layer.count_params():,}")
        if hasattr(layer, 'units'):
            print(f"    Units: {layer.units}")
        if hasattr(layer, 'activation'):
            print(f"    Activation: {layer.activation.__name__ if hasattr(layer.activation, '__name__') else layer.activation}")


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Train the deep RNN model"""
    print("\n" + "=" * 70)
    print("STEP 3: TRAIN DEEP RNN MODEL")
    print("=" * 70)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation split: 0.2")
    
    # Early stopping callback
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=2,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("\n✓ Model training completed")
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model on test set"""
    print("\n" + "=" * 70)
    print("STEP 4: EVALUATE MODEL")
    print("=" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = (np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_pred == 1)) if np.sum(y_pred == 1) > 0 else 0
    recall = (np.sum((y_pred == 1) & (y_test == 1)) / np.sum(y_test == 1)) if np.sum(y_test == 1) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], digits=4))
    
    return y_pred, y_pred_proba


def visualize_training_history(history):
    """Visualize training history with multiple metrics"""
    print("\n" + "=" * 70)
    print("STEP 5: VISUALIZE TRAINING HISTORY")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision and Recall
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].plot(history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 0].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
        axes[1, 0].set_ylabel('Score', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Precision and Recall Over Epochs', fontsize=12, fontweight='bold')
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate visualization (training/validation loss ratio)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    loss_ratio = np.array(val_loss) / np.array(train_loss)
    
    axes[1, 1].plot(loss_ratio, label='Val Loss / Train Loss', linewidth=2, color='purple')
    axes[1, 1].axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='Perfect (1.0)')
    axes[1, 1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Loss Ratio', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Overfitting Indicator (Val/Train Loss)', fontsize=12, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✓ Training history visualization displayed")


def visualize_roc_and_precision_recall(y_test, y_pred_proba):
    """Visualize ROC and Precision-Recall curves"""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZE ROC AND PRECISION-RECALL CURVES")
    print("=" * 70)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC Curve
    axes[0].plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    axes[0].set_title('ROC Curve', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    axes[1].plot(recall, precision, color='green', lw=2.5, label=f'PR Curve (AUC = {pr_auc:.4f})')
    axes[1].axhline(y=np.mean(y_test), color='red', linestyle='--', linewidth=1, label=f'Baseline ({np.mean(y_test):.2f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Precision', fontsize=11, fontweight='bold')
    axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print(f"✓ ROC and Precision-Recall curves displayed")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    
    return roc_auc, pr_auc


def visualize_confusion_matrix(y_test, y_pred):
    """Visualize confusion matrix"""
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZE CONFUSION MATRIX")
    print("=" * 70)
    
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title('Confusion Matrix - Deep RNN Sentiment Analysis', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives: {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives: {tp}")
    print(f"\nMetrics:")
    print(f"  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print("✓ Confusion matrix visualization displayed")


def visualize_prediction_confidence(y_test, y_pred_proba):
    """Visualize prediction confidence distribution"""
    print("\n" + "=" * 70)
    print("STEP 8: VISUALIZE PREDICTION CONFIDENCE")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correct vs Incorrect predictions
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    correct = y_pred_proba[y_test == y_pred].flatten()
    incorrect = y_pred_proba[y_test != y_pred].flatten()
    
    # Histogram of confidences
    axes[0].hist(correct, bins=30, alpha=0.7, label='Correct', color='green', edgecolor='black')
    axes[0].hist(incorrect, bins=30, alpha=0.7, label='Incorrect', color='red', edgecolor='black')
    axes[0].set_xlabel('Prediction Confidence', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('Confidence Distribution: Correct vs Incorrect', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot by true label
    confidence_data = [
        y_pred_proba[y_test == 0].flatten(),
        y_pred_proba[y_test == 1].flatten()
    ]
    
    bp = axes[1].boxplot(confidence_data, labels=['Negative (0)', 'Positive (1)'],
                         patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], ['lightcoral', 'lightgreen']):
        patch.set_facecolor(color)
    
    axes[1].set_ylabel('Prediction Confidence', fontsize=11, fontweight='bold')
    axes[1].set_title('Confidence Distribution by True Label', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    print("✓ Prediction confidence visualizations displayed")


def print_summary_statistics(history, y_test, y_pred, y_pred_proba, roc_auc, pr_auc):
    """Print comprehensive summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - DEEP RNN SENTIMENT ANALYSIS")
    print("=" * 70)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining Results:")
    print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")
    print(f"  PR AUC: {pr_auc:.4f}")
    print(f"  Correct Predictions: {np.sum(y_test == y_pred)} / {len(y_test)}")
    print(f"  Incorrect Predictions: {np.sum(y_test != y_pred)} / {len(y_test)}")
    
    print(f"\nModel Architecture Highlights:")
    print(f"  ✓ Deep Embedding Layer: Words → Dense Vectors ({EMBEDDING_DIM}D)")
    print(f"  ✓ Stacked LSTM Layers: 3 Bidirectional LSTM layers ({LSTM_UNITS_1}, {LSTM_UNITS_2}, {LSTM_UNITS_3} units)")
    print(f"  ✓ Deep Dense Layers: 3 fully connected layers ({DENSE_UNITS_1}, {DENSE_UNITS_2}, {DENSE_UNITS_3} units)")
    print(f"  ✓ Batch Normalization: Stabilizes training")
    print(f"  ✓ Dropout Regularization: Prevents overfitting")
    print(f"  ✓ Bidirectional Processing: Captures context from both directions")
    
    print(f"\nKey Insights:")
    print(f"  • Deep architecture learns hierarchical sentiment features")
    print(f"  • Bidirectional LSTM captures contextual information")
    print(f"  • Dropout and BatchNorm improve generalization")
    print(f"  • Multiple dense layers enable complex decision boundaries")
    print(f"  • The model learns semantic relationships in reviews")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("DEEP RNN NEURAL NETWORK FOR SENTIMENT ANALYSIS")
    print("Dataset: IMDB Movie Reviews")
    print("=" * 70)
    
    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Step 2: Build model
    model = build_deep_lstm_model()
    
    # Print architecture
    print_model_architecture(model)
    
    # Step 3: Train model
    history = train_model(model, X_train, y_train, epochs=8, batch_size=32)
    
    # Step 4: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Step 5: Visualize training history
    visualize_training_history(history)
    
    # Step 6: ROC and PR curves
    roc_auc, pr_auc = visualize_roc_and_precision_recall(y_test, y_pred_proba)
    
    # Step 7: Confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
    
    # Step 8: Prediction confidence
    visualize_prediction_confidence(y_test, y_pred_proba)
    
    # Summary statistics
    print_summary_statistics(history, y_test, y_pred, y_pred_proba, roc_auc, pr_auc)
    
    print("\n" + "=" * 70)
    print("Deep RNN Sentiment Analysis Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
