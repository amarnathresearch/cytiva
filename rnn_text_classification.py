"""
Recurrent Neural Network (RNN) for Text Classification
This script demonstrates RNN/LSTM for sentiment classification using movie reviews.
Uses TensorFlow/Keras and IMDB dataset for clarity and practical learning.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuration
VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128


def load_and_preprocess_data():
    """Load and preprocess IMDB dataset for sentiment analysis"""
    print("\n" + "=" * 70)
    print("STEP 1: LOAD AND PREPROCESS DATA")
    print("=" * 70)
    
    # Load IMDB dataset
    print("\nLoading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)
    
    print(f"\nDataset Shapes:")
    print(f"  Training reviews: {X_train.shape[0]}")
    print(f"  Test reviews: {X_test.shape[0]}")
    print(f"  Vocabulary size: {VOCAB_SIZE}")
    
    # Display sample review
    print(f"\nSample Review Statistics:")
    print(f"  Min length: {len(min(X_train, key=len))} words")
    print(f"  Max length: {len(max(X_train, key=len))} words")
    print(f"  Mean length: {np.mean([len(x) for x in X_train]):.2f} words")
    
    # Pad sequences to fixed length
    print(f"\nPadding sequences to {MAX_SEQUENCE_LENGTH} words...")
    X_train_padded = pad_sequences(X_train, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test_padded = pad_sequences(X_test, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    print(f"  Training data shape: {X_train_padded.shape}")
    print(f"  Test data shape: {X_test_padded.shape}")
    
    print(f"\nLabel Distribution:")
    print(f"  Positive (1): {np.sum(y_train)} training, {np.sum(y_test)} test")
    print(f"  Negative (0): {len(y_train) - np.sum(y_train)} training, {len(y_test) - np.sum(y_test)} test")
    
    return X_train_padded, X_test_padded, y_train, y_test


def get_word_index():
    """Get word-to-index mapping"""
    word_index = imdb.get_word_index()
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
    
    return word_index, reverse_word_index


def decode_review(text, reverse_word_index):
    """Convert text indices back to words"""
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])


def visualize_sample_reviews(X_train, y_train, X_test, y_test):
    """Visualize sample reviews"""
    print("\n" + "=" * 70)
    print("STEP 2: VISUALIZE SAMPLE REVIEWS")
    print("=" * 70)
    
    word_index, reverse_word_index = get_word_index()
    
    print("\nSample Training Reviews:")
    for i in range(3):
        review = decode_review(X_train[i], reverse_word_index)
        sentiment = "POSITIVE" if y_train[i] == 1 else "NEGATIVE"
        print(f"\n  Review {i+1} ({sentiment}):")
        print(f"  {review[:200]}...")
    
    # Plot sequence length distribution
    train_lengths = [len(x) for x in X_train]
    test_lengths = [len(x) for x in X_test]
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.hist(train_lengths, bins=50, alpha=0.7, label='Training', color='blue', edgecolor='black')
    ax.axvline(MAX_SEQUENCE_LENGTH, color='red', linestyle='--', linewidth=2, label=f'Max Length ({MAX_SEQUENCE_LENGTH})')
    ax.axvline(np.mean(train_lengths), color='green', linestyle='--', linewidth=2, label=f'Mean Length ({np.mean(train_lengths):.1f})')
    
    ax.set_xlabel('Sequence Length (words)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Distribution of Review Lengths', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("\n✓ Sample reviews visualization displayed")


def build_rnn_model(model_type='lstm'):
    """Build RNN model for text classification"""
    print("\n" + "=" * 70)
    print("STEP 3: BUILD RNN MODEL")
    print("=" * 70)
    
    model_type = model_type.lower()
    print(f"\nBuilding {model_type.upper()} model...")
    
    model = models.Sequential([
        # Embedding layer
        layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
        
        # Recurrent layers
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(0.5),
        
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dropout(0.5),
        
        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✓ Model compiled successfully")
    print("  Optimizer: Adam")
    print("  Loss: Binary Crossentropy")
    print("  Metrics: Accuracy")
    
    return model


def print_model_details(model):
    """Print detailed model information"""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE DETAILS")
    print("=" * 70)
    
    # Build model if not already built
    if not model.built:
        model.build(input_shape=(None, MAX_SEQUENCE_LENGTH))
    
    total_params = model.count_params()
    print(f"\nTotal Parameters: {total_params:,}")
    
    print("\nLayer Details:")
    for i, layer in enumerate(model.layers, 1):
        print(f"\n  Layer {i}: {layer.name}")
        print(f"    Type: {layer.__class__.__name__}")
        try:
            output_shape = str(layer.output.shape)
            print(f"    Output shape: {output_shape}")
        except:
            print(f"    Output shape: Not available")
        print(f"    Parameters: {layer.count_params():,}")


def train_model(model, X_train, y_train, epochs=5, batch_size=64):
    """Train the RNN model"""
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN MODEL")
    print("=" * 70)
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Steps per epoch: {len(X_train) // batch_size}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=1
    )
    
    print("\n✓ Model training completed")
    
    return history


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATE MODEL")
    print("=" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], digits=4))
    
    return y_pred, y_pred_proba


def visualize_training_history(history):
    """Visualize training history"""
    print("\n" + "=" * 70)
    print("STEP 6: VISUALIZE TRAINING HISTORY")
    print("=" * 70)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot accuracy
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=11, fontweight='bold')
    axes[0].set_title('Model Accuracy Over Epochs', fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Loss', fontsize=11, fontweight='bold')
    axes[1].set_title('Model Loss Over Epochs', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("✓ Training history visualization displayed")


def visualize_roc_curve(y_test, y_pred_proba):
    """Visualize ROC curve"""
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZE ROC CURVE")
    print("=" * 70)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title('ROC Curve - Sentiment Classification', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print(f"✓ ROC curve visualization displayed (AUC: {roc_auc:.4f})")
    
    return roc_auc


def visualize_predictions(X_test, y_test, y_pred, y_pred_proba):
    """Visualize sample predictions"""
    print("\n" + "=" * 70)
    print("STEP 8: VISUALIZE SAMPLE PREDICTIONS")
    print("=" * 70)
    
    word_index, reverse_word_index = get_word_index()
    
    print("\nSample Predictions on Test Set:")
    correct_count = 0
    
    for i in range(5):
        idx = np.random.randint(0, len(X_test))
        
        review = decode_review(X_test[idx], reverse_word_index)
        true_label = "POSITIVE" if y_test[idx] == 1 else "NEGATIVE"
        pred_label = "POSITIVE" if y_pred[idx] == 1 else "NEGATIVE"
        confidence = y_pred_proba[idx][0]
        
        is_correct = y_test[idx] == y_pred[idx]
        status = "✓ CORRECT" if is_correct else "✗ INCORRECT"
        
        if is_correct:
            correct_count += 1
        
        print(f"\n  Prediction {i+1}: {status}")
        print(f"    True: {true_label} | Predicted: {pred_label} (Confidence: {confidence:.4f})")
        print(f"    Review: {review[:150]}...")
    
    print(f"\n  Sample accuracy: {correct_count}/5")


def visualize_confusion_matrix(y_test, y_pred):
    """Visualize confusion matrix"""
    print("\n" + "=" * 70)
    print("STEP 9: VISUALIZE CONFUSION MATRIX")
    print("=" * 70)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                ax=ax, annot_kws={'size': 12, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title('Confusion Matrix - Sentiment Classification', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN): {tn}")
    print(f"  False Positives (FP): {fp}")
    print(f"  False Negatives (FN): {fn}")
    print(f"  True Positives (TP): {tp}")
    
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n  Sensitivity (Recall): {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    
    print("✓ Confusion matrix visualization displayed")


def predict_custom_review(model, review_text):
    """Predict sentiment for custom review"""
    print("\n" + "=" * 70)
    print("STEP 10: PREDICT CUSTOM REVIEW")
    print("=" * 70)
    
    word_index, reverse_word_index = get_word_index()
    
    # Convert text to indices
    words = review_text.lower().split()
    indices = []
    for word in words:
        # Find word in index (reverse lookup)
        for key, val in word_index.items():
            if val == word:
                indices.append(key)
                break
    
    # Pad sequence
    review_array = np.array([indices])
    review_padded = pad_sequences(review_array, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    
    # Get prediction
    prediction = model.predict(review_padded, verbose=0)[0][0]
    sentiment = "POSITIVE" if prediction > 0.5 else "NEGATIVE"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    print(f"\nCustom Review Analysis:")
    print(f"  Text: \"{review_text}\"")
    print(f"  Predicted Sentiment: {sentiment}")
    print(f"  Confidence: {confidence:.4f}")
    
    return prediction


def print_summary_statistics(history, accuracy, y_test, y_pred, roc_auc):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - RNN TEXT CLASSIFICATION PERFORMANCE")
    print("=" * 70)
    
    print(f"\nTraining Results:")
    print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    print(f"\nTest Set Results:")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  ROC AUC Score: {roc_auc:.4f}")
    print(f"  Correct Predictions: {np.sum(y_test == y_pred)} / {len(y_test)}")
    print(f"  Incorrect Predictions: {np.sum(y_test != y_pred)} / {len(y_test)}")
    
    print(f"\nModel Insights:")
    print(f"  ✓ Embedding layer converts words to dense vectors")
    print(f"  ✓ Bidirectional LSTM captures context from both directions")
    print(f"  ✓ Dropout prevents overfitting during training")
    print(f"  ✓ Sigmoid activation for binary classification")
    print(f"  ✓ Model learns semantic relationships between words")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("RECURRENT NEURAL NETWORK (RNN) FOR TEXT CLASSIFICATION")
    print("Dataset: IMDB Movie Reviews (Sentiment Analysis)")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    
    # Step 2: Visualize sample reviews
    visualize_sample_reviews(X_train, y_train, X_test, y_test)
    
    # Step 3: Build RNN model
    model = build_rnn_model(model_type='lstm')
    
    # Print model details
    print_model_details(model)
    
    # Step 4: Train model
    history = train_model(model, X_train, y_train, epochs=5, batch_size=64)
    
    # Step 5: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # Step 6: Visualize training history
    visualize_training_history(history)
    
    # Step 7: Visualize ROC curve
    roc_auc = visualize_roc_curve(y_test, y_pred_proba)
    
    # Step 8: Visualize sample predictions
    visualize_predictions(X_test, y_test, y_pred, y_pred_proba)
    
    # Step 9: Visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
    
    # Step 10: Predict custom reviews
    custom_reviews = [
        "This movie was absolutely fantastic! I loved every minute of it.",
        "Terrible movie. Waste of time and money. Very disappointed.",
        "It was okay, nothing special but entertaining enough."
    ]
    
    print("\n" + "=" * 70)
    print("CUSTOM REVIEW PREDICTIONS")
    print("=" * 70)
    
    for review in custom_reviews:
        predict_custom_review(model, review)
    
    # Summary statistics
    accuracy = np.mean(y_test == y_pred)
    print_summary_statistics(history, accuracy, y_test, y_pred, roc_auc)
    
    print("\n" + "=" * 70)
    print("RNN Text Classification Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
