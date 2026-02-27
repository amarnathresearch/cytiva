"""
Simple CNN (Convolutional Neural Network) Example
This script demonstrates a basic CNN for image classification using MNIST dataset.
Uses TensorFlow/Keras for simplicity and clarity.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    print("\n" + "=" * 70)
    print("STEP 1: LOAD AND PREPROCESS DATA")
    print("=" * 70)
    
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    print(f"\nDataset Shapes:")
    print(f"  Training images: {X_train.shape}")
    print(f"  Training labels: {y_train.shape}")
    print(f"  Test images: {X_test.shape}")
    print(f"  Test labels: {y_test.shape}")
    
    # Normalize pixel values to 0-1 range
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape images to (samples, height, width, channels)
    # MNIST is 28x28 grayscale images
    X_train = X_train.reshape(-1, 28, 28, 1)
    X_test = X_test.reshape(-1, 28, 28, 1)
    
    print(f"\nAfter Preprocessing:")
    print(f"  Training images shape: {X_train.shape}")
    print(f"  Test images shape: {X_test.shape}")
    print(f"  Pixel value range: [{X_train.min():.2f}, {X_train.max():.2f}]")
    
    # Convert labels to one-hot encoding
    y_train_encoded = keras.utils.to_categorical(y_train, 10)
    y_test_encoded = keras.utils.to_categorical(y_test, 10)
    
    print(f"\nOne-hot encoded labels shape: {y_train_encoded.shape}")
    print(f"Number of classes: 10 (digits 0-9)")
    
    return X_train, X_test, y_train, y_test, y_train_encoded, y_test_encoded


def visualize_sample_images(X_train, y_train):
    """Visualize sample images from the dataset"""
    print("\n" + "=" * 70)
    print("STEP 2: VISUALIZE SAMPLE IMAGES")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    fig.suptitle('Sample MNIST Images', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        # Select random image
        idx = np.random.randint(0, len(X_train))
        ax.imshow(X_train[idx].reshape(28, 28), cmap='gray')
        ax.set_title(f'Label: {y_train[idx]}', fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("✓ Sample images visualization displayed")


def build_cnn_model():
    """Build a simple CNN model"""
    print("\n" + "=" * 70)
    print("STEP 3: BUILD CNN MODEL")
    print("=" * 70)
    
    model = models.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    # Print model summary
    print("\nModel Architecture:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✓ Model compiled successfully")
    print("  Optimizer: Adam")
    print("  Loss: Categorical Crossentropy")
    print("  Metrics: Accuracy")
    
    return model


def print_model_details(model):
    """Print detailed model information"""
    print("\n" + "=" * 70)
    print("MODEL ARCHITECTURE DETAILS")
    print("=" * 70)
    
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


def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    """Train the CNN model"""
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
        validation_split=0.1,
        verbose=1
    )
    
    print("\n✓ Model training completed")
    
    return history


def evaluate_model(model, X_test, y_test, y_test_orig):
    """Evaluate model performance"""
    print("\n" + "=" * 70)
    print("STEP 5: EVALUATE MODEL")
    print("=" * 70)
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_orig, y_pred)
    
    print(f"\nTest Set Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test_orig, y_pred, digits=4))
    
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


def visualize_predictions(X_test, y_test_orig, y_pred, y_pred_proba):
    """Visualize model predictions"""
    print("\n" + "=" * 70)
    print("STEP 7: VISUALIZE PREDICTIONS")
    print("=" * 70)
    
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    fig.suptitle('Sample Predictions on Test Set', fontsize=14, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        idx = np.random.randint(0, len(X_test))
        
        # Display image
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        
        # Get prediction info
        true_label = y_test_orig[idx]
        pred_label = y_pred[idx]
        confidence = y_pred_proba[idx][pred_label]
        
        # Color title based on correctness
        color = 'green' if true_label == pred_label else 'red'
        
        title = f'True: {true_label} | Pred: {pred_label}\nConf: {confidence:.2f}'
        ax.set_title(title, fontweight='bold', color=color, fontsize=9)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("✓ Predictions visualization displayed")


def visualize_confusion_matrix(y_test_orig, y_pred):
    """Visualize confusion matrix"""
    print("\n" + "=" * 70)
    print("STEP 8: VISUALIZE CONFUSION MATRIX")
    print("=" * 70)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test_orig, y_pred)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'},
                xticklabels=range(10), yticklabels=range(10), ax=ax,
                annot_kws={'size': 9, 'weight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax.set_title('Confusion Matrix - MNIST CNN Predictions', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("✓ Confusion matrix visualization displayed")


def visualize_incorrect_predictions(X_test, y_test_orig, y_pred):
    """Visualize incorrect predictions"""
    print("\n" + "=" * 70)
    print("STEP 9: VISUALIZE INCORRECT PREDICTIONS")
    print("=" * 70)
    
    # Find incorrect predictions
    incorrect_mask = y_test_orig != y_pred
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) > 0:
        num_display = min(10, len(incorrect_indices))
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(f'Incorrect Predictions (showing {num_display} of {len(incorrect_indices)})', 
                     fontsize=14, fontweight='bold')
        
        for i, ax in enumerate(axes.flat):
            if i < num_display:
                idx = incorrect_indices[i]
                ax.imshow(X_test[idx].reshape(28, 28), cmap='gray')
                ax.set_title(f'True: {y_test_orig[idx]} | Pred: {y_pred[idx]}', 
                           fontweight='bold', color='red')
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()
        print(f"✓ Incorrect predictions visualization displayed")
        print(f"  Total incorrect predictions: {len(incorrect_indices)} / {len(y_test_orig)}")
    else:
        print("✓ No incorrect predictions found! Perfect accuracy!")


def print_summary_statistics(history, accuracy, y_test_orig, y_pred):
    """Print summary statistics"""
    print("\n" + "=" * 70)
    print("FINAL SUMMARY - CNN MODEL PERFORMANCE")
    print("=" * 70)
    
    print(f"\nTraining Results:")
    print(f"  Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"  Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"  Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"  Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    print(f"\nTest Set Results:")
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Correct Predictions: {np.sum(y_test_orig == y_pred)} / {len(y_test_orig)}")
    print(f"  Incorrect Predictions: {np.sum(y_test_orig != y_pred)} / {len(y_test_orig)}")
    
    accuracy_per_class = []
    for digit in range(10):
        mask = y_test_orig == digit
        if mask.sum() > 0:
            class_accuracy = (y_pred[mask] == digit).sum() / mask.sum()
            accuracy_per_class.append(class_accuracy)
            print(f"  Accuracy for digit {digit}: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
    
    print(f"\nModel Insights:")
    print(f"  ✓ CNN effectively learns spatial features")
    print(f"  ✓ Convolutional layers capture local patterns")
    print(f"  ✓ Pooling reduces dimensionality and parameters")
    print(f"  ✓ Dropout prevents overfitting")


def main():
    """Main function"""
    print("\n" + "=" * 70)
    print("SIMPLE CNN (CONVOLUTIONAL NEURAL NETWORK) EXAMPLE")
    print("Dataset: MNIST (Handwritten Digits)")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = load_and_preprocess_data()
    
    # Step 2: Visualize sample images
    visualize_sample_images(X_train, y_train)
    
    # Step 3: Build CNN model
    model = build_cnn_model()
    
    # Print model details
    print_model_details(model)
    
    # Step 4: Train model
    history = train_model(model, X_train, y_train_enc, epochs=10, batch_size=32)
    
    # Step 5: Evaluate model
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test_enc, y_test)
    
    # Step 6: Visualize training history
    visualize_training_history(history)
    
    # Step 7: Visualize predictions
    visualize_predictions(X_test, y_test, y_pred, y_pred_proba)
    
    # Step 8: Visualize confusion matrix
    visualize_confusion_matrix(y_test, y_pred)
    
    # Step 9: Visualize incorrect predictions
    visualize_incorrect_predictions(X_test, y_test, y_pred)
    
    # Summary statistics
    accuracy = np.mean(y_test == y_pred)
    print_summary_statistics(history, accuracy, y_test, y_pred)
    
    print("\n" + "=" * 70)
    print("CNN Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
